from typing import Callable, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from torch.autograd import Function
from torch.amp import custom_bwd, custom_fwd

from sas_utils import safe_normalize
from eval_sh import EvalSH

class Sin(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.sin(x)

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

class MLP(torch.nn.Module):
    '''Use Relu for density and Sine for radiance'''
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, activation=nn.ReLU):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.act = activation()

        net = []
        for l in range(num_layers):
            # Keep bias false
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = self.act(x)
            # x = self.act(x)
        return x

class Network(nn.Module):

    def __init__(
        self,
        num_dim: int = 3,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        max_SH_degree: int = 3,
        density_eps: float = 1e-9,
        device: Union[torch.device, str] = 'cuda',
        aabb: Union[torch.Tensor, List[float], None] = None,
        mlp_dim: int = 32,
        mlp_num_layers: int = 2,
        num_channels: int = 2
    ) -> None:
        super().__init__()

        self.num_dim = num_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size      
        # self.n_output_dim = num_channels # Number of output channels (e.g., 2 for complex numbers)
        self.n_output_dim = 2
        self.max_SH_degree = max_SH_degree
        self.density_eps = density_eps
        self.max_coeffs = (self.max_SH_degree + 1) ** 2
        
        self.device = device

        if aabb is not None:
            if not isinstance(aabb, torch.Tensor):
                aabb = torch.tensor(aabb, dtype=torch.float32)
            self.register_buffer("aabb", aabb)
        else:
            self.aabb = None
    
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        self.encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32,
        )

        self.mlp_base = MLP(dim_in=self.encoder.n_output_dims,
                                            dim_out=self.max_coeffs*self.n_output_dim,
                                            dim_hidden=mlp_dim,
                                            num_layers=mlp_num_layers).to(self.device)
        
    def query_density(self, x):

        x = self.encoder(x)
        SH_coeffs = self.mlp_base(x)

        SH_coeffs = SH_coeffs.reshape(-1, self.max_coeffs, self.n_output_dim)  # (Re, Im) for each SH coefficient

        # get the zero-th coefficient (density)
        density = 0.28209479177387814 * SH_coeffs[:,0,:]


        density = torch.complex(real=density[..., 0].float(), imag=density[..., 1].float())
        # density = torch.abs(density)

        if self.max_SH_degree == 0:
            sh_coeffs = None
        else:
            sh_coeffs = SH_coeffs.reshape(-1, 2) # CUDA kernel needs (num_points * max_coeffs, 2) shape

        return {"sigma": density, "sh_coeffs": sh_coeffs}


    def _query_scatter(self, coordinates, active_deg, rx_pos, sh_coefficients):
        """
        Query the scattering coefficients at given coordinates.
        :param coordinates: Tensor of shape (N, 3) with 3D coordinates.
        :param active_deg: Active SH degree.
        :param rx_pos: Receiver position.
        :param sh_coefficients: SH coefficients tensor.
        :return: Complex scattering coefficients.
        """
        eval_sh = EvalSH()
        sh_coefficients = sh_coefficients.float() # always use float for SH coefficients
        out_sig = eval_sh(coordinates, active_deg, self.max_coeffs, rx_pos, sh_coefficients)
        
        out_sig = out_sig.reshape(-1, 2)
        return torch.complex(out_sig[..., 0].float(), out_sig[..., 1].float())


    def forward(self, positions, rx_pos, active_deg=None, compute_normals=True):

        positions = positions.reshape(-1, self.num_dim)
        B = positions.shape[0]
        device = positions.device
        sigma = torch.zeros(B, dtype=torch.complex64, device=device)
        scatter_c = torch.zeros(B, dtype=torch.complex64, device=device)
        sh_coeffs = torch.zeros(B, self.max_coeffs, 2, dtype=torch.float32, device=device)
        
        normals = torch.zeros(B, 3, dtype=torch.float32, device=device) if compute_normals else None

        if self.aabb is None:
            selector = torch.ones(B, dtype=torch.bool, device=device)
            aabb_min = aabb_max = None
            scale = torch.tensor(1.0, device=device)
        else:
            aabb_min, aabb_max = self.aabb.view(2, 3)
            scale = (aabb_max - aabb_min).clamp_min(1e-6)
            norm_pos = (positions - aabb_min) / scale
            selector = ((norm_pos >= 0.) & (norm_pos <= 1.)).all(dim=-1)

        if not selector.any():
            return {"sigma": sigma, "scatterers_to": scatter_c, "normals": normals, "sh_coeffs": sh_coeffs}  # Explicit early return for empty

        # Masked compute
        selected_positions = positions[selector].float()
        pos_w = selected_positions.detach().clone() if compute_normals else selected_positions  # Clone only if needed
        if compute_normals:
            pos_w.requires_grad_(True)
        pos_n = (pos_w - aabb_min) / scale if self.aabb is not None else pos_w

        out = self.query_density(pos_n)
        sig = out["sigma"]
        sh = out["sh_coeffs"]
        if sh is not None:
            if active_deg is None:
                active_deg = self.max_SH_degree
            scat = self._query_scatter(pos_w, active_deg, rx_pos, sh)
            sh_coeffs[selector] = sh.reshape(-1, self.max_coeffs, 2)
        else:
            scat = sig
            sh_coeffs = None
        if compute_normals:
            g_local = torch.autograd.grad(sig.abs().sum(), pos_n, create_graph=True)[0]
            g_world = g_local / scale  # Adjust for world space
            n_w = -safe_normalize(g_world).float()
            n_w[torch.isnan(n_w)] = 0
            normals[selector] = n_w

        sigma[selector] = sig
        scatter_c[selector] = scat
        

        return {"sigma": sigma, "scatterers_to": scatter_c, "normals": normals, "sh_coeffs": sh_coeffs}