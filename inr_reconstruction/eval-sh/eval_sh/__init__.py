from typing import NamedTuple, Union
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class EvalSH(nn.Module):
    def init(self):
        super().__init__()

    def forward(self, coordinates, active_deg, max_coeffs, 
                sh_coefficients, rx_pos):
        return EvaluateSH.apply(coordinates, active_deg, max_coeffs, 
                                sh_coefficients, rx_pos)

class EvaluateSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, active_deg, max_coeffs, 
                rx_pos, sh_coefficients):

        num_channels = sh_coefficients.shape[-1]

        out_sig = torch.zeros((coordinates.shape[0], num_channels),
                          dtype=sh_coefficients.dtype, device=sh_coefficients.device)
        
        
        _C.evalSH_forward(coordinates.contiguous(),
                          sh_coefficients.contiguous(), 
                          active_deg, max_coeffs,
                          rx_pos, out_sig)
        

        ctx.coordinates = coordinates
        ctx.num_channels = num_channels
        ctx.max_coeffs = max_coeffs
        ctx.active_deg = active_deg
        ctx.rx_pos = rx_pos

        return out_sig
    
    @staticmethod
    def backward(ctx, grad_sig_out):

        coordinates = ctx.coordinates
        num_channels = ctx.num_channels
        max_coeffs = ctx.max_coeffs
        active_deg = ctx.active_deg
        rx_pos = ctx.rx_pos

        grad_SH_grid = torch.zeros(
            (coordinates.shape[0] * max_coeffs, num_channels),
            dtype=grad_sig_out.dtype,
            device=grad_sig_out.device)

        
        _C.evalSH_backward(grad_sig_out.contiguous(),
                           coordinates.contiguous(),
                           active_deg,
                           max_coeffs,
                           rx_pos,
                           grad_SH_grid)
        
        # Return gradients in the same shape as forward inputs
        return (*[None]*4, grad_SH_grid.view(-1, num_channels))