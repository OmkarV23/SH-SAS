#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <glm/vec3.hpp>               // glm::vec3
#include <glm/vec2.hpp>               // glm::vec2
#include <glm/geometric.hpp>          // glm::length, normalize
#include <torch/types.h>
#include <torch/extension.h>

namespace py = pybind11;

////////////////////////////////////////////////////////////////
//  Pre-computed SH constants (up to l = 3).                  //
////////////////////////////////////////////////////////////////
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__global__ void eval_sh_forward(
        int n,
        int active_deg,
        int max_coeffs,
        const glm::vec3* __restrict__ voxels,   // [N]
        const float* __restrict__ cam_pos,      // single 3-vector
        const float* __restrict__ shs,          // [N,max_coeffs,2] flattened
        glm::vec2* __restrict__ out)            // [N]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    //------------------------------------------------------------------
    glm::vec3 dir = glm::normalize(voxels[idx] - glm::vec3(cam_pos[0], cam_pos[1], cam_pos[2]));

    const glm::vec2* sh = reinterpret_cast<const glm::vec2*>(shs) +
                          idx * max_coeffs;

    float  x=dir.x, y=dir.y, z=dir.z;
    float  xx=x*x, yy=y*y, zz=z*z,  xy=x*y, yz=y*z, xz=x*z;

    glm::vec2 r = SH_C0 * sh[0];

    if (active_deg > 0){
        r = r - SH_C1*y * sh[1] + SH_C1*z * sh[2] - SH_C1*x * sh[3];
    }

    if (active_deg > 1){
    r +=  SH_C2[0]*xy              * sh[4]
        + SH_C2[1]*yz              * sh[5]
        + SH_C2[2]*(2*zz-xx-yy)    * sh[6]
        + SH_C2[3]*xz              * sh[7]
        + SH_C2[4]*(xx-yy)         * sh[8];
    }

    if (active_deg > 2){
    r +=  SH_C3[0]*y*(3*xx-yy)             * sh[ 9]
        + SH_C3[1]*xy*z                    * sh[10]
        + SH_C3[2]*y*(4*zz-xx-yy)          * sh[11]
        + SH_C3[3]*z*(2*zz-3*(xx+yy))      * sh[12]
        + SH_C3[4]*x*(4*zz-xx-yy)          * sh[13]
        + SH_C3[5]*z*(xx-yy)               * sh[14]
        + SH_C3[6]*x*(xx-3*yy)             * sh[15];
    }
    out[idx] = r;
}

__global__ void eval_sh_backward(
        int n,
        int active_deg,
        int max_coeffs,
        const glm::vec3* __restrict__ voxels,   // [N]
        const float* __restrict__ cam_pos,
        const glm::vec2* __restrict__ grad_out, // [N]  dL/dR
        glm::vec2* __restrict__ grad_sh)        // [N,max_coeffs]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    glm::vec3 dir = glm::normalize(voxels[idx] - glm::vec3(cam_pos[0], cam_pos[1], cam_pos[2]));
    float  x=dir.x, y=dir.y, z=dir.z;
    float  xx=x*x, yy=y*y, zz=z*z,  xy=x*y, yz=y*z, xz=x*z;

    glm::vec2 dL = grad_out[idx];
    glm::vec2* g = grad_sh + idx * max_coeffs;   // slice for this voxel

    // l = 0
    g[0] = dL * SH_C0;

    // l = 1
    if (active_deg > 0){
    g[1] = dL * (-SH_C1 * y);
    g[2] = dL * ( SH_C1 * z);
    g[3] = dL * (-SH_C1 * x);
    }

    // l = 2
    if (active_deg > 1){
    g[4] = dL * (SH_C2[0] * xy);
    g[5] = dL * (SH_C2[1] * yz);
    g[6] = dL * (SH_C2[2] * (2*zz-xx-yy));
    g[7] = dL * (SH_C2[3] * xz);
    g[8] = dL * (SH_C2[4] * (xx-yy));   
    }

    // l = 3
    if (active_deg > 2){
    g[ 9] = dL * (SH_C3[0] * y*(3*xx-yy));
    g[10] = dL * (SH_C3[1] * xy*z);
    g[11] = dL * (SH_C3[2] * y*(4*zz-xx-yy));
    g[12] = dL * (SH_C3[3] * z*(2*zz-3*(xx+yy)));
    g[13] = dL * (SH_C3[4] * x*(4*zz-xx-yy));
    g[14] = dL * (SH_C3[5] * z*(xx-yy));
    g[15] = dL * (SH_C3[6] * x*(xx-3*yy));
    }
}


void evalSH_forward(
    const torch::Tensor& coordinates,
    const torch::Tensor& sh_coefficients,
    int                 active_deg,
    int                  max_coeffs,
    const torch::Tensor& rx_pos,
    torch::Tensor&       output)
{
    TORCH_CHECK(coordinates.is_cuda() && sh_coefficients.is_cuda()
                && rx_pos.is_cuda() && output.is_cuda(),
                "All tensors must reside on the same CUDA device.");

    int n   = coordinates.size(0);
    dim3 blk(256), grd((n + blk.x - 1)/blk.x);

    c10::cuda::CUDAGuard guard(coordinates.device());
        
    eval_sh_forward<<<grd, blk>>>(
        n, active_deg, max_coeffs,
        (glm::vec3*)coordinates.contiguous().data_ptr<float>(),
        rx_pos.data_ptr<float>(),
        sh_coefficients.data_ptr<float>(),
        (glm::vec2*)output.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "evalSH kernel launch failed: ",
                cudaGetErrorString(err));
}

void evalSH_backward(const torch::Tensor&  grad_out,
                     const torch::Tensor&  coordinates,
                     int                 active_deg,
                     int                  max_coeffs,
                     const torch::Tensor&  rx_pos,
                     torch::Tensor&        grad_sh)
{
    TORCH_CHECK(coordinates.is_cuda() && grad_out.is_cuda() &&
                rx_pos.is_cuda() && grad_sh.is_cuda(),
                "All tensors must be CUDA.");

    int n = coordinates.size(0);
    dim3 blk(256), grd((n + blk.x - 1)/blk.x);

    c10::cuda::CUDAGuard guard(coordinates.device());

    eval_sh_backward<<<grd, blk>>>(
        n, active_deg, max_coeffs,
        (glm::vec3*)coordinates.contiguous().data_ptr<float>(),
        rx_pos.data_ptr<float>(),
        (glm::vec2*)grad_out.data_ptr<float>(),
        (glm::vec2*)grad_sh.data_ptr<float>());


    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "evalSH backward kernel launch failed: ",
                cudaGetErrorString(err));
}