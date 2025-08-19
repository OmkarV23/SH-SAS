// sh_bindings.cpp
// ----------------
// C++ ⇄ Python glue for the evalSH CUDA kernel.

#include <torch/extension.h>
#include <c10/util/Optional.h>
namespace py = pybind11;

// ────────────────────────────────────────────────────────────────
// Forward declaration
// (the implementation lives in sh_eval.cu)
// ────────────────────────────────────────────────────────────────
void evalSH_forward(
    const torch::Tensor& coordinates,
    const torch::Tensor& sh_coefficients,
    int                 active_deg,
    int                  max_coeffs,
    const torch::Tensor& rx_pos,
    torch::Tensor&       output);

void evalSH_backward(
    const torch::Tensor&  grad_out,
    const torch::Tensor&  coordinates,
    int                 active_deg,
    int                  max_coeffs,
    const torch::Tensor&  rx_pos,
    torch::Tensor&        grad_sh);


// ────────────────────────────────────────────────────────────────
// PyBind11 module
// TORCH_EXTENSION_NAME is set automatically by torch.utils.cpp_extension
// to the correct sub-module name ("_C").
// ────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evalSH_forward", &evalSH_forward, "Evaluate spherical harmonics forward pass");
    m.def("evalSH_backward", &evalSH_backward, "Evaluate spherical harmonics backward pass");
}