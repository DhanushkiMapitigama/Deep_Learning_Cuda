#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void linear_forward_kernel(const float* input, const float* weights, const float* bias, float* output,
                                  int batch_size, int in_features, int out_features) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < batch_size * out_features; i += total_threads) {
        int row = i / out_features; // Batch index
        int col = i % out_features; // Output feature index

        float sum = 0.0;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weights[col * in_features + k];
        }
        output[i] = sum + bias[col];
    }
}


__global__ void linear_backward_kernel(const float* grad_out, const float* input, const float* weight,
                                       float* grad_input, float* grad_weight, float* grad_bias,
                                       int B, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * in_features) return;

    int b = idx / in_features;
    int i = idx % in_features;

    float grad = 0.0;
    for (int o = 0; o < out_features; ++o) {
        grad += grad_out[b * out_features + o] * weight[o * in_features + i];
    }
    grad_input[b * in_features + i] = grad;

    if (b == 0) {
        for (int o = 0; o < out_features; ++o) {
            atomicAdd(&grad_bias[o], grad_out[o]);
            for (int j = 0; j < in_features; ++j) {
                atomicAdd(&grad_weight[o * in_features + j], grad_out[o] * input[j]);
            }
        }
    }
}

void linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int B = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    const int threads = 8;
    const int blocks = (B * out_features + threads - 1) / threads;

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, in_features, out_features);
}

void linear_backward(torch::Tensor grad_out, torch::Tensor input, torch::Tensor weight,
                     torch::Tensor grad_input, torch::Tensor grad_weight, torch::Tensor grad_bias) {
    const int B = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    const int threads = 8;
    const int blocks = (B * in_features + threads - 1) / threads;

    linear_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, in_features, out_features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward (CUDA)");
    m.def("linear_backward", &linear_backward, "Linear backward (CUDA)");
}