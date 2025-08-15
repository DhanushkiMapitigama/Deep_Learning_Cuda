#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> 

#include <cublas_v2.h>
#include <cuda_runtime.h>


//________________________CUDA__________________________________

// ------------------ Forward with 1D kernel ------------------

__global__ void linear_forward_kernel(
    const float* __restrict__ input,   // Dimensions: [batch_size, in_features]
    const float* __restrict__ weights, // Dimensions: [out_features, in_features]
    const float* __restrict__ bias,    // Dimensions: [out_features]
    float* __restrict__ output,        // Dimensions: [batch_size, out_features]
    int batch_size, int in_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= batch_size * out_features) return;

    int b = idx / out_features;   // batch index
    int o = idx % out_features;   // output feature index

    float sum = bias[o];
    for (int i = 0; i < in_features; ++i) {
        sum += input[b * in_features + i] * weights[o * in_features + i];
    }
    output[b * out_features + o] = sum;
}

void linear_forward(torch::Tensor input, torch::Tensor weight,
                    torch::Tensor bias, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
}

// ------------------ Forward with 2D kernel ------------------
__global__ void linear_forward_2d_kernel(
    const float* __restrict__ input,    // [batch_size, in_features]
    const float* __restrict__ weights,  // [out_features, in_features]
    const float* __restrict__ bias,     // [out_features]
    float* __restrict__ output,         // [batch_size, out_features]
    int batch_size, int in_features, int out_features
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x; // output feature index
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch index

    if (b >= batch_size || o >= out_features) return;

    float sum = bias[o];
    for (int i = 0; i < in_features; ++i) {
        sum += input[b * in_features + i] * weights[o * in_features + i];
    }
    output[b * out_features + o] = sum;
}

void linear_forward_2d(torch::Tensor input, torch::Tensor weight,
                       torch::Tensor bias, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((out_features + threads.x - 1) / threads.x,
                (batch_size + threads.y - 1) / threads.y);

    linear_forward_2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features
    );
}


// ------------------ Backward with 1D kernel ------------------

// Computation: grad_input[b, i] = sum_o grad_out[b, o] * weight[o, i]
__global__ void grad_input_kernel(
    const float* __restrict__ grad_out, // Dimensions: [batch_size, out_features]
    const float* __restrict__ weight,   // Dimensions: [out_features, in_features]
    float* __restrict__ grad_input,     // Dimensions: [batch_size, in_features]
    int batch_size, int in_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_features) return;

    int b = idx / in_features;
    int i = idx % in_features;

    float sum = 0.0f;
    for (int o = 0; o < out_features; ++o) {
        sum += grad_out[b * out_features + o] * weight[o * in_features + i];
    }
    grad_input[b * in_features + i] = sum;
}

// Computation: grad_weight[o, i] = sum_b grad_out[b, o] * input[b, i]
__global__ void grad_weight_kernel(
    const float* __restrict__ grad_out, // Dimensions: [batch_size, out_features]
    const float* __restrict__ input,    // Dimensions: [batch_size, in_features]
    float* __restrict__ grad_weight,    // Dimensions: [out_features, in_features]
    int batch_size, int in_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_features * in_features) return;

    int o = idx / in_features;
    int i = idx % in_features;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += grad_out[b * out_features + o] * input[b * in_features + i];
    }
    grad_weight[o * in_features + i] = sum;
}

// Computation: grad_bias[o] = sum_b grad_out[b, o]
__global__ void grad_bias_kernel(
    const float* __restrict__ grad_out, // Dimensions: [batch_size, out_features]
    float* __restrict__ grad_bias,      // Dimensions: [out_features]
    int batch_size, int out_features
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += grad_out[b * out_features + o];
    }
    grad_bias[o] = sum;
}

void linear_backward(torch::Tensor grad_out, 
                    torch::Tensor input, 
                    torch::Tensor weight,
                    torch::Tensor grad_input, 
                    torch::Tensor grad_weight, 
                    torch::Tensor grad_bias) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    int threads = 512;

    // compute grad_input
    {
        int blocks = (batch_size * in_features + threads - 1) / threads;
        grad_input_kernel<<<blocks, threads>>>(
            grad_out.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size, in_features, out_features
        );
    }

    // compute grad_weight
    {
        int blocks = (out_features * in_features + threads - 1) / threads;
        grad_weight_kernel<<<blocks, threads>>>(
            grad_out.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            batch_size, in_features, out_features
        );
    }

    // compute grad_bias
    {
        int blocks = (out_features + threads - 1) / threads;
        grad_bias_kernel<<<blocks, threads>>>(
            grad_out.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            batch_size, out_features
        );
    }
}

//________________________CuBLAS__________________________________

cublasHandle_t get_cublas_handle() { 
    static cublasHandle_t handle = nullptr; 
    if (!handle) { 
        cublasCreate(&handle); 
    } 
    return handle;
}

// ------------------ Forward ------------------

__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;
    int o = idx % out_features;
    output[idx] += bias[o];
}

void linear_forward_cublas(
    torch::Tensor input,  // [batch_size, in_features]
    torch::Tensor weight, // [out_features, in_features]
    torch::Tensor bias,   // [out_features]
    torch::Tensor output  // [batch_size, out_features]
) {
    
    auto handle = get_cublas_handle();
    
    const int batch_size   = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = weight.size(0);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T, 
        CUBLAS_OP_N, 
        out_features,        
        batch_size,          
        in_features,         
        &alpha,
        weight.data_ptr<float>(), in_features, 
        input.data_ptr<float>(),  in_features, 
        &beta,
        output.data_ptr<float>(), out_features 
    );
    
    int total = batch_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );
}

// ------------------ Backward ------------------

__global__ void grad_bias_kernel_cublas(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    int batch_size, int out_features
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += grad_out[b * out_features + o];
    }
    grad_bias[o] = sum;
}

void linear_backward_cublas(torch::Tensor grad_out,
                            torch::Tensor input,
                            torch::Tensor weight,
                            torch::Tensor grad_input,
                            torch::Tensor grad_weight,
                            torch::Tensor grad_bias) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    // -------------------------
    // grad_input = grad_out × weight 
    // Inputs to CuBLAS function interchnaged to get grad_input in row major
    // -------------------------
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                in_features, batch_size, out_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                grad_out.data_ptr<float>(), out_features,
                &beta,
                grad_input.data_ptr<float>(), in_features);

    // -------------------------
    // grad_weight = grad_out^T × input
    // Inputs to CuBLAS function interchnaged to get grad_weight in row major
    // -------------------------
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                in_features, out_features, batch_size,
                &alpha,
                input.data_ptr<float>(), in_features,
                grad_out.data_ptr<float>(), out_features,
                &beta,
                grad_weight.data_ptr<float>(), in_features);

   // compute grad_bias
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    grad_bias_kernel_cublas<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        batch_size, out_features
    );

    cublasDestroy(handle);
}


// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward (CUDA)");
    m.def("linear_backward", &linear_backward, "Linear backward (CUDA)");
    m.def("linear_forward_cublas", &linear_forward_cublas, "Linear forward with cuBLAS");
    m.def("linear_backward_cublas", &linear_backward_cublas, "Linear backward with cuBLAS");
}
