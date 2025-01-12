#include <cuda/std/span>
#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
__global__ void gather_kernel(
  cuda::std::span<const T> data,
  cuda::std::span<const int> indices,
  cuda::std::span<T> output,
  int axis,
  int data_shape_axis,
  int output_size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < output_size) {
    int data_idx = 0;
    if (axis == 0) {
      int row = indices[idx / data_shape_axis];
      int col = idx % data_shape_axis;
      data_idx = row * data_shape_axis + col;
    } else if (axis == 1) {
      int row = idx / indices.size();        
      int col = indices[idx % indices.size()]; 
      data_idx = row * data_shape_axis + col; 
    }
    output[idx] = data[data_idx];
  }
}

extern "C" {
  void gather_cuda_f16(
    void const *data, void const *indices, void *output,
    int axis, int data_shape_axis, int output_size, int indices_size) {
    int blockSize = 256;
    int gridSize = (output_size + blockSize - 1) / blockSize;
    gather_kernel<<<gridSize, blockSize>>>(
      cuda::std::span<const __half>((const __half *)data, data_shape_axis * 3),
      cuda::std::span<const int>((const int *)indices, indices_size),
      cuda::std::span<__half>((__half *)output, output_size),
      axis,
      data_shape_axis,
      output_size
    );
  }


  void gather_cuda_f32(
    void const *data, void const *indices, void *output,
    int axis, int data_shape_axis, int output_size, int indices_size) {
    int blockSize = 256;
    int gridSize = (output_size + blockSize - 1) / blockSize;
    gather_kernel<<<gridSize, blockSize>>>(
      cuda::std::span<const float>((const float *)data, data_shape_axis * 3),
      cuda::std::span<const int>((const int *)indices, indices_size),
      cuda::std::span<float>((float *)output, output_size),
      axis,
      data_shape_axis,
      output_size
    );
  }
}

