#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { \
    std::cout << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\n"; exit(1); } }

__global__ void invert_gray(unsigned char *in, unsigned char *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        out[idx] = 255 - in[idx];
    }
}

__global__ void apply_pseudocolor(unsigned char *gray, unsigned char *rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char g = gray[idx];
        if (g < 128) {
            rgb[idx * 3 + 0] = 0;
            rgb[idx * 3 + 1] = g * 2;
            rgb[idx * 3 + 2] = 255 - (g * 2);
        } else {
            rgb[idx * 3 + 0] = (g - 128) * 2;
            rgb[idx * 3 + 1] = 255 - ((g - 128) * 2);
            rgb[idx * 3 + 2] = 0;
        }
    }
}

int main(int argc, char **argv) {
    const char *input_file = (argc > 1) ? argv[1] : "input_xray.png";
    int width, height, channels;
    unsigned char *h_gray = stbi_load(input_file, &width, &height, &channels, 1);
    if (h_gray == NULL) {
        std::cout << "Error: Could not load the image: " << stbi_failure_reason() << "\n";
        return -1;
    }
    std::cout << "Processing X-Ray: " << width << "x" << height << " pixels\n";

    size_t gray_size = (size_t)width * height;
    size_t rgb_size  = gray_size * 3;
    unsigned char *h_inv    = (unsigned char *)malloc(gray_size);
    unsigned char *h_color  = (unsigned char *)malloc(rgb_size);
    unsigned char *h_invcol = (unsigned char *)malloc(rgb_size);

    unsigned char *d_gray, *d_inv, *d_color, *d_invcol;
    CUDA_CHECK(cudaMalloc((void **)&d_gray,   gray_size));
    CUDA_CHECK(cudaMalloc((void **)&d_inv,    gray_size));
    CUDA_CHECK(cudaMalloc((void **)&d_color,  rgb_size));
    CUDA_CHECK(cudaMalloc((void **)&d_invcol, rgb_size));
    CUDA_CHECK(cudaMemcpy(d_gray, h_gray, gray_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    invert_gray<<<gridSize, blockSize>>>(d_gray, d_inv, width, height);
    apply_pseudocolor<<<gridSize, blockSize>>>(d_gray, d_color, width, height);
    apply_pseudocolor<<<gridSize, blockSize>>>(d_inv, d_invcol, width, height);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU kernel time (3 passes): " << ms << " ms\n";

    CUDA_CHECK(cudaMemcpy(h_inv,    d_inv,    gray_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_color,  d_color,  rgb_size,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_invcol, d_invcol, rgb_size,  cudaMemcpyDeviceToHost));

    stbi_write_png("output_inverted.png",             width, height, 1, h_inv,    width);
    stbi_write_png("output_pseudocolor.png",          width, height, 3, h_color,  width * 3);
    stbi_write_png("output_inverted_pseudocolor.png", width, height, 3, h_invcol, width * 3);
    std::cout << "Success! Saved 3 output images.\n";

    cudaFree(d_gray); cudaFree(d_inv); cudaFree(d_color); cudaFree(d_invcol);
    stbi_image_free(h_gray);
    free(h_inv); free(h_color); free(h_invcol);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
