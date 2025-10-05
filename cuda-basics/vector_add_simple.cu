/****************************************************************************************
 * CUDA 1D VECTOR ADDITION
 *
 * Each thread adds one element of two input vectors A and B,
 * and stores the result in vector C.
 *
 * Demonstrates CPU → GPU memory copy, kernel launch, and GPU → CPU copy.
 *
 * Compile:  nvcc vector_add_logged.cu -o vector_add_logged
 * Run:      ./vector_add_logged
 ****************************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_SIZE 256

// ---------------- GPU Kernel ----------------
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global thread index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    cout << "================ CUDA VECTOR ADDITION ================\n";

    // 1. Set vector size (1 million elements)
    int N = 1 << 20;  // 1,048,576 elements
    size_t size = N * sizeof(float);
    cout << "[1] Vector size: " << N << " elements (" << size / (1024.0 * 1024.0) << " MB)\n";

    // 2. Allocate host (CPU) memory
    cout << "[2] Allocating host memory...\n";
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize vectors A and B
    init_vector(h_A, N);
    init_vector(h_B, N);
    cout << "    ✅ Host memory ready.\n";

    // 3. Allocate device (GPU) memory
    cout << "[3] Allocating GPU memory...\n";
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cout << "    ✅ GPU memory allocated.\n";

    // 4. Copy input data from host → device
    cout << "[4] Copying data from CPU → GPU...\n";
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cout << "    ✅ Data copied to GPU.\n";

    // 5. Launch kernel on GPU
    cout << "[5] Launching kernel...\n";
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "    → Grid size: " << blocksPerGrid << " blocks\n";
    cout << "    → Block size: " << BLOCK_SIZE << " threads\n";

    vectorAdd<<<blocksPerGrid, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cout << "    ✅ Kernel execution complete.\n";

    // 6. Copy result from device → host
    cout << "[6] Copying result from GPU → CPU...\n";
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cout << "    ✅ Results copied back.\n";

    // 7. Display first 10 results (to verify correctness)
    cout << "[7] Sample Results:\n";
    for (int i = 0; i < 10; i++)
        cout << "    " << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";

    // 8. Free GPU memory
    cout << "[8] Freeing GPU memory...\n";
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 9. Free CPU memory
    cout << "[9] Freeing host memory...\n";
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cout << "\n✅ All done successfully!\n";
    cout << "======================================================\n";
    return 0;
}
