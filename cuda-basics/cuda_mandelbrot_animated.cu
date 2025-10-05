/****************************************************************************************
 * CUDA ASCII MANDELBROT (FIXED VISUAL VERSION)
 * Works well in most terminals (even SSH / EC2)
 *
 * Compile:  nvcc cuda_mandelbrot_fixed.cu -o mandelbrot_fixed
 * Run:      ./mandelbrot_fixed
 ****************************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>  // for usleep()

__global__ void mandelbrotKernel(char* output, int width, int height,
                                 double xMin, double xMax, double yMin, double yMax, int maxIter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    double x0 = xMin + col * (xMax - xMin) / width;
    double y0 = yMin + row * (yMax - yMin) / height;

    double x = 0.0, y = 0.0;
    int iter = 0;
    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xTemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xTemp;
        iter++;
    }

    const char charset[] = " .:-=+*#%@";
    int nChars = sizeof(charset) - 1;
    output[row * width + col] = charset[(iter * nChars) / maxIter];
}

int main() {
    const int width = 120;   // works well for most terminals
    const int height = 40;
    const int maxIter = 1000;

    char* d_output;
    cudaMalloc(&d_output, width * height);
    char* h_output = new char[width * height];

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    // Better center for nice visuals
    double xCenter = -0.7435;
    double yCenter = 0.1314;
    double scale = 0.01;

    std::cout << "\033[2J"; // clear screen
    for (int frame = 0; frame < 120; ++frame) {
        double zoom = scale * pow(0.97, frame);
        double aspect = (double)height / width * 2.0; // fix aspect ratio
        double xMin = xCenter - zoom;
        double xMax = xCenter + zoom;
        double yMin = yCenter - zoom * aspect;
        double yMax = yCenter + zoom * aspect;

        mandelbrotKernel<<<blocks, threads>>>(d_output, width, height, xMin, xMax, yMin, yMax, maxIter);
        cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        std::cout << "\033[H";
        std::cout << "🎨 CUDA Mandelbrot Zoom\n";
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col)
                std::cout << h_output[row * width + col];
            std::cout << "\n";
        }
        std::cout << "Frame: " << frame << " | Zoom: " << zoom << "\n";
        usleep(120000); // ~8 FPS
    }

    delete[] h_output;
    cudaFree(d_output);
    std::cout << "\n✅ Animation finished!\n";
}
