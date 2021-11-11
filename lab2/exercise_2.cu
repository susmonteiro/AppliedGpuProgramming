#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#define N 214748364

__global__ void kernel(float* a, float* x, float* y, float* out){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    out[idx] = *a * x[idx] + y[idx];
}

void cpu_saxpy(float* out, float a, float* x, float* y) {
    for (int i = 0; i < N; i++) {
        out[i] = a* x[i] + y[i];
    }
    
   printf("Computing SAXPY on the CPU... Done!\n\n");
}

int main() {
    // Save variables on the CPU
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float a = 2.0;

    for(int i = 0; i < N; i++) x[i] = (float)i;
    for(int i = 0; i < N; i++) y[i] = (float)i;

    // Copy data to the gpu
    float *d_x, *d_y, *d_a, *d_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_a, sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));

    float* out_gpu = (float*)malloc(N * sizeof(float));
    float* out_cpu = (float*)malloc(N * sizeof(float));

    cudaMemcpy(d_a, &a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    
    kernel<<<(N + 255) / 256, 256>>>(d_a, d_x, d_y, d_out);
    cudaDeviceSynchronize();
    printf("Computing SAXPY on the GPU... Done!\n\n");

    cudaMemcpy(out_gpu, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < N; i++) printf("out[%d]: %f\n", i, out_gpu[i]);

    cpu_saxpy(out_cpu, a, x, y);
    
    // Compare results
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (abs(out_cpu[i] - out_gpu[i]) < 0.0001) ++cnt;
        // printf("%d: %f %f\n", i, out_cpu[i], out_gpu[i]);
    }

    if (cnt == N) printf("Comparing the output for each implementation... Correct!"); 
    else printf("Comparing the output for each implementation... Incorrect :(");

    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    free(x);
    free(y);
    free(out_gpu);
    free(out_cpu);

    return 0;
}