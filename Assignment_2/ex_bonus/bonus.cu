#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>

#define NUM_ITER 20
#define count 1000000


__global__ void kernel(int* c, curandState *states){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	int seed = id; // different seed per thread
    curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND

    int inside = 0;
    for(int i = 0; i < count; i++) {
        float x = curand_uniform(&states[id]);
        float y = curand_uniform(&states[id]);
        if(sqrt(x*x + y*y) < 1) ++inside;
    }
    
    c[id] = inside;
    // printf("[%d] %d\n", id, c[id]);
    return;
}

int main() {
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, size_t(NUM_ITER) * sizeof(curandState));

    int* d_c = NULL;
    cudaMalloc((void**)&d_c, size_t(NUM_ITER) * sizeof(int));
    
    // Time
    clock_t t;
    t = clock();

    kernel<<<(size_t(NUM_ITER) + 127) / 128, 128>>>(d_c, dev_random);
    cudaDeviceSynchronize();


    int* out_gpu = (int*) malloc(size_t(NUM_ITER) * sizeof(int));
    memset(out_gpu, 0, size_t(NUM_ITER) * sizeof(int));
    cudaMemcpy(out_gpu, d_c, size_t(NUM_ITER) * sizeof(int), cudaMemcpyDeviceToHost);

    
    // Time
    t = clock() - t;
    printf("Time: %f\n", ((float)t)/CLOCKS_PER_SEC);

    int sum = 0;
    for(int i = 0; i < size_t(NUM_ITER); i++) {
        // printf("After [%d] %d\n", i, out_gpu[i]);
        sum += out_gpu[i];
    }

    float pi = 4 * ((float) sum / (count * size_t(NUM_ITER)));
    printf("PI=%f\n", pi);

    cudaFree(d_c);
    cudaFree(dev_random);
    free(out_gpu);

    return 0;
}
