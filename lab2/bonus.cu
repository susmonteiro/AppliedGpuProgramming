#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>

#define NUM_ITER 2000000
#define count 100


__global__ void kernel(float* c, curandState *states){
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
    
    return;
}

int main() {
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NUM_ITER*sizeof(curandState));

    float* d_c;
    cudaMalloc(&d_c, NUM_ITER * sizeof(float));
    // cudaMemset(d_c, 0, NUM_ITER * sizeof(int));
    
    clock_t t;
    t = clock();

    kernel<<<(NUM_ITER + 255) / 256, 256>>>(d_c, dev_random);

    int* out_gpu = (int*)malloc(NUM_ITER * sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(out_gpu, d_c, NUM_ITER * sizeof(int), cudaMemcpyDeviceToHost);
    t = clock() - t;

    printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);

    unsigned long long sum = 0;
    for(int i = 0; i < NUM_ITER; i++) {
        sum += out_gpu[i];
    }

    double pi = 4 * ((double)sum / (count * NUM_ITER));
    printf("%f\n", pi);

    return 0;
}