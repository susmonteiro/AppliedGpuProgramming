#include <stdio.h>
#include <sys/time.h>

typedef struct particle {
    float3 position;
    float3 velocity;
} Particle;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void gpu_update_position(Particle* particles, float* rnx, float* rny, float* rnz) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    particles[i].velocity.x = particles[i].velocity.x + rnx[i];
    particles[i].velocity.y = particles[i].velocity.y + rny[i];
    particles[i].velocity.z = particles[i].velocity.z + rnz[i];
    particles[i].position.x = particles[i].position.x + particles[i].velocity.x;
    particles[i].position.y = particles[i].position.y + particles[i].velocity.y;
    particles[i].position.z = particles[i].position.z + particles[i].velocity.z;
}

void cpu_update_position(Particle* particles, float* random_x, float* random_y, float* random_z, int NUM_PARTICLES, int NUM_ITERATIONS) {
    for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < NUM_PARTICLES; i++) {
            particles[i].velocity.x = particles[i].velocity.x + random_x[i];
            particles[i].velocity.y = particles[i].velocity.y + random_y[i];
            particles[i].velocity.z = particles[i].velocity.z + random_z[i];
            particles[i].position.x = particles[i].position.x + particles[i].velocity.x;
            particles[i].position.y = particles[i].position.y + particles[i].velocity.y;
            particles[i].position.z = particles[i].position.z + particles[i].velocity.z;
        }
    }
}

int main(int argc, char **argv) {
    int NUM_PARTICLES = 10000;
    int BLOCK_SIZE = 32;
    int NUM_ITERATIONS = 10000;

    if (argc > 1) NUM_PARTICLES = atoi(argv[1]);
    if (argc > 2) BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3) NUM_ITERATIONS = atoi(argv[3]);

    Particle* particles_cpu = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);
    Particle* particles_gpu = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);

    float *rn_x = (float*)malloc(sizeof(float) * NUM_PARTICLES);
    float *rn_y = (float*)malloc(sizeof(float) * NUM_PARTICLES);
    float *rn_z = (float*)malloc(sizeof(float) * NUM_PARTICLES);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        rn_x[i] = ((float)(rand() % 10000 - 5000))/10000000000;
        rn_y[i] = ((float)(rand() % 10000 - 5000))/10000000000;
        rn_z[i] = ((float)(rand() % 10000 - 5000))/10000000000;
    }

    double iStart = cpuSecond();
    cpu_update_position(particles_cpu, rn_x, rn_y, rn_z, NUM_PARTICLES, NUM_ITERATIONS);
    double iCPUElaps = cpuSecond() - iStart;
    printf("Computing on the CPU... Done!\n\n");
    printf("Time elapsed CPU: %f\n", iCPUElaps);

    Particle *d_particles;
    float *d_rnx, *d_rny, *d_rnz;

    cudaMalloc(&d_particles, NUM_PARTICLES*sizeof(Particle));
    cudaMalloc(&d_rnx, NUM_PARTICLES*sizeof(float));
    cudaMalloc(&d_rny, NUM_PARTICLES*sizeof(float));
    cudaMalloc(&d_rnz, NUM_PARTICLES*sizeof(float));

    cudaMemcpy(d_rnx, rn_x, NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rny, rn_y, NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnz, rn_z, NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles, particles_gpu, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

    iStart = cpuSecond();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        gpu_update_position<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, d_rnx, d_rny, d_rnz);
    }
    cudaDeviceSynchronize();
    double iGPUElaps = cpuSecond() - iStart;
    printf("Computing on the GPU... Done!\n\n");
    printf("Time elapsed GPU: %f\n", iGPUElaps);

    cudaMemcpy(particles_gpu, d_particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);

    // Compare results
    int i;
    for (i = 0; i < NUM_PARTICLES; i++) {
        if (abs(particles_cpu[i].position.x - particles_gpu[i].position.x) > 0.0001
            && abs(particles_cpu[i].position.y - particles_gpu[i].position.y) > 0.0001
            && abs(particles_cpu[i].position.z - particles_gpu[i].position.z) > 0.0001) break;
    }

    if (i == NUM_PARTICLES) printf("Comparing the output for each implementation... Correct!\n");
    else printf("Comparing the output for each implementation... Incorrect :(\n");

    cudaFree(d_particles);
    cudaFree(d_rnx);
    cudaFree(d_rny);
    cudaFree(d_rnz);

    free(particles_cpu);
    free(particles_gpu);
    free(rn_x);
    free(rn_y);
    free(rn_z);

    return 0;
}
