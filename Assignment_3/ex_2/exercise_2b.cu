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

__global__ void gpu_update_position(Particle* particles, float* rnx, float* rny, float* rnz, int nparticles) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i < nparticles) {
        particles[i].velocity.x = particles[i].velocity.x + rnx[i];
        particles[i].velocity.y = particles[i].velocity.y + rny[i];
        particles[i].velocity.z = particles[i].velocity.z + rnz[i];
        particles[i].position.x = particles[i].position.x + particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + particles[i].velocity.z;
    }
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

	// using managed memory allocator
	Particle* particles_gpu;
	cudaMallocManaged(&particles_gpu, sizeof(Particle) * NUM_PARTICLES);

	float *rn_x, *rn_y, *rn_z;
    cudaMallocManaged(&rn_x, sizeof(float) * NUM_PARTICLES);
	cudaMallocManaged(&rn_y, sizeof(float) * NUM_PARTICLES);
	cudaMallocManaged(&rn_z, sizeof(float) * NUM_PARTICLES);


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

	// Modify the program, such that
	// All particles are copied to the GPU at the beginning of a time step.
	// All the particles are copied back to the host after the kernel completes, before proceeding to the next time step.

    iStart = cpuSecond();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
		gpu_update_position<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(particles_gpu, rn_x, rn_y, rn_z, NUM_PARTICLES);
    }
    cudaDeviceSynchronize();

    double iGPUElaps = cpuSecond() - iStart;
    printf("Computing on the GPU... Done!\n\n");
	printf("Time elapsed GPU including copying time: %f\n", iGPUElaps);



    // Compare results
    int i;
    for (i = 0; i < NUM_PARTICLES; i++) {
        if (abs(particles_cpu[i].position.x - particles_gpu[i].position.x) > 0.0001
            && abs(particles_cpu[i].position.y - particles_gpu[i].position.y) > 0.0001
            && abs(particles_cpu[i].position.z - particles_gpu[i].position.z) > 0.0001) break;
    }

    if (i == NUM_PARTICLES) printf("Comparing the output for each implementation... Correct!\n");
    else printf("Comparing the output for each implementation... Incorrect :(\n");

	cudaFree(particles_gpu);
    cudaFree(rn_x);
    cudaFree(rn_y);
    cudaFree(rn_z);

    free(particles_cpu);

    return 0;
}
