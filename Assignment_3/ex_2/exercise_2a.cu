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

__global__ void gpu_update_position(Particle* particles, int nparticles) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < nparticles) {
        particles[i].velocity.x = particles[i].position.x / 10;
        particles[i].velocity.y = particles[i].position.y / 10;
        particles[i].velocity.z = particles[i].position.z / 10;
        particles[i].position.x += particles[i].velocity.x;
        particles[i].position.y += particles[i].velocity.y;
        particles[i].position.z += particles[i].velocity.z;
    }
}

void cpu_update_position(Particle* particles, int NUM_PARTICLES, int NUM_ITERATIONS) {
    for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < NUM_PARTICLES; i++) {
            particles[i].velocity.x = particles[i].position.x / 10;
            particles[i].velocity.y = particles[i].position.y / 10;
            particles[i].velocity.z = particles[i].position.z / 10;
            particles[i].position.x += particles[i].velocity.x;
            particles[i].position.y += particles[i].velocity.y;
            particles[i].position.z += particles[i].velocity.z;
        }
    }
}

void generatePositions(Particle* particles_cpu, Particle* particles_gpu, int nparticles) {
    for (int i = 0; i < nparticles; i++) {
        particles_cpu[i].position.x = (float)rand()/RAND_MAX;
        particles_cpu[i].position.y = (float)rand()/RAND_MAX;
        particles_cpu[i].position.z = (float)rand()/RAND_MAX;
        particles_gpu[i].position.x = particles_cpu[i].position.x;
        particles_gpu[i].position.y = particles_cpu[i].position.y;
        particles_gpu[i].position.z = particles_cpu[i].position.z;
        particles_cpu[i].velocity = {};
        particles_gpu[i].velocity = {};
    }
}

int main(int argc, char **argv) {
    int NUM_PARTICLES = 10000;
    int BLOCK_SIZE = 32;
    int NUM_ITERATIONS = 1;

    if (argc > 1) NUM_PARTICLES = atoi(argv[1]);
    if (argc > 2) BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3) NUM_ITERATIONS = atoi(argv[3]);

    Particle* particles_cpu = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);

	// Allocating particles_gpu in pinned memory
	Particle* particles_gpu;
	cudaMallocHost(&particles_gpu, sizeof(Particle) * NUM_PARTICLES);

    generatePositions(particles_cpu, particles_gpu, NUM_PARTICLES);

    double iStart = cpuSecond();
    cpu_update_position(particles_cpu, NUM_PARTICLES, NUM_ITERATIONS);
    double iCPUElaps = cpuSecond() - iStart;
    printf("Computing on the CPU... Done!\n\n");
    printf("Time elapsed CPU: %f\n", iCPUElaps);

    Particle *d_particles;

    cudaMalloc(&d_particles, NUM_PARTICLES*sizeof(Particle));

	// Modify the program, such that
	// All particles are copied to the GPU at the beginning of a time step.
	// All the particles are copied back to the host after the kernel completes, before proceeding to the next time step.

    iStart = cpuSecond();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
	    cudaMemcpy(d_particles, particles_gpu, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
        gpu_update_position<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_particles, NUM_PARTICLES);
		cudaMemcpy(particles_gpu, d_particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
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

    cudaFree(d_particles);
	cudaFreeHost(particles_gpu);  // because it was allocated with cudaMallocHost

    free(particles_cpu);

    return 0;
}
