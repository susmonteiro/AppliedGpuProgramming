#include <stdio.h>
#include <sys/time.h>

#define MIN(a, b) (a < b) ? a : b

typedef struct particle {
    float3 position;
    float3 velocity;
} Particle;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void gpu_update_position(Particle* particles, int size) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size) {
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

void createStreams(cudaStream_t *streams, int nstreams) {
    for (int s = 0; s < nstreams; s++)
        cudaStreamCreate(&streams[s]);
}

void destroyStreams(cudaStream_t *streams, int nstreams) {
    for (int s = 0; s < nstreams; s++)
        cudaStreamDestroy(streams[s]);
}

int main(int argc, char **argv) {

    const int BATCH_SIZE = (argc > 1) ? atoi(argv[1]) : 2048;
    const int BLOCK_SIZE = (argc > 2) ? atoi(argv[2]) : 32;
    const int NUM_ITERATIONS = (argc > 3) ? atoi(argv[3]) : 10000;
    const int NUM_PARTICLES =  (argc > 4) ? atoi(argv[4]) : 100000;

    const int NUM_BATCHES = (NUM_PARTICLES / BATCH_SIZE);
    const int LAST_BATCH = NUM_PARTICLES - (BATCH_SIZE * NUM_BATCHES);
    const int NUM_STREAMS = 4;

    printf("Num batches: %d\t\tBatch size: %d\t\tLast batch size: %d\n", NUM_BATCHES, BATCH_SIZE, LAST_BATCH);

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

    // Create Streams
    cudaStream_t streams[NUM_STREAMS];
    createStreams(streams, NUM_STREAMS);

	// Modify the program, such that
	// All particles are copied to the GPU at the beginning of a time step.
	// All the particles are copied back to the host after the kernel completes, before proceeding to the next time step.

    iStart = cpuSecond();
    cudaStream_t stream;
    int offset;
    int batch;
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int stream_n = 0;
        for (batch = 0; batch < NUM_BATCHES; batch++) {
            stream = streams[stream_n];
            offset = batch * BATCH_SIZE;

            cudaMemcpyAsync(&d_particles[offset], &particles_gpu[offset], BATCH_SIZE*sizeof(Particle), cudaMemcpyHostToDevice, stream);
            gpu_update_position<<<(BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(&d_particles[offset], BATCH_SIZE);    
            cudaMemcpyAsync(&particles_gpu[offset], &d_particles[offset], BATCH_SIZE*sizeof(Particle), cudaMemcpyDeviceToHost, stream);
            
            stream_n = (stream_n + 1) % NUM_STREAMS;
        }
        offset = batch * BATCH_SIZE;
        stream = streams[stream_n];

        cudaMemcpyAsync(&d_particles[offset], &particles_gpu[offset], LAST_BATCH*sizeof(Particle), cudaMemcpyHostToDevice, stream);
        gpu_update_position<<<(BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(&d_particles[offset], LAST_BATCH);  
        cudaMemcpyAsync(&particles_gpu[offset], &d_particles[offset], LAST_BATCH*sizeof(Particle), cudaMemcpyDeviceToHost, stream);
        
        cudaDeviceSynchronize();
    }

    double iGPUElaps = cpuSecond() - iStart;
    printf("Computing on the GPU... Done!\n\n");
	printf("Time elapsed GPU including copying time: %f\n", iGPUElaps);

    // Destroy Streams
    destroyStreams(streams, NUM_STREAMS);


    // Compare results
    int i;
    for (i = 0; i < NUM_PARTICLES; i++) {
        if (abs(particles_cpu[i].position.x - particles_gpu[i].position.x) > 0.0001
            && abs(particles_cpu[i].position.y - particles_gpu[i].position.y) > 0.0001
            && abs(particles_cpu[i].position.z - particles_gpu[i].position.z) > 0.0001) {
                printf("=== FAILED on particle %d ===\nxc: %f\t\txg: %f\nyc: %f\t\tyg: %f\nzc: %f\t\tzg: %f\n", i,
                    particles_cpu[i].position.x, particles_gpu[i].position.x,
                    particles_cpu[i].position.y, particles_gpu[i].position.y,
                    particles_cpu[i].position.z, particles_gpu[i].position.z
                );  
                break;
            }
    }

    if (i == NUM_PARTICLES) printf("Comparing the output for each implementation... Correct!\n");
    else printf("Comparing the output for each implementation... Incorrect :(\n");

    cudaFree(d_particles);
	cudaFreeHost(particles_gpu);  // because it was allocated with cudaMallocHost

    free(particles_cpu);

    return 0;
}

