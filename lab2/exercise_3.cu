#include <stdio.h>
#include <sys/time.h>
#define NUM_PARTICLES 10
#define NUM_ITERATIONS 1
#define BLOCK_SIZE 32

typedef struct particle {
    float3 position;
    float3 velocity;
} Particle;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void cpu_update_position() {
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        p[i+1].position.x = p[i].position.x + rand() % 70;
    }
}

int main() {
    struct Particle* old_part = malloc(NUM_PARTICLES*sizeof(struct Particle))

    double iStart = cpuSecond();
    // kernel_name<<>>(argument list);
    cpu_update_position();
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    return 0;
}