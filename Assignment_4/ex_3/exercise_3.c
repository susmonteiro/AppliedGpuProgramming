// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>

#define N 10000

// This is a macro for checking the error variable.
// #define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
// const char* clGetErrorString(int);


// const char *mykernel = "\
//   __kernel\
//   void saxpy(\
//     float a,\
//     __global float* x,\
//     __global float* y\
//   ) {\
//     int index = get_global_id(0);\
//     y[index] += a*x[index];\
//   }";

double cpuSecond() {
 struct timeval tp;
 gettimeofday(&tp,NULL);
 return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void cpu_saxpy(float a, float* x, float* y, float Y[N]) {
  int cnt = 0;
  for (int i = 0; i < N; i++) {
    y[i] += a*x[i];
    if (abs(y[i] - Y[i]) < 0.0001) ++cnt;
  }


  if (cnt == N) printf("Comparing the output for each implementation... Correct!");
  else printf("Comparing the output for each implementation... Incorrect :(");
}

int main(int argc, char *argv) {
  // initialise on GPU
  int array_size = N * sizeof(float);
  float X[N], Y[N], A = 2.0;

  // initialize on CPU
  float* x = (float*)malloc(N * sizeof(float));
  float* y = (float*)malloc(N * sizeof(float));
  float a = 2.0;

  for(int i = 0; i < N; i++) {
    x[i] = y[i] = X[i] = Y[i] = (float)i;
  }

  // compute on GPU
  double gpuStart = cpuSecond();
  #pragma acc parallel loop 
  for (int i = 0; i < N; i++) {
	Y[i] += a*X[i];
  }
  double gpuElaps = cpuSecond() - gpuStart;

  printf("Computing SAXPY on the GPU... Done!\n\n");
  printf("GPU time: %d \n\n", gpuElaps);

  // check the result with CPU
  double cpuStart = cpuSecond();
  cpu_saxpy(a, x, y, Y); // y contains the result
  double cpuElaps = cpuSecond() - cpuStart;

  printf("Computing SAXPY in the CPU... Done!\n\n");
  printf("CPU time: %d \n\n", cpuElaps);

  free(x);
  free(y);

  return 0;
}
