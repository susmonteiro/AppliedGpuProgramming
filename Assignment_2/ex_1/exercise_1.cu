#include <stdio.h>

__global__ void kernel()
{
  printf("Hello world! My threadId is %d\n", threadIdx.x);
  return;
}

int main()
{
  
  kernel<<<1, 256>>>();

  cudaDeviceSynchronize();
  
  return 0;
}