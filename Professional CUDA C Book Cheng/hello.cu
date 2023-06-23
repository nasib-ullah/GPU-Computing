#include<stdio.h>


__global__ void hellofromGPU(void)
{
   int tid = threadIdx.x;
   //printf("%d",tid);
   if(tid==5){
   printf("Hello world from GPU %d\n",tid);
   }
}

int main(void)
{
  printf("Hello world from CPU \n ");
  hellofromGPU<<<1,10>>>();
  cudaDeviceReset();
  return 0;
}
