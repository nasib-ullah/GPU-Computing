#include<stdio.h>
#include<sys/time.h>
#include<cuda_runtime.h>

void checkResult(float *hostRef, float *gpuRef, const int N) {
    //printf("Checking results");
    double epsilon = 1.0E-8; bool match = 1; 
    for (int i=0; i<N; i++) { 
       if (abs(hostRef[i] - gpuRef[i]) > epsilon) 
           {    match = 0;
                printf("Arrays do not match!\n"); 
                printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
                break; 
               } } 
               
       if (match) printf("Arrays match.\n\n"); 
   }


double cpuSecond() {
    struct timeval tp; 
    gettimeofday(&tp,NULL); 
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6); 
   }

void initialData(float *ip,int size) { 
    // generate different seed for random number 
    time_t t; 
    srand((unsigned) time(&t)); 
    for (int i=0; i<size; i++) { 
        ip[i] = (float)( rand() & 0xFF )/10.0f; 
    }
}

void sumMatrixOnHost(float *A, float *B, float *C,const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for(int iy=0;iy<ny;iy++){
        for(int ix=0;ix<nx;ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia +=nx; 
        ib +=nx; 
        ic +=nx;
    }

}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;
    if (ix<nx && iy<ny){
        MatC[iy*nx+ix] = MatA[iy*nx+ix] + MatB[iy*nx+ix];
    }
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, const int nx, const int ny){

    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
    if (ix<nx){
        for (int iy=0;iy<ny;iy++)
        {
            MatC[iy*nx+ix] = MatA[iy*nx+ix] + MatB[iy*nx+ix];
        }
    }
}

int main(int argc, char **argv){
    printf("%s Starting.... \n",argv[0]);

    //setup devices
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d:%s \n",dev,deviceProp.name);
    cudaSetDevice(dev);

    //setup data size of matrix
    int nx = 1<<14;
    int ny = 1<<14;

    int nxy = nx*ny;
    size_t nBytes = nxy*sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx, ny);

    //malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    hostRef = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);

    //Initialize data at the host side
    double iStart, iElaps;
    iStart = cpuSecond();
    initialData(h_A,nxy);
    initialData(h_B,nxy);
    iElaps = cpuSecond() - iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    //add matrix at host side for result check
    iStart = cpuSecond();
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnCPU Time elapsed %f sec\n", iElaps);

    //malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((float**)&d_MatA ,nBytes);
    cudaMalloc((float**)&d_MatB ,nBytes);
    cudaMalloc((float**)&d_MatC ,nBytes);

    //transfer data from host to device
    cudaMemcpy( d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    //invoke kernel at host side
    int dimx = 32;
    int dimy = 32; //[32,1]
    dim3 block (dimx,dimy);
    dim3 grid ((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);    //((nx+block.x-1)/block.x,(ny+block.y-1)/block.y)

    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
    //sumMatrixOnGPU1D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    //copy kernel results back to host memory
    cudaMemcpy( gpuRef, d_MatC,nBytes , cudaMemcpyDeviceToHost);

    // check device results 
    checkResult(hostRef, gpuRef, nxy);

    //free device memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    //free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device 
    cudaDeviceReset();

    return 0;
}