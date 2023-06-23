#include<iostream>
#include <chrono> // Method-I for timer
#include<cmath>


using namespace std;
using namespace std::chrono;

void vecadd_cpu(float* x, float* y, float* z, int N){
    for(unsigned int i=0;i<N;++i){
        z[i] = x[i] + y[i]; 
    }
} 

__global__ void vecadd_kernel(float* x, float* y, float* z, int N){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
    z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float* x, float* y, float* z,int N){

    //allocate GPU memory
    auto start1 = high_resolution_clock::now();

    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d,N*sizeof(float));
    cudaMalloc((void**)&y_d,N*sizeof(float));
    cudaMalloc((void**)&z_d,N*sizeof(float));
    cudaDeviceSynchronize();
    auto stop1 = high_resolution_clock::now(); //Method-I for timer
    auto duration1 = duration_cast<milliseconds>(stop1 - start1); //Method-I for timer
    cout << "GPU Memory Allocation Time: "<< duration1.count() << " milliseconds" << endl; //Method-I for timer

    //copy to the GPU
    auto start2 = high_resolution_clock::now();

    cudaMemcpy(x_d,x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,y,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto stop2 = high_resolution_clock::now(); //Method-I for timer
    auto duration2 = duration_cast<milliseconds>(stop2 - start2); //Method-I for timer
    cout << "Copy to GPU Time: "<< duration2.count() << " milliseconds" << endl; //Method-I for timer

    //Call a GPU Kernel function
    auto start3 = high_resolution_clock::now();

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1)/numThreadsPerBlock;
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d,y_d,z_d,N);
    cudaDeviceSynchronize();
    auto stop3 = high_resolution_clock::now(); //Method-I for timer
    auto duration3 = duration_cast<milliseconds>(stop3 - start3); //Method-I for timer
    cout << "Kernel Time: "<< duration3.count() << " milliseconds" << endl; //Method-I for timer

    double sum=0;
    for(unsigned int i=0;i<N;++i){
        sum += z[i];
    }
    cout<<"GPU Sum:"<<sum<<endl;

    //Copy from the GPU
    auto start4 = high_resolution_clock::now();
    cudaMemcpy(z,z_d,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto stop4 = high_resolution_clock::now(); //Method-I for timer
    auto duration4 = duration_cast<milliseconds>(stop4 - start4); //Method-I for timer
    cout << "Copy from GPU Time: "<< duration4.count() << " milliseconds" << endl; //Method-I for timer

    //Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

}

int main(int argc,char **argv){
    cudaDeviceSynchronize();
    
    
    //Allocating memory and initialize data
    unsigned int N=10000000;
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for(unsigned int i=0;i<N;++i){
        x[i] = rand();
        y[i] = rand();
    }

    //sleep(10);

    //vector addition on CPU
    auto start1 = high_resolution_clock::now(); //Method-I  for timer
    vecadd_cpu(x,y,z,N);
    auto stop1 = high_resolution_clock::now(); //Method-I for timer
    auto duration1 = duration_cast<milliseconds>(stop1 - start1); //Method-I for timer
    cout << "CPU Vector addition Time: "<< duration1.count() << " milliseconds" << endl; //Method-I for timer

    double sum=0;
    for(unsigned int i=0;i<N;++i){
        sum += z[i];
    }
    cout<<"CPU SUM:"<<sum<<endl;

    //vector addition on GPU
    auto start = high_resolution_clock::now();
    vecadd_gpu(x,y,z,N);
    auto stop = high_resolution_clock::now(); //Method-I for timer
    auto duration = duration_cast<milliseconds>(stop - start); //Method-I for timer
    cout << "Total GPU Vector addition Time: "<< duration.count() << " milliseconds" << endl; //Method-I for timer

    sum=0;
    for(unsigned int i=0;i<N;++i){
        sum += z[i];
    }
    cout<<"GPU Sum:"<<sum<<endl;

    free(x);
    free(y);
    free(z);

    return 0;
}