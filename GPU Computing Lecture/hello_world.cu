#include<iostream>

using namespace std;

__global__ void cuda_hello(){
    float a = 0;
    
}

int main() {
    cuda_hello<<<1,1>>>(); 
    cout<<"Hello World from GPU!"<<"\n";
    return 0;
}
