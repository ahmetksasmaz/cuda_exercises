#include <iostream>
#include "cuda_runtime.h"

#define ARRAY_SIZE 64

using namespace std;

__global__ void array_sum_pre(int* arr){

    __shared__ int s_arr[ARRAY_SIZE];

    s_arr[threadIdx.x] = arr[threadIdx.x];

    __syncthreads();

    int sum = 0;

    for(int i = 0; i <= threadIdx.x; i++){
        sum += s_arr[i];
    }

    arr[threadIdx.x] = sum;

}

int main(){

    // Initialize Variables
    int* h_arr = 0;
    int* d_arr = 0;
    int size = ARRAY_SIZE * sizeof(int);
    float estimated_time;
    dim3 grid, block;
    grid.x = 1;
    block.x = ARRAY_SIZE;
    cudaEvent_t start, stop;

    // Allocate Memory
    h_arr = (int*)malloc(size);
    cudaMalloc((void**)&d_arr, size);

    // Fill Array
    for(int i = 0; i < ARRAY_SIZE; i++){
        h_arr[i] = i;
    }

    // Copy Memory
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Launch Kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    array_sum_pre<<<grid, block>>>(d_arr);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&estimated_time, start, stop);

    // Copy Memory
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Print Array and Statistics
    for(int i = 0; i < ARRAY_SIZE; i++){
        cout << h_arr[i] << " ";
    }
    cout << endl;
    cout << "Estimated time : " << estimated_time << "ms." << endl;
    // Deallocate Memory

    free(h_arr);
    cudaFree(d_arr);

    return 0;

}