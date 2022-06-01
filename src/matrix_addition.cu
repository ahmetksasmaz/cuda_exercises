#include <iostream>
#include "cuda_runtime.h"

using namespace std;

__global__ void add(int * a, int * b, int * c){

    int index = threadIdx.y * blockDim.y + threadIdx.x;
    c[index] = a[index] + b[index];

}

int main(){

    // Set Variables
    int matrix_w = 4, matrix_h = 4;
    int num_of_bytes = matrix_w * matrix_h * sizeof(int);
    dim3 grid, block;
    grid.x = 1;
    block.x = 4;
    block.y = 4;

    int * h_a = 0;
    int * h_b = 0;
    int * h_c = 0;

    int* d_a = 0;
    int* d_b = 0;
    int* d_c = 0;

    // Allocate Memory
    h_a = (int *)malloc(num_of_bytes);
    h_b = (int *)malloc(num_of_bytes);
    h_c = (int *)malloc(num_of_bytes);

    cudaMalloc((void **)&d_a, num_of_bytes);
    cudaMalloc((void **)&d_b, num_of_bytes);
    cudaMalloc((void **)&d_c, num_of_bytes);

    // Set Data
    for(int i = 0; i < matrix_h; i++){
        for(int j = 0; j < matrix_w; j++){
            h_a[i*matrix_h+j] = i;
            h_b[i*matrix_h+j] = j;
        }
    }

    // Copy Memory
    cudaMemcpy(d_a, h_a, num_of_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_of_bytes, cudaMemcpyHostToDevice);

    // Launch Kernel
    add<<<grid, block>>>(d_a, d_b, d_c);

    // Get Data
    cudaMemcpy(h_c, d_c, num_of_bytes, cudaMemcpyDeviceToHost);

    // Print Data
    for(int i = 0; i < matrix_h; i++){
        for(int j = 0; j < matrix_w; j++){
            cout << h_c[i*matrix_h+j] << " ";
        }
        cout << endl;
    }

    // Deallocate Memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}