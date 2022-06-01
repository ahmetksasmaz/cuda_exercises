#include <iostream>
#include "cuda_runtime.h"

#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32

using namespace std;

__global__ void transpose_serial(int* d_matrix_in, int* d_matrix_out, int width, int height){
    
    if(blockDim.x*blockIdx.x+threadIdx.x < width && blockDim.y*blockIdx.y+threadIdx.y < height){
        int i = (blockIdx.y*blockDim.y+threadIdx.y)*width+blockIdx.x*blockDim.x+threadIdx.x;
        int j = (blockIdx.x*blockDim.x+threadIdx.x)*height+blockIdx.y*blockDim.y+threadIdx.y;
        d_matrix_out[j] = d_matrix_in[i];
    }

}

__global__ void transpose_tile(int* d_matrix_in, int* d_matrix_out, int width, int height){
    
    __shared__ int s_tile[BLOCK_HEIGHT][BLOCK_WIDTH];

    if(blockDim.x*blockIdx.x+threadIdx.x < width && blockDim.y*blockIdx.y+threadIdx.y < height){

        int i = (blockIdx.y*blockDim.y+threadIdx.y)*width+blockIdx.x*blockDim.x+threadIdx.x;
        int j = (blockIdx.x*blockDim.x+threadIdx.x)*height+blockIdx.y*blockDim.y+threadIdx.y;

        s_tile[threadIdx.y][threadIdx.x] = d_matrix_in[i];

        __syncthreads();
        
        d_matrix_out[j] = s_tile[threadIdx.x][threadIdx.y];
    }

}

int main(int argc, char* argv[]){

    if(argc != 3){
        cout << "Usage : exercise_matrix_transpose [0|1] experiment_count" << endl;
        return -1;
    }

    // Initialize Variables
    int* h_matrix_in = 0;
    int* h_matrix_out = 0;
    int* d_matrix_in = 0;
    int* d_matrix_out = 0;
    int width;
    int height;
    int numbytes;
    float time;
    cudaEvent_t start, stop;
    dim3 grid, block;

    width = 1;
    height = 1;

    for(int i = 0; i < 13; i++){
        width *= 2;
        height *= 2;

        float time_sum = 0.0;
        float time_avg;

        for(int i = 0; i < atoi(argv[2]); i++){

            block.x = BLOCK_WIDTH;
            block.y = BLOCK_HEIGHT;
            grid.x = (width / block.x) + ( width % block.x ? 1 : 0 );
            grid.y = (height / block.y) + ( height % block.y ? 1 : 0 );

            // Allocate Memory
            numbytes = width * height * sizeof(int);
            h_matrix_in = (int*)malloc(numbytes);
            h_matrix_out = (int*)malloc(numbytes);
            cudaMalloc((void**)&d_matrix_in, numbytes);
            cudaMalloc((void**)&d_matrix_out, numbytes);

            // Set Matrix
            /*
            int index;
            cout << "Original Matrix" << endl;
            for(int i = 0; i < height; i++){
                for(int j = 0; j < width; j++){
                    index = i * width + j;
                    h_matrix_in[index] = index;
                    cout << index << " ";
                }
                cout << endl;
            }
            cout << endl;
            */
            // Copy Memory
            cudaMemcpy(d_matrix_in, h_matrix_in, numbytes, cudaMemcpyHostToDevice);

            // Launch Kernel
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            if(atoi(argv[1])){
                transpose_tile<<<grid, block, BLOCK_HEIGHT * BLOCK_WIDTH * sizeof(int)>>>(d_matrix_in, d_matrix_out, width, height);
            }
            else{
                transpose_serial<<<grid, block>>>(d_matrix_in, d_matrix_out, width, height);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&time, start, stop);
            time_sum += time;

            // Copy Memory
            cudaMemcpy(h_matrix_out, d_matrix_out, numbytes, cudaMemcpyDeviceToHost);

            // Print Matrix and Statistics
            /*
            cout << "Transposed Matrix" << endl;
            for(int i = 0; i < width; i++){
                for(int j = 0; j < height; j++){
                    cout << h_matrix_out[i * height + j] <<" ";
                }
                cout << endl;
            }
            cout << endl;
            cout << "Elapsed time : " << time << "ms." << endl;
            */
            // Deallocate Memory
            free(h_matrix_in);
            free(h_matrix_out);
            cudaFree(d_matrix_in);
            cudaFree(d_matrix_out);

        }

        time_avg = time_sum / atoi(argv[2]);

        cout << "Average time for [" << width << "x" << height << "] :" << time_avg << "ms." << endl;

    }

    return 0;
}