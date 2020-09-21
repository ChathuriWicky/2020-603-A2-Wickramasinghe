
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdint.h>
#include <iterator>
#include<algorithm>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <bitset>
#include <time.h>
#include <map>
#include <vector>
#include <set>
#include<list>
#include<random>

#define size 10

using namespace std;

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

__global__ void calc_distance_matrix(float *d_dataset, float *d_distance_mat, int no_of_data_records,int no_of_features){
    int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;
    int tid    =  row * no_of_data_records + column; //( blockDim.x * gridDim.x * row ) + column;

    if ( row < no_of_data_records && column < no_of_data_records )
    {

          if (row==column){
            d_distance_mat[tid]=10000;
          }

          else{
            float distance = 0;

              for(int k = 0; k < no_of_features ; k++) // compute the distance between the two instances
              {
                  float diff = d_dataset[row* no_of_features + k] - d_dataset[ column * no_of_features + k];
                  distance += diff * diff;
              }

              distance = sqrt(distance);
              d_distance_mat[tid]=distance;
              //printf("tid: %d  %f\n", tid, distance);
          }
    }

}

/*
__device__ void test(int tid){
  printf("tid: %d \n", tid);
}

__global__ void calc_predictions(float *d_distance_mat, int no_of_data_records,int no_of_features, int K, int *d_class, int *d_predictions ){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < no_of_data_records){
    __shared__ int *temp[size];
    //  __shared__ int temp_class_prediction[no_of_data_records];
      test(tid);
      int predicted_class= 0;
      d_predictions[tid]= predicted_class;

    }


}
*/

int get_mode(int* class_array, int class_array_size) {
  int* ipRepetition = new int[class_array_size];
  for (int i = 0; i < class_array_size; ++i) {
      ipRepetition[i] = 0;
      int j = 0;
      bool bFound = false;
      while ((j < i) && (class_array[i] != class_array[j])) {
          if (class_array[i] != class_array[j]) {
              ++j;
          }
      }
      ++(ipRepetition[j]);
  }
  int index_max_repeated = 0;
  for (int i = 1; i < class_array_size; ++i) {
      if (ipRepetition[i] > ipRepetition[index_max_repeated]) {
          index_max_repeated = i;
      }
  }
  delete [] ipRepetition;
  return class_array[index_max_repeated];


}

int find_smallest_distance_index(float *distances, int no_of_data_records){
  int smallest_index= 0 ;
  for(int i=1; i< no_of_data_records; i++){
      if( distances[i] < distances[smallest_index]){
        smallest_index = i ;
      }
  }
  return smallest_index;
}



int find_class(float* distances, ArffData* dataset,  int K ){
  int no_of_data_records = dataset->num_instances();
  //std::cout << "no_of_data_records: " << no_of_data_records <<" \n";

  int* predictions = (int*)malloc(K * sizeof(int));
  for (int k_idx=0; k_idx <K; k_idx ++){
      int index = find_smallest_distance_index(distances, no_of_data_records);//indexofSmallestElement(distances, dataset->num_instances());
      distances[index]=999999;
      predictions[k_idx]= dataset->get_instance(index)->get(dataset->num_attributes() - 1)->operator int32();
      //printf("index %lu, ", index);
  }
  //printf("k predictions calculated \n");
    int predicted_class = get_mode(predictions, K) ; // or get_mode(predictions, K) or getMode2(predictions, K)
      //printf("mode calculated predicted class%lu\n", predicted_class);
    return predicted_class;
}



int main(int argc, char* argv[])
{

  if(argc != 3)
  {
      cout << "Usage: ./main datasets/datasetFile.arff K" << endl;

      exit(0);
  }
  // Open the dataset
  ArffParser parser(argv[1]);
  ArffData *dataset = parser.parse();
  int K = atoi(argv[2]);





    int no_of_data_records = dataset->num_instances(); // square matrix matrixSize * matrixSize
    int no_of_features = dataset->num_attributes()-1;
    printf("K:%lu no_of_data_records:%d no_of_features:%d\n", K, no_of_data_records, no_of_features);

    // Allocate host memory
    float *h_dataset = (float *)malloc(no_of_data_records* no_of_features * sizeof(float));
    float *h_distance_mat = (float *)malloc(no_of_data_records* no_of_data_records * sizeof(float));
    int *h_class = (int *)malloc(no_of_data_records * sizeof(int));


    // Initialize the host input matrixs
    for(int row=0; row < no_of_data_records; row++){
      for( int col=0; col< no_of_features; col++){
        h_dataset[ (row* no_of_features) + col] = dataset->get_instance(row)->get(col)->operator float();
      }
      h_class[row] = dataset->get_instance(row)->get(dataset->num_attributes() - 1)->operator int32();
    }





    // Allocate the device input matrix A
    float *d_distance_mat, *d_dataset;
    int *d_class, *d_predictions;

    cudaMalloc(&d_dataset, no_of_data_records* no_of_features * sizeof(float));
    cudaMalloc(&d_distance_mat, no_of_data_records* no_of_data_records * sizeof(float));
    cudaMalloc(&d_class, no_of_data_records * sizeof(int));
    cudaMalloc(&d_predictions, no_of_data_records * sizeof(int));

    // Copy the host input to the device input matrixs in
    cudaMemcpy(d_dataset, h_dataset, no_of_data_records* no_of_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class, h_class, no_of_data_records * sizeof(int), cudaMemcpyHostToDevice);


    int threadsPerBlockDim = 32;
    int gridDimSize = (no_of_data_records + threadsPerBlockDim - 1) / threadsPerBlockDim;

    dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
    dim3 gridSize (gridDimSize, gridDimSize);

    printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    calc_distance_matrix<<<gridSize, blockSize>>>(d_dataset, d_distance_mat, no_of_data_records,no_of_features);

    cudaMemcpy(h_distance_mat, d_distance_mat, no_of_data_records* no_of_data_records * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU option 1 time to sum the matrixes %f ms\n", milliseconds);
    //end of matrix calculate
    /*
    for (int i = 0; i < no_of_data_records; ++i)
    {
        for(int j=0;j<no_of_data_records;j++){
          printf("%f ",h_distance_mat[ i*no_of_data_records + j ]);
        }

        printf(" \n");
    }
    */

    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);

    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    for (int i = 0; i < no_of_data_records; ++i)
    {
        float *distances_temp= (float *)malloc(no_of_data_records * sizeof(float));
        //get the distance matrix
        for(int j=0;j<no_of_data_records;j++){
          distances_temp[j] = h_distance_mat[ i*no_of_data_records + j ];
          //printf("%f ,", distances_temp[j] );
        }

        predictions[i]= find_class(distances_temp, dataset, K);
        //printf("%lu ,", predictions[i] );
    }


    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
    uint64_t diff = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;

    printf("\n The KNN with K=%lu classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", K, dataset->num_instances(), (long long unsigned int) diff, accuracy);
    printf("Total runtime: %0.4f", milliseconds+diff);

    /*
    printf("second \n");
    {
    //calc nearest neighbors
    int threadsPerBlockDim = 8;
    int blocksPerGrid = (no_of_data_records + threadsPerBlockDim - 1) / threadsPerBlockDim;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlockDim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    calc_predictions<<<blocksPerGrid, threadsPerBlockDim>>>(d_distance_mat, no_of_data_records,no_of_features, K, d_class, d_predictions);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU option 1 time to sum the matrixes %f ms\n", milliseconds);




    }
    */

    //free cuda memroy

    printf("done\n" );

    return 0;
}


/***
// Allocate host memory
float *h_A = (float *)malloc(numElements * sizeof(float));
float *h_B = (float *)malloc(numElements * sizeof(float));
float *h_C = (float *)malloc(numElements * sizeof(float));
// Initialize the host input matrixs
for (int i = 0; i < numElements; ++i)
{
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
}
// Allocate the device input matrix A
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, numElements * sizeof(float));
cudaMalloc(&d_B, numElements * sizeof(float));
cudaMalloc(&d_C, numElements * sizeof(float));
// Copy the host input matrixs A and B in host memory to the device input matrixs in
cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
float milliseconds = 0;
// Option 1: 2D grid of 2D thread blocks 16x16 (OK)
{
int threadsPerBlockDim = 16;
int gridDimSize = (matrixSize + threadsPerBlockDim - 1) / threadsPerBlockDim;
dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
dim3 gridSize (gridDimSize, gridDimSize);
printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);
cudaEventRecord(start);
matrixAddv1<<<gridSize, blockSize>>>(d_A, d_B, d_C, matrixSize);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU option 1 time to sum the matrixes %f ms\n", milliseconds);
// Copy the device result matrix in device memory to the host result matrix
cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
cudaError_t cudaError = cudaGetLastError();
if(cudaError != cudaSuccess)
{
    fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    exit(EXIT_FAILURE);
}
// Verify that the result matrix is correct
for (int i = 0; i < numElements; i++)
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
printf("Sum of the matrixes was OK\n");
}
// Option 2: 1D grid of 1D thread blocks 1x256 (INEFFICIENT ON PURPOSE), multiple memory transactions!!
{
int threadsPerBlock = 256;
int gridDim = (numElements + threadsPerBlock - 1) / threadsPerBlock; // the dimensionality per grid dimension cannot be larger than 65536 for GPUs using CC 2.0
dim3 blocksize(1, threadsPerBlock);
printf("CUDA kernel launch with %d blocks of 1x%d threads\n", gridDim, threadsPerBlock);
cudaEventRecord(start);
matrixAddv2v3<<<gridDim, blocksize>>>(d_A, d_B, d_C, numElements);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU option 2 time to sum the matrixes %f ms\n", milliseconds);
// Copy the device result matrix in device memory to the host result matrix
cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
cudaError_t cudaError = cudaGetLastError();
if(cudaError != cudaSuccess)
{
    fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    exit(EXIT_FAILURE);
}
// Verify that the result matrix is correct
for (int i = 0; i < numElements; i++)
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
printf("Sum of the matrixes was OK\n");
}
// Option 3: 1D grid of 1D thread blocks (MOST EFFICIENT), smaller number of larger transactions
{
int threadsPerBlock = 256;
int gridDim = (numElements + threadsPerBlock - 1) / threadsPerBlock; // the dimensionality per grid dimension cannot be larger than 65536 for GPUs using CC 2.0
printf("CUDA kernel launch with %d blocks of %dx1 threads\n", gridDim, threadsPerBlock);
cudaEventRecord(start);
matrixAddv2v3<<<gridDim, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU option 3 time to sum the matrixes %f ms\n", milliseconds);
// Copy the device result matrix in device memory to the host result matrix
cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
cudaError_t cudaError = cudaGetLastError();
if(cudaError != cudaSuccess)
{
    fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    exit(EXIT_FAILURE);
}
// Verify that the result matrix is correct
for (int i = 0; i < numElements; i++)
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
printf("Sum of the matrixes was OK\n");
}
// Compute CPU time
cudaEventRecord(start);
for (int i = 0; i < numElements; i++)
h_C[i] = h_A[i] + h_B[i];
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("CPU time to sum the matrixes %f ms\n", milliseconds);
// Free device global memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
// Free host memory
free(h_A);
free(h_B);
free(h_C);
***/
