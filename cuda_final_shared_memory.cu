
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

    if ( row < no_of_data_records && column < no_of_data_records && column >= row)
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
              d_distance_mat[column * no_of_data_records + row]= distance;
              //printf("tid: %d  %f\n", tid, distance);
          }
    }

}

__global__ void finding_class_shared(float *d_distance_mat, int *d_class, int no_of_data_records, int i, int K, float *temp_array1, int *temp_array2, int *d_predictions, int noOfBlocks, int *K_class_array){
      __shared__ float sharedMemory[256];
      __shared__ int sharedMemory_class[256];

      int tid = blockIdx.x*blockDim.x + threadIdx.x;
      sharedMemory[threadIdx.x] = (tid < no_of_data_records) ? d_distance_mat[i * no_of_data_records + tid] : 10000;
      sharedMemory_class[threadIdx.x] = (tid < no_of_data_records) ? d_class[tid] : 0;
      __syncthreads();
      //ODD EVEN sort

      for (int s = 0 ; s < blockDim.x; s++)
    	{
          //odd phase
            if(s %2 ==0 && threadIdx.x % 2 == 0 && threadIdx.x < blockDim.x -1 ){
                  float val1= sharedMemory[threadIdx.x];
                  float val2= sharedMemory[threadIdx.x + 1];
                  if (val2 < val1){
                    sharedMemory[threadIdx.x] = val2;
                    sharedMemory[threadIdx.x + 1] = val1;
                    int class1 =  sharedMemory_class[threadIdx.x];
                    sharedMemory_class[threadIdx.x] = sharedMemory_class[threadIdx.x +1];
                    sharedMemory_class[threadIdx.x + 1] = class1;
                  }

            }

            if( s %2 !=0 && threadIdx.x % 2 != 0 && threadIdx.x < blockDim.x -1 ){
              float val1= sharedMemory[threadIdx.x];
              float val2= sharedMemory[threadIdx.x + 1];
              if (val2 < val1){
                sharedMemory[threadIdx.x] = val2;
                sharedMemory[threadIdx.x + 1] = val1;
                int class1 =  sharedMemory_class[threadIdx.x];
                sharedMemory_class[threadIdx.x] = sharedMemory_class[threadIdx.x +1];
                sharedMemory_class[threadIdx.x + 1] = class1;
              }

            }
    		__syncthreads();
    	}

      __syncthreads();

      // write result for this block to global memory
      if (threadIdx.x == 0)
      {
            int idx=0;
            for(int k=blockIdx.x * K ; k < (blockIdx.x * K) + K ; k++){
              temp_array1[k]=sharedMemory[idx];
              temp_array2[k]=sharedMemory_class[idx];
              idx++;
            }

      }
      __syncthreads();
      if (tid == 0)
      {
            for (int j=0; j<K; j++){
                int min_idx=0;
                for (int i=0; i< K * noOfBlocks; i++ ){
                  if(temp_array1[i]< temp_array1[min_idx]){
                    min_idx=i;
                  }
                  //printf(":: %f ", temp_array1[i]);
                }
                temp_array1[min_idx]=10000;
                K_class_array[j] = temp_array2[min_idx];
            }

            int count = 1;
            int max = 0;
            int mode = K_class_array[0];
            for (int i = 0; i < K - 1; i++)
            {
               if ( K_class_array[i] == K_class_array[i+1] )
               {
                  count++;
                  if ( count > max )
                  {
                      max = count;
                      mode = K_class_array[i];
                  }
               } else
                  count = 1;
            }
            //printf("\n:: pedicted class %d \n", mode);
            d_predictions[i] = mode;

      }

}



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

    // Open the dataset and read K
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
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));



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
    int *k_class_array;

    //Allocate devie memory
    cudaMalloc(&d_dataset, no_of_data_records* no_of_features * sizeof(float));
    cudaMalloc(&d_distance_mat, no_of_data_records* no_of_data_records * sizeof(float));
    cudaMalloc(&d_class, no_of_data_records * sizeof(int));
    cudaMalloc(&d_predictions, no_of_data_records * sizeof(int));
    cudaMalloc(&k_class_array, K * sizeof(int));



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
    printf("GPU section takes only to calculate distance matrix %f ms\n", milliseconds);


    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);


    int threadsPerBlock = 256;
    int blocksPerGrid = (no_of_data_records + threadsPerBlock - 1) / threadsPerBlock;

    float* temp_array1; //for k distances
    int* temp_array2 ; // for class labels
    cudaMalloc(&temp_array1, K * blocksPerGrid * sizeof(float));
    cudaMalloc(&temp_array2, K * blocksPerGrid * sizeof(int));

    //float total_time=0;
    for (int i = 0; i < no_of_data_records; ++i)
    {

      cudaEventRecord(start);
      finding_class_shared<<<blocksPerGrid, threadsPerBlock>>>(d_distance_mat, d_class, no_of_data_records, i, K, temp_array1, temp_array2, d_predictions, blocksPerGrid, k_class_array);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);


    }
    cudaMemcpy(predictions, d_predictions, no_of_data_records*  sizeof(int), cudaMemcpyDeviceToHost);
    //


    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);
    uint64_t diff = (1000000000L * (end_cpu.tv_sec - start_cpu.tv_sec) + end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
    printf("Second GPU section takes only %llu ms\n", diff);


    printf("\n The KNN with K=%lu classifier for %lu instances, accuracy was %.4f\n", K, dataset->num_instances(), accuracy);
    printf("Total runtime is: %0.4f", milliseconds+diff);



    //free cuda memroy

    printf("done\n" );
    free(h_dataset);
    free(h_distance_mat);
    free(h_class);
    free(predictions);
    cudaFree(d_distance_mat);
    cudaFree(d_dataset);
    cudaFree(d_class);
    cudaFree(d_predictions);
    return 0;
}
