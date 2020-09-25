
//this program store K elements from all blocks to a device d_array
#include <stdio.h>
using namespace std;

__global__ void reduce_atomic(int *result, int *array, int numElements, int numberThreads)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int sharedMemory[256];

    if (tid < numberThreads)
    {
    	int localSum = 0;

    	for(int i = tid; i < numElements; i += numberThreads)
    		localSum += array[i];

    	sharedMemory[threadIdx.x] = localSum;

    	__syncthreads();

    	if (threadIdx.x == 0)
    	{
    		for(int i = 1; i < blockDim.x; i++)
    			localSum += sharedMemory[i];

        		atomicAdd(result, localSum);
    	}
    }
}

__global__ void reduce_shared(int *result, int *array, int numElements)
{
    __shared__ int sharedMemory[256];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    sharedMemory[threadIdx.x] = (tid < numElements) ? array[tid] : 0;

    __syncthreads();

  // do reduction in shared memory
  for (int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + s];

		__syncthreads();
	}

    // write result for this block to global memory
    if (threadIdx.x == 0)
        atomicAdd(result, sharedMemory[0]);
}

__global__ void min_shared(int *result, int *array, int numElements)
{
    __shared__ int sharedMemory[256];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;



    sharedMemory[threadIdx.x] = (tid < numElements) ? array[tid] : 10000;

    __syncthreads();

  // do reduction in shared memory
  for (int s = blockDim.x/2; s > 7; s >>= 1)
	{
  		if (threadIdx.x < s){
          int min_val= min(sharedMemory[threadIdx.x],sharedMemory[threadIdx.x + s] );
    			sharedMemory[threadIdx.x] = min_val;
      }


		__syncthreads();
	}

__syncthreads();
    // write result for this block to global memory
    if (threadIdx.x == 0)
    {
      printf("thread 0 %d %d %d %d %d\n", sharedMemory[0],sharedMemory[1], sharedMemory[2],sharedMemory[3],sharedMemory[4]);

    }





    if (threadIdx.x == 0)
    {
        atomicMin(result, sharedMemory[0]);

        // << "\n";
        //if(result == sharedMemory[0]){
          //printf("%llu : ");
      //  }

    }

}


__global__ void min_shared2(int *result, int *array, int * d_temp, int numElements, int noOfBlocks)
{
    __shared__ int sharedMemory[256];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    sharedMemory[threadIdx.x] = (tid < numElements) ? array[tid] : 10000;
    __syncthreads();

  // Odd Even
  for (int s = 0 ; s < blockDim.x; s++)
	{
      //odd phase
        if(s %2 ==0 && threadIdx.x % 2 == 0 && threadIdx.x < blockDim.x -1 ){
              int val1= sharedMemory[threadIdx.x];
              int val2= sharedMemory[threadIdx.x + 1];
              sharedMemory[threadIdx.x] = min(val1, val2);
              sharedMemory[threadIdx.x +1 ] = max(val1, val2);
        }

        if( s %2 !=0 && threadIdx.x % 2 != 0 && threadIdx.x < blockDim.x -1 ){
            int val1= sharedMemory[threadIdx.x];
            int val2= sharedMemory[threadIdx.x + 1];
            sharedMemory[threadIdx.x] = min(val1, val2);
            sharedMemory[threadIdx.x +1 ] = max(val1, val2);

        }
		__syncthreads();
	}

__syncthreads();
    // write result for this block to global memory
    if (threadIdx.x == 0)
    {
      printf("thread 0 %d %d %d %d %d   blockIdx.x*blockDim.x =%d\n", sharedMemory[0],sharedMemory[1], sharedMemory[2],sharedMemory[3],sharedMemory[4], blockIdx.x*blockDim.x);
      int idx=0;
      for(int k=blockIdx.x *5 ; k < (blockIdx.x*5) + 5 ; k++){
        d_temp[k]=sharedMemory[idx];
        idx++;
      }

    }

    __syncthreads();
    if (tid == 0)
    {
        for (int i=0; i< 5 * noOfBlocks; i++ ){
          printf(":: %d ", d_temp[i]);
        }


    }

}




int main(int argc, char* argv[])
{
  int numElements = 1e4;
  int K= 5;
  // Allocate host memory
  int *h_array  = (int *)malloc(numElements * sizeof(int));
  int *h_result  = (int *)malloc(K * sizeof(int));



  // Initialize the host input vectors
  for (int i = 0; i < numElements; i++)
      h_array[i] = (i+1);

  h_array[10]=-10;
  h_array[3000]=-10;

  for (int i=0; i<K; i++)
      h_result[i]=100000;
  // Allocate the device input vector
  int *d_array, *d_result;
  cudaMalloc(&d_array, numElements * sizeof(int));
  cudaMalloc(&d_result, K * sizeof(int));

  // Copy the host input vector
  cudaMemcpy(d_array, h_array, numElements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_array, K * sizeof(int), cudaMemcpyHostToDevice);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  int *h_temp = (int *)malloc( K * blocksPerGrid * sizeof(int));
  for (int i=0; i<K * blocksPerGrid; i++)
      h_temp[i]=10000;

  int *d_temp ;
  cudaMalloc(&d_temp, K * blocksPerGrid * sizeof(int));
  //cudaMemcpy(d_temp, h_temp, K * blocksPerGrid * sizeof(int), cudaMemcpyHostToDevice);

  printf("%d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  cudaEventRecord(start);

  min_shared2<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_array, d_temp, numElements, blocksPerGrid);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time %f ms\n\n", milliseconds);

  // Copy the result
  cudaMemcpy(h_result, d_result, K * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<K; i++)
      printf("%d ",h_result[i]);

  cudaError_t cudaError = cudaGetLastError();

  if(cudaError != cudaSuccess)
  {
      fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
      exit(EXIT_FAILURE);
  }

  cudaEventRecord(start);

  int CPU_result = 0;

  for (int i = 0; i < numElements; i++)
    CPU_result += h_array[i];

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\n\n CPU time %f ms\n", milliseconds);

  printf("GPU result %d, CPU result %d, %s!\n", *h_result, CPU_result, *h_result == CPU_result ? "CORRECT" : "ERROR" );

  // Free device global memory
  cudaFree(d_array);
  cudaFree(d_result);

  // Free host memory
  free(h_result);

    return 0;
}

/***



return 0;

int numElements = 1e4;
int K =5;
// Allocate host memory
int *h_array  = (int *)malloc(numElements * sizeof(int));
int *h_result = (int *)malloc(K * sizeof(int));

// Initialize the host input vectors
for (int i = 0; i < numElements; i++)
    h_array[i] = (i+1);


h_array[10]=-10;
h_array[3000]=-10;

for (int i=0; i<K; i++)
    h_result[i]=100000;

// Allocate the device input vector
int *d_array, *d_result;
cudaMalloc(&d_array, numElements * sizeof(int));
cudaMalloc(&d_result, K * sizeof(int));

// Copy the host input vector
cudaMemcpy(d_array, h_array, numElements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemset(d_result, h_result, K * sizeof(int), cudaMemcpyHostToDevice);

// Launch the Vector Add CUDA Kernel
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

printf("%d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
float milliseconds = 0;

cudaEventRecord(start);

min_shared2<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_array, numElements);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU time %f ms\n", milliseconds);

// Copy the result
cudaMemcpy(h_result, d_result, K* sizeof(int), cudaMemcpyDeviceToHost);
for (int i=0; i<K; i++)
    printf("%d ",h_result[i]);

cudaError_t cudaError = cudaGetLastError();

if(cudaError != cudaSuccess)
{
    fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    exit(EXIT_FAILURE);
}

cudaEventRecord(start);

int CPU_result = 0;

for (int i = 0; i < numElements; i++)
  CPU_result += h_array[i];

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
printf("CPU time %f ms\n", milliseconds);

printf("GPU result %d, CPU result %d, %s!\n", *h_result[0], CPU_result, *h_result == CPU_result ? "CORRECT" : "ERROR" );

// Free device global memory
cudaFree(d_array);
cudaFree(d_result);

// Free host memory
free(h_result);


**/
