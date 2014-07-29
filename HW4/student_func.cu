//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <stdio.h>
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#define RADIXBIT 1 //4
#define NUMBINS  2
//#define RADIXBYTE
#if defined(RADIXBYTE)
  #define ANDOPERA 0x1111
  #define NUMBINS  16
#endif

__global__ void createHisto(const unsigned * const data, unsigned* histo, unsigned order, unsigned numElems){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id>=numElems) return;
#if defined(RADIXBYTE)
    atomicAdd(histo+ (unsigned)(data[id]>>(order*RADIXBIT))&(ANDOPERA) +blockIdx.x*NUMBINS,1u);
#else
    atomicAdd(histo+ (unsigned)((data[id]>>order)&(1u)) + blockIdx.x*2,1u);
#endif
}

__global__ void reduceHisto(unsigned *histos){
  unsigned idl = threadIdx.x;
  extern __shared__ unsigned histoidl[];
  int i;
  histoidl[idl] = histos[idl*gridDim.x+blockIdx.x];
  __syncthreads();
  i = gridDim.x>>1;
  while(i){
    if(idl<i)
      histoidl[idl] += histoidl[idl+i];
    __syncthreads();
    i=i>>1;
  }
  if(0==idl){
    histos[blockIdx.x] = histoidl[0];
  }
   
}

__global__ void HillisSteeleScan(unsigned *data, unsigned *d_cdf){
    unsigned idl = threadIdx.x; //blockDim.x=1
    extern __shared__ unsigned datasegment[];
    datasegment[idl] = data[idl];
    __syncthreads();
    for(int step=1;step<blockDim.x;step<<=1){
    if(idl<step)
        datasegment[idl+blockDim.x] = datasegment[idl];
     else
        datasegment[idl+blockDim.x] = datasegment[idl] + datasegment[idl-step];
     __syncthreads();     
     datasegment[idl] = datasegment[idl+blockDim.x];
     __syncthreads();
    }
    
    d_cdf[idl] = datasegment[idl+blockDim.x];
    
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

    printf("There are %d elements to be sorted.\n",numElems);
    /*test 1bit radix*/
    unsigned keybit = sizeof(unsigned)*8;
    unsigned int* d_histos;
    unsigned numHistos = 1024;
    checkCudaErrors(cudaMalloc(&d_histos,sizeof(unsigned)*NUMBINS*numHistos));
    for(unsigned i=0;i<keybit/RADIXBIT;i++){
      createHisto<<<numHistos,(numElems+numHistos-1)/numHistos>>>(d_inputVals, d_histos, i, numElems);     
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());    

      reduceHisto<<<NUMBINS, numHistos, sizeof(unsigned)*numHistos>>>(d_histos);  
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());    

      HillisSteeleScan<<<1, NUMBINS, sizeof(unsigned)*NUMBINS*2>>>(d_histos,d_histos);
      //Algorithm only allows one block, otherwise kernel give segments scanned but not totally scanned
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  
   }


}
