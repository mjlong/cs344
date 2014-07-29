//Udacity HW 4
//Radix Sorting

#include "utils.h"
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
//#define RADIXBIT 4
#if defined(RADIXBIT)
  #define ANDOPERA 0x1111
  #define NUMBINS  16
#endif

__global__ void createHisto(const unsigned * const data, unsigned* histo, unsigned order, unsigned numElems){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id>=numElems) return;
#if defined(RADIXBIT)
    atomicAdd(histo+ (data[id]>>(order*RADIXBIT))&(ANDOPERA) +blockIdx.x*NUMBINS,1u);
#else
    atomicAdd(histo+ (data[id]>>order)&(0x1) + blockIdx.x*2,1u);
#endif
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
        
    }


}
