/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void minval(const float* const d_logLuminance, float *mins, unsigned pixels){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id>=pixels) return;
  unsigned idl = threadIdx.x;
  extern __shared__ float shared[];
  //size of shared[] is given as 3rd parameter while launching the kernel
  int i,size;
  shared[idl] = d_logLuminance[id];
  __syncthreads();
  size = blockDim.x;
  i = blockDim.x>>1;
  while(i){
    if(idl<i)
      shared[idl]      = min(shared[idl],shared[idl+i]);
    __syncthreads();
    i=i>>1;
  }
  if(0==idl){
    mins[blockIdx.x] = shared[0];
  }
}

__global__ void maxval(const float* const d_logLuminance, float *maxs, unsigned pixels){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id>=pixels) return;
  unsigned idl = threadIdx.x;
  extern __shared__ float shared[];
  //size of shared[] is given as 3rd parameter while launching the kernel
  int i,size;
  shared[idl] = d_logLuminance[id];
  __syncthreads();
  size = blockDim.x;
  i = blockDim.x>>1;
  while(i){
    if(idl<i)
      shared[idl]      = max(shared[idl],shared[idl+i]);
    __syncthreads();
    i=i>>1;
  }
  if(0==idl){
    mins[blockIdx.x] = shared[0];
  }
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  unsigned pixels = numRows*numCols;
  unsigned threads = 512;
  unsigned blocks = (pixels+threads-1)/threads; 
  float *mins = new float[blocks];
  float *maxs = new float[blocks];
  minval<<<blocks,threads, size(float)*threads>>>(mins,d_logLuminance,pixels);
  maxval<<<blocks,threads, size(float)*threads>>>(maxs,d_logLuminance,pixels);
  minval<<<1, blocks, sizeof(float)*blocks>>>(&min_logLum,mins,blocks);
  maxval<<<1, blocks, sizeof(float)*blocks>>>(&max_logLum,maxs,blocks);
  printf("min = %6.3f, max=%6.3f\n", *min_logLum, *max_logLum);    

}
