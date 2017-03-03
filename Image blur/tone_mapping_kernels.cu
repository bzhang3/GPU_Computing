#include "fast_histogram_kernels.cu"

__host__ __device__
void rgb_to_xyY(float r, float g, float b, float* x, float* y, float* logY) {
    float delta = .0001f;

    float X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
    float Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
    float Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );

    float L = X + Y + Z;
    *x = X / L;
    *y = Y / L;

    *logY = log10f( delta + Y );
}
/*
__global__ void
incorrectHistogram(int * d_bins, const float* d_logY, int numBins, int image_size, float
rangeLogY, float* d_minlogY,unsigned int* d_cdf) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < image_size) {
        int bin = whichBin((d_logY[i] - *d_minlogY)/rangeLogY, numBins);
       // d_bins[bin]++;
        atomicAdd(&d_bins[bin],1);
    }
   // thrust::exclusive_scan(d_bins,d_bins+numBins,d_cdf);
}
*/
__global__ void Xclusivescan(int* d_bins, int image_size,unsigned int* d_cdf,int numBins )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx<numBins)
  {
    d_cdf[idx]=0;
  }
  for(int i=0;i<idx;i++)
  {
    d_cdf[idx] = d_cdf[idx]+d_bins[i];
  }
}

__global__ void findmax(float*d_localmax,int image_size,float* d_maxlogY)
{
  int allblocks=(image_size+1023)/1024;
  int fullsize = allblocks*1024;
  int iterate = fullsize/1024;
  int idx = threadIdx.x;
  __shared__ float temp[1024];
  float* findmaxarray = new float[iterate];
  temp[idx]=0;
  if((idx >= image_size) && (idx < (fullsize-1)))
  {
    d_localmax[idx] = -99;
  }
  for(int i=0;i<iterate;i++)
  {
    if (d_localmax[iterate*idx+i] >temp[idx])
    {
      temp[idx]=d_localmax[iterate*idx+i];
    }
    __syncthreads();
  }
  for(unsigned int s = 1024/2;s>0;s=s/2)
  {
    if(idx<s)
    {
      if(temp[idx] < temp[idx+s])
      {
        temp[idx] = temp[idx+s];
      }
      __syncthreads();
    }
  }
  *d_maxlogY=temp[0];
}

__global__ void rgb( int image_size,float* d_r,float* d_g,float* d_b,float* d_x,float*
d_y,float* d_logY)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //int threadid = threadIdx.x;
  float R,G,B;
  int fullblocks=image_size/256;
  int allblocks=(image_size+255)/256;
  int fullsize = allblocks*256;
  int remaining=image_size%256;
  //int remaining_thread_start = 256 * fullblocks + remaining;
  /*if(idx<image_size)
  {
    d_r[idx] = image[3*idx];
    d_g[idx] = image[3*idx+1];
    d_b[idx] = image[3*idx+2];
  }*/
  if(idx < image_size)
  {
    R=d_r[idx];
    G=d_g[idx];
    B=d_b[idx];
    rgb_to_xyY(R,G,B,&d_x[idx],&d_y[idx],&d_logY[idx]);
  }
}


__global__ void step_one( int image_size,float* d_logY,float* d_localmin,float* d_minlogY)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //int threadid = threadIdx.x;
  //float R,G,B;
  int fullblocks=image_size/256;
  int allblocks=(image_size+255)/256;
  int fullsize = allblocks*256;
  int remaining=image_size%256;
  if(idx < image_size)
  {
    d_localmin[idx] = d_logY[idx];
  }
  else if((idx >= image_size) && (idx < (fullsize-1)))
  {
    d_localmin[idx] = 99;
  }
 for(unsigned int s = fullsize/2;s>0;s=s/2) 
  {
    if(idx<s)
    {
      if(d_localmin[idx] > d_localmin[idx+s])
      {
        d_localmin[idx] = d_localmin[idx+s];
      }
    }
  }
  *d_minlogY = d_localmin[0];
}

void computeHistogramAndCDF(float *hdrImage, int width, int height, int numBins, unsigned int* d_cdf, float& minLogY, float& maxLogY) 
{
  int image_size = width*height;
  float img_size = sizeof(float)*image_size;
  int blocks = (image_size+255)/256;
//step1
  float* d_image;
  float* d_r;
  float* d_g;
  float* d_b;
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_r,image_size*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_g,image_size*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_b,image_size*sizeof(float)));
  
  static float* d_x;
  static float* d_y;
  static float* d_logY;
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_x,image_size*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_y,image_size*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_logY,image_size*sizeof(float)));

  float* d_localmin;
  float* d_localmax;
  float* d_minlogY;
  float* d_maxlogY;
  float range;
  int* d_bins;
  float* h_r = new float[image_size];
  float* h_g = new float[image_size];
  float* h_b = new float[image_size];
  int* h_bins = new int[numBins];
  for(int i=0;i<image_size;i++)
  {
    h_r[i] = hdrImage[3*i];
    h_g[i] = hdrImage[3*i+1];
    h_b[i] = hdrImage[3*i+2];
  }
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_image,3*image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_r,image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_g,image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_b,image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_x,image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_y,image_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_logY,image_size*sizeof(float)));//initialize r,g,b,x,y,logY
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_localmin,blocks*256*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_localmax,blocks*256*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_minlogY,1*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_maxlogY,1*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_bins,numBins*sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_r,h_r,image_size*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_g,h_g,image_size*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_b,h_b,image_size*sizeof(float),cudaMemcpyHostToDevice));
  rgb<<<blocks,256>>>(image_size,d_r,d_g,d_b,d_x,d_y,d_logY);
  float *h_logY = (float*)malloc(image_size*sizeof(float));
  CUDA_SAFE_CALL( cudaMemcpy( h_logY,d_logY,image_size*sizeof(float),cudaMemcpyDeviceToHost) );
  float temp = h_logY[0];
   for(int a=0;a<width*height;a++)
   {
     
    if(temp>h_logY[a]){temp=h_logY[a];}
   }
   printf("cpu test min: %f\n",temp);
//step2 find min 
  //float* d_localmin;
  //float* d_localmax;
  //float* d_minlogY;
  //float* d_maxlogY;

  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_localmin,blocks*256*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_localmax,blocks*256*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_minlogY,1*sizeof(float)));
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_maxlogY,1*sizeof(float)));
  
  step_one<<<blocks,256>>>(image_size,d_logY,d_localmin,d_minlogY);
  CUDA_SAFE_CALL(cudaMemcpy(&minLogY,d_minlogY,1*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_minlogY));
  CUDA_SAFE_CALL(cudaFree(d_localmin));
//find max
  findmax<<<1,1024>>>(d_localmax,image_size,d_maxlogY);
  CUDA_SAFE_CALL(cudaMemcpy(&maxLogY,d_maxlogY,1*sizeof(float),cudaMemcpyDeviceToHost)); 
  printf("after kernel\n");
  printf("%f    %f   %f  %f   %f\n",minLogY,maxLogY,h_logY[123],h_logY[560],hdrImage[2]);
//step3  
  range = maxLogY-minLogY;
//step4 
  //incorrectHistogram<<<blocks,256>>>(d_bins,d_logY,numBins,image_size,range,d_minlogY,d_cdf);
  // CUDA_SAFE_CALL(cudaMalloc((void**) &d_bins,numBins*sizeof(int)));
   float *element = (float*)malloc(img_size);
   for(int find=0;find<image_size;find++)
   {
    element[find] = (h_logY[find]-minLogY)/range; 
   }
   float *d_element;
   CUDA_SAFE_CALL(cudaMalloc( (void**) &d_element, img_size));
   CUDA_SAFE_CALL( cudaMemcpy( d_element,element,img_size,cudaMemcpyHostToDevice) );
   incorrectHistogram<<<blocks,256>>>(d_bins,d_element,numBins,image_size);
   //histogram(numBins, img_size,d_bins,d_element); 
//step5
  Xclusivescan<<<1,1024>>>(d_bins,image_size,d_cdf,numBins);
  CUDA_SAFE_CALL(cudaMemcpy(h_bins,d_bins,numBins*sizeof(int),cudaMemcpyDeviceToHost));

//cleanup mem
  //CUDA_SAFE_CALL(cudaFree(d_minlogY));
  CUDA_SAFE_CALL(cudaFree(d_maxlogY));
  CUDA_SAFE_CALL(cudaFree(d_localmax));
  //CUDA_SAFE_CALL(cudaFree(d_localmin));
}
