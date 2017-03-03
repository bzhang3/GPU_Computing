// TODO Declare any extra pointers as global variables (if needed)


// Allocates any extra memory
// This is a host-side function
void setup(int numBins, int numElements)
{
    // TODO Implement this function (if needed)
}

// Free the memory allocated in the setup() function
// This is a host-side function
void finish()
{
    // TODO Implement this function (if needed)
}

__global__ void incorrectHistogram(int *d_bins, const float*d_idata,int numBins,int numElements)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < numElements) 
    {
        int bin = whichBin(d_idata[i], numBins);
        atomicAdd(&d_bins[bin],1);
    }
}
// Generate a histogram into d_bins for the data in d_idata
// This is a host-side function.  It should call your kernel(s) to generate the histogram
//     This function should NOT free d_bins
//     This function should NOT free and NOT modify d_idata
void histogram(int numBins, int numElements, int* d_bins, const float* d_idata)
{
    // TODO Implement this function
    int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    incorrectHistogram<<<blocks,threadsPerBlock>>>(d_bins,d_idata,numBins,numElements);
}
