#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "kmeansCPU.cu"

// includes, project
#include <cuda.h>
#define CUDA_SAFE_CALL_NO_SYNC(call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL(call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

////////////////////////////////////////////////////////////////////////

typedef struct CoreLocat{
  int label;
  float *center;
}Core_t;

typedef struct ICD_t{
  float *col;
}ICD_table;

typedef struct RID_t{
  float *col;
}RID_table;

typedef struct SetData{
  int *Group;
  int label;
}Set;

typedef struct ClusterSet_t{
  Set *GroupA;
}ClusterSet;

typedef struct RandCluster_t{
  float *vector;
  int label;
}RandCluster;

//-------------------------------------------------------------------------------------------
//Read data from file

//Initialize the centers of the clusters
Core_t Getcentroid(int dimen, RandCluster *h_sample, Core_t *core, int k, int data){
  for(int i = 0; i < k; i++)
  {
    int randNum = rand()%(data);
    core[i].center = h_sample[randNum].vector;
    core[i].label = i;
  }

  return *core;
}

float dimcalculate(Core_t *core,float **ICD, int k, int dimen){

  for(int i = 0; i < k; i++){
    for(int j=0 ; j<k; j++){
      for(int d=0; d<dimen; d++){
        ICD[i][j] += abs((float)core[i].center[d] -(float) core[j].center[d]);
      }
    }
  }
  return **ICD;
}

RID_table sorting(ICD_table *ICD, RID_table *RID, int s){
  int i, j,m;
  float temp;
  float array[s][s];

  for(i = 0; i < s; i++ )
    for(j = 0; j < s; j++)
     array[i][j] = ICD[i].col[j];


  for(i = 0; i < s; i++ )
    for(j = 0; j < s  ; j++)
      for(m = j+1; m < s; m++){
        if(ICD[i].col[j] > ICD[i].col[m]){
          temp = ICD[i].col[j];
          ICD[i].col[j] = ICD[i].col[m];
          ICD[i].col[m] = temp;
        }
      }

  for(i = 0; i < s; i++ )
    for(j = 0; j < s; j++)
      for(m = 0; m < s; m++){
        if(array[i][m] == ICD[i].col[j])
          RID[i].col[j] = m;
      }
  return *RID;
}

__device__
float dist(int index, float* d_point, float *core, int dimen ){
  float result = 0;
  for(int d=0; d<dimen; d++)
        result += abs(core[d] - d_point[index*dimen+d]);
  return result;
}


__global__
void KmeanKernel(float* d_M, float* d_temp,Core_t *d_core, float *d_point,ICD_table *ICD, RID_table *RID, int k, int dimen,int* label, int Data){

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  int oldCnt;
  float oldDist;
  int newCnt ;
  int curCnt;
  float curDist;
  float newDist = 0;
  int count = 0;
  extern __shared__ int smem[];

  if(i<Data){
     smem[0] = 0;

  for(int in = 0; in < Data; in++)
  {
        smem[in] = count;
        count++;
  };
 __syncthreads();

    oldCnt = label[i];
    oldDist = dist(smem[i],d_point, d_core[oldCnt].center, dimen);
    newCnt = oldCnt;
    newDist = dist(smem[i], d_point,d_core[newCnt].center, dimen);
     for(int j = 2; j < k; j++){
        curCnt = RID[oldCnt].col[j];
        if(ICD[oldCnt].col[curCnt] > 2*oldDist) break;

        curDist = dist(smem[i],d_point, d_core[curCnt].center, dimen);

        if(curDist < newDist){
          newDist = curDist;
          newCnt = curCnt;
        }
      }
    d_M[i] = dist(smem[i], d_point, d_core[newCnt].center, dimen)/Data;

    label[i] = newCnt;
 }
__syncthreads();
}
__global__
void Kmean( float *d_output, int k, int Data, float *d_point, int dimen, int*index, int *count){
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

   if(tid < k){
     for(int a = 0; a<Data; a++){
        if(index[a] == tid)
         atomicAdd(&count[tid],1);
    }


    for(int i = 0; i<Data; i++)
        if(index[i] == tid){
         for(int d =0; d<dimen; d++)
           d_output[tid*dimen+d] += d_point[i*dimen+d] ;
        }

    for(int d = 0; d<dimen; d++){
       if(count[tid] == 0)
        d_output[tid*dimen+d] = d_point[tid*dimen+d];
       else
        d_output[tid*dimen+d] = d_output[tid*dimen+d]/count[tid] ;
    }
 }


}



int main(int argc, char **argv){

    cudaEvent_t ICD_start, ICD_stop;
    cudaEventCreate(&ICD_start);
    cudaEventCreate(&ICD_stop);
    cudaEvent_t RID_start, RID_stop;
    cudaEventCreate(&RID_start);
    cudaEventCreate(&RID_stop);
    cudaEvent_t Kmean_start, Kmean_stop;
    cudaEventCreate(&Kmean_start);
    cudaEventCreate(&Kmean_stop);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t Kmean_Algor_start, Kmean_Algor_stop;
    cudaEventCreate(&Kmean_Algor_start);
    cudaEventCreate(&Kmean_Algor_stop);
  srand(time(NULL));
  if (argc < 2) {
        fprintf(stderr, "Usage: %s dimension numData Cluster\n", argv[0]);
        exit(1);
    }
    int k = atoi(argv[1]);        //Cluster
    printf("Value K: %d\n", k);
//---------------start reading data-------------------------
int *sample;
int numberOfData = 0;
FILE *file = fopen("test.txt", "r");
  if(file == NULL){
        printf("Error opening file!\n");
        exit(1);
  }
int ch;
int numberofLines =0;
   while((ch = fgetc(file))!=EOF)
        if(ch == '\n')  numberofLines++;
   fclose(file);

   file = fopen("test.txt", "r");

   while(fscanf(file, "%d ", &sample) != EOF){
        numberOfData++;
   }
fclose(file);

//Read the data from file 
  int Data = numberofLines;
  int n = numberOfData;
  int m  = numberofLines;
  int dimen = (n/m);

  RandCluster *Points = new RandCluster[ Data];
  for(int i = 0; i < Data; i++)
      Points[i].vector = new float[dimen];

float *FixedData;
FixedData = (float*)malloc(sizeof(float)*Data*dimen);

file = fopen("test.txt","r");
float **temp;
temp = (float**)malloc(Data*sizeof(float*));
        for(int i=0; i<Data; i++)
        temp[i] = (float*)malloc(sizeof(float)*dimen);
while(!feof(file)){
      for(int i = 0; i < Data ; i++){
        for(int j = 0; j < dimen; j++){
           fscanf(file,"%f ",& temp[i][j]);
        Points[i].vector[j] = temp[i][j];
        FixedData[j+i*dimen] = Points[i].vector[j];
        Points[i].label = 0;
        }
     }
}
 fclose(file);


//(----------------Read data ends------------------


  Core_t *h_core;
  

  float **ICD;  // kxk matrix stnt c = 0; c < k; c++){

  ICD = (float**)malloc(k*sizeof(float*));
        for(int i = 0; i<k; i++)
                ICD[i] = (float*)malloc(sizeof(float)*k);

  for(int i = 0; i < k; i++)
   for(int j = 0; j < k; j++)
        ICD[i][j] = 0;
//initialize function set
  h_core = (Core_t*)malloc(sizeof(Core_t)*k);
  for(int i = 0; i < k ; i++)
    h_core[i].center = (float*)malloc(sizeof(float)*dimen);

 ICD_table *ICD_row = (ICD_table*)malloc(sizeof( ICD_table)*k);
  for(int i = 0; i < k ; i++)
    ICD_row[i].col = new float[k];

  RID_table *RID_row = new RID_table[k];
  for(int i = 0; i < k ; i++)
    RID_row[i].col = new float[k];

   for(int i=0; i<k; i++)
    for(int j=0; j<k; j++){
     ICD_row[i].col[j] = 0;
  }
//-----------------Algorithm Two-----------------------------------------------

//---Initialize parts for algotrihtm two-----------------------------------------

 float* d_output;
 int  *d_index;
 int  *h_count = (int*)malloc(sizeof(int)*k);
 float *d_input;
 float *h_output = new float[Data*dimen];
 int *count;
 int *zero = new int[k];
 for(int i = 0; i < k; i++) zero[i] = 0;
 int threadsPerblock = 32;
 int numBlocks = (Data + threadsPerblock - 1) / threadsPerblock;
  float *d_point;
  int *initial = new int[Data];
  int *d_Fixedlabel;
  float *h_temp = new float[dimen];
  float *d_temp;
  float *T_inti = new float[dimen];
  for(int i = 0; i < dimen; i++) T_inti[i] = 0;
 int *compareLabel = new int[Data];
 for(int i = 0; i < Data; i++) compareLabel[i] = 0;

  int mm = 0;
   for(int i = 0; i < k; i++){
    for(int j = 0; j < Data/k; j++){
    initial[j+i*Data/k] = mm;
    }
    mm++;
   }

  float *h_M =( float*)malloc(sizeof(float)*Data);
  float *d_M ;
  float *initial_M = new float[Data];
  for(int i = 0; i < Data; i++)
        initial_M[i] = 0;

  float *h_sum_M = (float*)malloc(sizeof(float));
  int *Fixedlabel = new int[Data];

 //--------------------------device core define-------------------------------
  Core_t *d_core;
  Core_t *core = (Core_t*)malloc(sizeof(Core_t)*k);
  float *d_center;
//--------------------------device core define-------------------------------
  Core_t *d_core;
  Core_t *core = (Core_t*)malloc(sizeof(Core_t)*k);
  float *d_center;

  //device matrix ICD and RID----
  ICD_table *d_DDC;
  ICD_table *DDC = (ICD_table*)malloc(sizeof(ICD_table)*k);
  float *ICD_Value;

  RID_table *d_RRD;
  RID_table *RRD = (RID_table*)malloc(sizeof(RID_table)*k);
  float *RRD_Value;
//------------algorithm two---------------------------------------------------
  Getcentroid(dimen, Points ,h_core, k, Data);

  int Jump = 0;

  cudaEventRecord(start,0);
  do{

  cudaEventRecord(ICD_start, 0);
  dimcalculate(h_core, ICD, k, dimen);
  cudaEventRecord(ICD_stop, 0);

  cudaEventRecord(RID_start, 0);
  sorting(ICD_row, RID_row, k);
  cudaEventRecord(RID_stop, 0);

  for(int d = 0; d < k; d++){
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_core,  k*sizeof(Core_t)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_center,  dimen*sizeof(float)));
    CUDA_SAFE_CALL( cudaMemcpy( d_center, h_core[d].center,  dimen*sizeof(float),  cudaMemcpyHostToDevice) );
    core[d].label = h_core[d].label;
    core[d].center = d_center;
    CUDA_SAFE_CALL( cudaMemcpy( d_core, core,  k*sizeof(Core_t),  cudaMemcpyHostToDevice) );
  }
  for(int d = 0; d < k; d++){
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_DDC,  k*sizeof(ICD_table)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_RRD,  k*sizeof(RID_table)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &ICD_Value,  k*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &RRD_Value,  k*sizeof(float)));
    CUDA_SAFE_CALL( cudaMemcpy( ICD_Value, ICD_row[d].col, k*sizeof(float),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( RRD_Value, RID_row[d].col, k*sizeof(float),  cudaMemcpyHostToDevice) );
    DDC[d].col = ICD_Value;
    RRD[d].col = RRD_Value;
    CUDA_SAFE_CALL( cudaMemcpy( d_DDC, DDC, k*sizeof(ICD_table),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_RRD, RRD, k*sizeof(RID_table),  cudaMemcpyHostToDevice) );
  }

 //---------------------------------------------------------------------------
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_M, Data*sizeof(int)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Fixedlabel, Data*sizeof(int)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_temp, dimen*sizeof(int)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_point,  Data*dimen*sizeof(float)));
    CUDA_SAFE_CALL( cudaMemcpy( d_Fixedlabel, initial, Data*sizeof(int),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_point, FixedData, dimen*Data*sizeof(float),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_temp, T_inti, dimen*sizeof(int),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_M, h_M, Data*sizeof(float),  cudaMemcpyHostToDevice) );
 cudaEventRecord(Kmean_Algor_start, 0);
 KmeanKernel<<<numBlocks, threadsPerblock, Data*sizeof(int)>>>(d_M,d_temp,d_core, d_point, d_DDC, d_RRD, k, dimen, d_Fixedlabel, Data);
 cudaEventRecord(Kmean_Algor_stop, 0);
    CUDA_SAFE_CALL( cudaMemcpy( Fixedlabel, d_Fixedlabel, Data*sizeof(int),  cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( h_M, d_M, Data*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &count, k*sizeof(int)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_input, Data*dimen*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_output, k*dimen*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_index,  Data*sizeof(int)));
    CUDA_SAFE_CALL( cudaMemcpy( d_index, Fixedlabel, Data*sizeof(int),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_input, FixedData, Data*dimen*sizeof(float),  cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( count, zero, k*sizeof(int),  cudaMemcpyHostToDevice) );
    float sum_M = 0;
    for(int i = 0; i < Data; i++){
       sum_M += h_M[i];
    }

    cudaEventRecord(Kmean_start, 0);
    Kmean<<<numBlocks, threadsPerblock>>>( d_output, k, Data, d_input, dimen, d_index, count);
    cudaEventRecord(Kmean_stop, 0);

    CUDA_SAFE_CALL( cudaMemcpy( h_output, d_output, k*dimen*sizeof(float),  cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( h_count, count, k*sizeof(float),  cudaMemcpyDeviceToHost) );
  for(int i =0; i < k; i++)
   for(int j = 0; j < dimen; j++)
      h_core[i].center[j] = h_output[i*dimen+j];


  if(compareLabel == Fixedlabel) Jump++;
  else Jump = 0;
    if(Jump > 100) break;

   compareLabel = Fixedlabel;

    for(int i = 0; i < k; i ++){
      if(h_count[i] == 0){
        Getcentroid(dimen, Points ,h_core, k, Data);
      }
     }
} while(1);
cudaEventRecord(stop,0);
cudaEventSynchronize(start);
cudaEventSynchronize(ICD_start);
cudaEventSynchronize(RID_start);
cudaEventSynchronize(Kmean_start);
cudaEventSynchronize(Kmean_Algor_start);

float Total_time;
float ICD_time;
float RID_time;
float Kmean_Algor_time;
float Kmean_time;

cudaEventElapsedTime(&Total_time, start, stop);
cudaEventElapsedTime(&ICD_time, ICD_start, ICD_stop);
cudaEventElapsedTime(&RID_time, RID_start, RID_stop);
cudaEventElapsedTime(&Kmean_Algor_time, Kmean_Algor_start, Kmean_Algor_stop);
cudaEventElapsedTime(&Kmean_time, Kmean_start, Kmean_stop);
//for(int i = 0; i < k ; i++)
// printf("%d\t", h_count[i]);

   printf("\n");
    for(int i = 0; i< k; i++){
      printf("Cluster center : [%d] ", i);
     for(int j = 0; j < dimen; j++){
      printf("%f\t", h_output[i*dimen+j]);
      h_core[i].center[j] = h_output[i*dimen+j];
     }
     printf("\n");
    }

printf("Total Processing time: %f\n", Total_time);
printf("ICD table processing time (one iteration): %f\n", ICD_time);
printf("RID table processing time (one iteration); %f\n", RID_time);
printf("Kmean Algorithm processing time (one iteration: %f\n", Kmean_Algor_time);
printf("Kmean calculation processing time (one iteration: %f\n", Kmean_time);


}


