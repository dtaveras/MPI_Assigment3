/* Copyright 2014 15418 Staff */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>

#include "parallelSort.h"

//Text Color Options
#define BLACK 0
#define RED 1
#define GREEN 2
#define YELLOW 3
#define BLUE 4
#define MAGENTA 5
#define CYAN 6
#define WHITE 7

#define SAMP_THRESHOLD 100
#define MIN_WORKLOAD 4 
#define ROOT 0
#define MAX_NAME_SIZE 256

#define LOG_10(X) (log(X))/(log(10))

#define DBG_3
//#define DBG_1
//#define DBG_2

using namespace std;

void textcolor(unsigned int color)
{
  printf("%c[%d;%d;%dm",0x1B,0,30 + color,40);
}

void printArr(const char* arrName, int *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %d %d %d %d\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void printArr(const char* arrName, float *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %f %f %f %f\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void randomSample(float *data, size_t dataSize, float *sample, size_t sampleSize) {
  for (size_t i=0; i<sampleSize; i++) {
    sample[i] = data[rand()%dataSize];
  }
}
  // Implement parallel sort algorithm as described in assignment 3
  // handout. 
  // Input:
  //  data[]: input arrays of unsorted data, distributed across p processors
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //
  // Step 1: Choosing Pivots to Define Buckets
  // Step 2: Bucketing Elements of the Input Array
  // Step 3: Redistributing Elements
  // Step 5: Final Local Sort
  // ***********************************************************************
  // Output:
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!

void parallelSort(float *data, float *&sortedData, int procs, 
		  int procId, size_t dataSize, size_t &localSize) {

  if(localSize == 0) return;
  //size_t sampSize = dataSize/2;
  //size_t locSampleSize = sampSize/procs; // Local Sample Size
  size_t locSampleSize = (size_t)float(12*LOG_10(dataSize));
  if(locSampleSize >= localSize){
    printf("locSampleSize >= localSize:true\n");
    locSampleSize = (2*LOG_10(localSize))*(localSize/50 + 1) + 1;
  }
  size_t sampSize = locSampleSize*procs;

#ifdef DBG_3
  if(procId == 0){
    textcolor(RED);
    printf("---------------------------------------\n");
    printf("DataSize: %d    ",dataSize);
    printf("Original Local Size: %d\n", localSize);
    printf("Sample Size: %d     ", sampSize);
    printf("Local Sample Size: %d\n", locSampleSize);
    printf("---------------------------------------\n");
    textcolor(WHITE);
  }
#endif

  float* Samples;
  if(procId == 0)
    Samples = (float*)malloc(sizeof(float)*locSampleSize*procs);

  float* randLocSamples = (float*)malloc(sizeof(float)*locSampleSize);
  randomSample(data, localSize, randLocSamples, locSampleSize);

#ifdef DBG_1
  textcolor(CYAN);
  printArr("RandomSample: ", randLocSamples, locSampleSize, procId);
  textcolor(WHITE);
#endif

  MPI_Gather(randLocSamples, locSampleSize, MPI_FLOAT, Samples,
	     locSampleSize, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

  float* pivots;
  //Only Processor 0 is working now
  if(procId == ROOT){
#ifdef DBG_1
    textcolor(YELLOW);
    printArr("Samples: ", randLocSamples, locSampleSize, procId);
    textcolor(WHITE);
#endif

    //NLOG(N) sorting time
    sort(Samples, Samples + locSampleSize*procs);

    pivots = (float*)malloc(sizeof(float)*(procs-1));
    pivots[0] = Samples[1];
    for(int i=1; i < procs - 1; i++){
      pivots[i] = Samples[(sampSize/procs)*i];
    }

#ifdef DBG_3
    //printArr("Data: ", data, localSize, procId);
    printf("Number of Pivots: %d ",procs-1);
    printArr("Pivots:", pivots, procs-1, procId);
#endif    

    //Send the pivots to all the other processes
    for(int i=1; i < procs; i++){
      MPI_Send(pivots, procs-1, MPI_FLOAT, i, i, MPI_COMM_WORLD);
    }

#ifdef DBG_1
    textcolor(YELLOW);
    printArr("Samples Sorted: ", Samples, locSampleSize*procs, procId);
    textcolor(WHITE);
#endif
  }
  else{
#ifdef DBG_1
    textcolor(MAGENTA);
    printf("ProcID:%d\n", procId);
#endif
    MPI_Status stat;
    //We know this size ahead of time so condsider making it static
    pivots = (float*)malloc(sizeof(float)*(procs-1));

    //Note that in the future this could be made an Asynchronous Recieve
    MPI_Recv(pivots, procs-1, MPI_FLOAT, ROOT, procId, MPI_COMM_WORLD, &stat);

#ifdef DBG_1
    printArr("Pivots:", pivots, procs-1, procId);
    textcolor(WHITE);
#endif    
  }
  
  //By Now everyone has a copy of the pivots
  //-1 Sort then get indices and send 
  //First we need an all to all communication to indicate each other how many
  //elements each will be receiving

  //NLOG(N) sorting time
  sort(data, data + localSize);
  //calculate a send count
  int* srcCount = (int*)malloc(sizeof(int)*procs);
  int* srcDispl = (int*)malloc(sizeof(int)*procs);

  srcDispl[0] = 0;
  int pivIndex = 0, displ = 0, count = 0; 
  for(int i=0; i < localSize; i++){
    if(pivIndex == 0){
      if(data[i] < pivots[pivIndex])
	count += 1;
      else{
	srcDispl[pivIndex] = displ;
	srcCount[pivIndex] = count;
	displ += count;
	count = 0;
	i -= 1; //repeat loop with new pivIndex
	pivIndex += 1;
      }
    }
    else if(pivIndex == procs-1){
      if(data[i] >= pivots[pivIndex])
	count += 1;
    }
    else if(pivots[pivIndex-1] <= data[i] && data[i] < pivots[pivIndex])
      count += 1;
    else{
      srcDispl[pivIndex] = displ;
      srcCount[pivIndex] = count;
      displ += count;
      count = 0;
      i -= 1; //repeat loop with new pivIndex
      pivIndex += 1;
    }
  }
  //last Pivot count and disp
  srcCount[pivIndex] = count;
  srcDispl[pivIndex] = displ;

  //If we did not reach the end of the pivots set the remaining counts
  //to 0 and the displacement to max displacement
  while(pivIndex != procs-1){
    pivIndex += 1;
    srcCount[pivIndex] = 0;
    srcDispl[pivIndex] = srcDispl[pivIndex-1];
  }

#ifdef DBG_2
  textcolor(RED+procId);
  printArr("Pivots:", pivots, procs-1, procId);
  printArr("Data:", data, localSize, procId);
  printArr("Count:",srcCount, procs, procId);
  printArr("Displ:",srcDispl, procs, procId);
  textcolor(WHITE);
#endif

  //exchange counts with all to all
  int* destCount = (int*)malloc(sizeof(int)*procs);
  int* destDispl = (int*)malloc(sizeof(int)*procs);
  MPI_Alltoall(srcCount, 1, MPI_INT, destCount, 1, MPI_INT, MPI_COMM_WORLD);

#ifdef DBG_2
  textcolor(RED+procId);
  printArr("DestCount:", destCount, procs, procId);
  textcolor(WHITE);
#endif  
  
  //Calculate Receive Displacement
  destDispl[0] = 0;
  for(int i=1; i < procs; i++){
    destDispl[i] = destDispl[i-1] + destCount[i-1];
  }
  
  localSize = destDispl[procs-1] + destCount[procs-1];

#ifdef DBG_3
  textcolor(RED+procId);
  printf("ProcId:%d Final_Local_Size:%d Fraction:%.3f%\n", procId, localSize,
	 float(localSize)/float(dataSize)*100.0f);
  textcolor(WHITE);
#endif

  sortedData = (float*)malloc(sizeof(float)*localSize);
  
  //Exchange bucket data
  MPI_Alltoallv(data, srcCount, srcDispl, MPI_INT,
		sortedData, destCount, destDispl, MPI_INT,
		MPI_COMM_WORLD);

  //Final Local Sort
  //NLOG(N)
  sort(sortedData, sortedData + localSize);
  return;
}

