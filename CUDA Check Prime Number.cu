#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>
using namespace std;

#define blocks_num 384
#define thread_per_block 1024

// function to check the num if prime can call from host and device
bool Is_Prime(unsigned long long int num)
{
	for (unsigned long long int i = 2; i <= sqrtf(num); ++i)
	{
		if (num % i == 0)
			return false;
	}
	return true;
}

// Create a kernel to check number if prime
__global__ void Check_Prime(unsigned long long int *d_number,bool *d_out,int *d_iteration)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int loop_begin = index * (*d_iteration);
	int loop_end = (index + 1)* (*d_iteration);
	if(index == 0)
		loop_begin +=2;
	for(long long i = loop_begin; i<= loop_end ; i++)
	{
		if (*d_number % i == 0)
		{
			*d_out = false;
			return;
		}
		else
		{
			*d_out = true;
		}
			
	}		
}

void main()
{
	float CPU_TIME;
	float GPU_TIME;

	cout << "\t\t\t*** CUDA TASK ***\n\t\t\t==================\n\n";
	cout << "Checking Numbers ... :\n----------------------\n";
	
	// host variables
	unsigned long long int number = 100000000000000003;
    bool *out;
	int iteration_per_thread = int(sqrtf(number)/(blocks_num * thread_per_block));

	// device var
	unsigned long long int *d_number;
	bool *d_out;
	int *d_iteration;

	// allocate device data
	cudaMalloc((void **)&d_number, sizeof(unsigned long long int));
	cudaMalloc((void **)&d_out, sizeof(bool));
	cudaMalloc((void **)&d_iteration, sizeof(int));

	// copy data from host to device 
	cudaMemcpy(d_number, &number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_iteration, &iteration_per_thread, sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;       // define  2 events
	cudaEventCreate(&start);       // create start event
	cudaEventCreate(&stop);		   // create stop event	
	cudaEventRecord(start, 0);     // begin start event

	// call check_prime kernal
	Check_Prime <<< blocks_num, thread_per_block >>> (d_number, d_out,d_iteration);

	cudaEventRecord(stop, 0);      // begin stop event   
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_TIME, start, stop);   // calculate execution time
	
	// destroy 2 events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// copy data back from device to host
	cudaMemcpy(out, d_out, sizeof(bool), cudaMemcpyDeviceToHost);
	
	// print GPU data
	cout << "the number = "<<number<<" is ";
	if(*out == true)
		cout << "prime\n";
	else 
		cout << "not prime\n";  
	cout << "GPU Time = " << GPU_TIME << endl << "--------------------\n\n";

	// sequential code  
	unsigned long long int cpu_start = clock();      // cpu start time 
	
	cout << "the number = "<<number<<" is ";
	if (Is_Prime(number))
		cout << "prime\n";
	else
		cout << "not prime\n";  

	unsigned long long int cpu_stop = clock();       // cpu stop time
	CPU_TIME = float(cpu_stop - cpu_start);          // cpu execution time
	cout << "CPU Time = " << CPU_TIME << endl << "--------------------\n";  // print CPU data

	// free allocated data on device
	cudaFree(d_number);
	cudaFree(d_out);	
}