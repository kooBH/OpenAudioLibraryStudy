#include <stdio.h>


__global__ void add( int* a,int* b, int*c )
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	c[tid] = a[tid] + b[tid];

}

int main()
{
	const int thread_per_block = 32; //Max 512 per block
	const int number_of_block = 1; //Max 65,535 per grid
	
	int size = thread_per_block * number_of_block;
	
	
	int *host_a,*host_b,*host_c;

	host_a = (int*)malloc(sizeof(int)*size);
	host_b = (int*)malloc(sizeof(int) * size );
	host_c = (int*)malloc(sizeof(int) * size );
	
	for(int i=0;i<size;i++)
	{
		host_a[i]=i;
		host_b[i]=i;
		host_c[i]=0;
	}

	int *device_a;	
	int *device_b;	
	int *device_c;	


	cudaMalloc((void**)&device_a,size*sizeof(int));
	cudaMalloc((void**)&device_b,size*sizeof(int));
	cudaMalloc((void**)&device_c,size*sizeof(int));

	cudaMemcpy(device_a, host_a,size*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(device_b, host_b,size*sizeof(int), cudaMemcpyHostToDevice);	

	add<<<number_of_block,thread_per_block >>>(device_a,device_b,device_c);

	cudaMemcpy(host_c,device_c,size*sizeof(int),cudaMemcpyDeviceToHost);

//	for(int i=0;i<number_of_block;i++)		printf("ARRAY[%d] =  %d\n",i*thread_per_block,host_c[i*thread_per_block]);

	for(int i=0;i<size;i++)
		printf("ARRAY[%d] = %d\n",i,host_c[i]);
	
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	free(host_a);
	free(host_b);
	free(host_c);
	
		

	return 0;
}
