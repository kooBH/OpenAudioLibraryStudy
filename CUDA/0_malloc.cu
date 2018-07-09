#include <stdio.h>

int main()
{
	int in[3] = {1,2,3};
	int out[3]  = {0,};

	int* Gmem;
	
	cudaMalloc((void**)&Gmem,3*sizeof(int));
	

	//int -> Gmem
	cudaMemcpy(Gmem, in, 3 * sizeof(int), cudaMemcpyHostToDevice );

	//Gmem -> out
	cudaMemcpy(out, Gmem, 3 * sizeof(int) ,cudaMemcpyDeviceToHost );

	for(int i=0;i<3;i++)
 		printf("%d\n",out[i]);

	cudaFree(Gmem);

	return 0;
}
