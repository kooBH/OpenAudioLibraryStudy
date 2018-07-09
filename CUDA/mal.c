#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main()
{
	double *a;
	int s3 = N*N*N;
	double *b;
	a = (double*)malloc(sizeof(double)*s3*10);
	if(a == NULL)
		printf("NULL\n");

	b = (double*)malloc((long long)((long long)sizeof(double)*(long long)1000000000));
	if(b == NULL)
		printf("NULL\n");
	
	free(a);
	free(b);
	return 0;
}
