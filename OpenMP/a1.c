#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20000000
#define CHUNKSIZE 20000

int main()
{
	double start,end;
    double gap;	
	
	float *a,*b,*c ;

	long  nthreads,tid,i,chunk;

	a=(float*)malloc(sizeof(float)*N);
	b=(float*)malloc(sizeof(float)*N);
	c=(float*)malloc(sizeof(float)*N);

	for(i=0;i<N ; i++)
		a[i]=b[i] = i*1.0;
	chunk = CHUNKSIZE;
	
start=omp_get_wtime();
#pragma omp parallel shared(a,b,c,chunk) private(i,tid)
	{
		#pragma omp for schedule (dynamic,chunk) nowait
		for (i=0;i<N;i++)
		{
			c[i] = a[i] + b[i];
		}
	}

end=omp_get_wtime();
gap = end-start;
printf("gap : %lf\n",gap);

	
start=omp_get_wtime();
for (i=0;i<N;i++)
{	
	c[i] = a[i] + b[i];
}
		end=omp_get_wtime();
gap = end-start;
printf("gap : %lf\n",gap);

free(a);free(b);free(c);

return 0;
}


