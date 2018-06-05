#include <stdio.h>
#include <omp.h>

int main()
{

int tid,nthread;
//parallel threads with private variable 'tid'
#pragma omp parallel private(tid)
{
	tid = omp_get_thread_num();
	printf("hello from %d\n",tid);

	//only master thread 
	if(tid==0)
	{
		nthread = omp_get_num_threads();
	    printf("num of threads = %d\n",nthread);	
	}


//invisible barrier, only the master thread shall pass here
}

	return 0;
}
