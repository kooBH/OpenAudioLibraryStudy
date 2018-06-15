# [Open source Message Passing Interface](../)<a name = "TOP"></a>

OpenMPI 는 분산형 시스템에서 사용되는 병렬연산 API 이다.  
 -> 통상적인 상황에서는 용도가 없다  

1. [install](#install)
2. [usage](#usage)

**FEAUTRES**  
+ Full MPI-3.1 standards conformance
+ Thread safety and concurrency
+ Dynamic process spawning
+ Network and process fault tolerance
+ Support network heterogeneity
+ Single library supports all networks
+ Run-time instrumentation
+ Many job schedulers supported 
+ Many OS's supported (32 and 64 bit)
+ Tunable by installers and end-users
+ Component-based design, documented APIs
+ Active, responsive mailing list
+ Open source license based on the BSD license 

### [INSTALL](#TOP)<a name = "install"></a>

[Latest Version](https://www.open-mpi.org/software/ompi/)  

```bash
$ tar -xzvf openmpi-x.x.x.tar.gz
$ cd openmpi-x.x.x
$ ./config

... # 좀 걸림

$ make

... # 좀 많이 걸림 10분 +  a

$ make install

$ export PATH="$PATH:/home/$USER/.openmpi/bin"    
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"  

$ mpirun
--------------------------------------------------------------------------
mpirun could not find anything to do.

It is possible that you forgot to specify how many processes to run
via the "-np" argument.
--------------------------------------------------------------------------

```


+ 환경변수 추가   
vim ~/.profile    
또는
vim ~/.bashrc  
끝 자락에   

export PATH="$PATH:/home/$USER/.openmpi/bin"    
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"  



### [USAGE](#TOP)<a name = "usage"></a>

```C++
#include <mpi.h>
#include <stdio.h>

int main()
{
	MPI_Init(NULL,NULL);

	int num_of_process;
	MPI_Comm_size(MPI_COMM_WORLD,&num_of_process);

	int rank_of_process;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank_of_process);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int len_name;
	MPI_Get_processor_name(processor_name,&len_name);

	printf("name : %s\nrank : %d\nnum : %d\n",processor_name,rank_of_process,num_of_process);


	MPI_Finalize();
	return 0;
}

```

```bash
$ mpicc hello_mpi.c
name : FRIST-FROM-ENTRANCE
rank : 0
num : 1
-> 단일 컴퓨터로 수행하였기 때문에 결과도 1개 뿐이다  

```









````
