# Open source Message Passing Interface<a name = "TOP"></a>

1. [install](#install)

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










````
