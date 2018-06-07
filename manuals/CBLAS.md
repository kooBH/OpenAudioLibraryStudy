
# [CBLAS](../README.md)
1. [OpenBLAS](#OpenBLAS)
2. [MKL](#MKL)
3. [예시](#cblas_ex)

+ OpenBLAS<a name="OpenBLAS"></a>  
	* Linux   
	1. 설치
	```bash
	$ sudo apt-get install openblas-base 
	#/usr/lib/openblas-base/에 .a와 .so만 받는다 
	    
	$ git clone https://github.com/xianyi/OpenBLAS.git	
	#openblas project를 받는다  
	 ```
	 apt로 package를 받았을 경우 바로 사용하면된다  
	 git으로 받았을 경우에는  
	 make를 하면 CPU에 맞게 빌드해 준다  
	 또는 make TARGET=(CPU이름) 으로 지정해 줄 수도 있다
	   지원하는 CPU는 TargetList.txt에 있다  
	  	
	 2. 컴파일
	   + package를 받았을 경우   
	   -lopenblas  
	   만 해도 링크가 된다  
	   + 프로젝트를 받았을 경우
	    make 했을 때, libopenblas_CPU이름-r0.3.0.dev  .a 와 .so 가 생성된다  
	    -lopenblas_CPU이름-r0.3.0.dev 해주거나 라이브러리 파일의 이름을 바꿔줘서 옵션으로 받아주면 된다  
	    같은 이름의 라이브러리가 2개 나오기 때문에 -static 이나 -shared 로 명시를 해줘야 한다  
	    
	    또한 Thread를 포함하기 때문에  
	    -lpthread  
	    를 해주어야 한다
	    
	 3. 사용 
	   #include "cbals.h"  
	 
	 * Windows   
	   https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio  
	   
+ <a name="MKL">Intel MKL</a>
	* Linux    
	 1. 설치
	https://software.seek.intel.com/performance-libraries
	에서 Submit 하고 파일 받아서  
	Sudo tar -xzvf 파일명  
	하면 나오는 install.sh 를 실행  
	 또는 install_GUI.sh 를 써도 된다
	
	2. 환경 변수
	(설치폴더)/compilers_and_libraries_2018/linux/mkl/bin/mklvars.sh  
	는 환경 변수를 설정해주는 스크립트	
	$ source (mkvars경로)/mklvars.sh (arch) 
	로 적용  
	(arch) 는 32bit 면 ia32 64bit면 intel64  
	  스크립트로 export한 환경변수는 터미널이 닫히면 지속되지 않으므로  
	  ~/.bashrc(터미널을 열때마다 실행 )  이나  
	  ~/.profile(부팅 후 유저 로그인 시 실행)  에
	  source (mkvars경로)/mklvars.sh (arch) 를 추가해주면 된다  
	  
	3. 컴파일  
	[예제 파일](http://software.intel.com/sites/default/files/article/171460/mkl-lab-solution.c)       
	[컴파일 옵션 알아보기](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/)   
	컴파일 옵션 알아보는 사이트에서 자신의 조건에 맞는 컴파일 옵션을 찾는다  
		예 )
		+ link line  
		-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
		+ compile option  
		-DMKL_ILP64 -m64 -I${MKLROOT}/include
		+ 실제 명령  

		```bash 
		gcc  -DMKL_ILP64 -m64 -I${MKLROOT}/include  mkl-lab-solution.o  -Wl,--start-    group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/lib    mkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthr    ead -lm -ldl  -lm
		```  

		 옵션의 순서가 중요하다. 순서가 다르면 빌드 되지 않는다  
		[Guide](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2018-getting-started)
	4. 사용  
	  ```C++
	  #include "mkl.h"
	 ```  
	 * Windows 
	https://software.seek.intel.com/performance-libraries  
	에서 Submit 하고 파일 받아서 설치하면 VS에 통합까지 해준다  
 	  사용할 때는
	```C++
	  #include "mkl.h"  
	```    
 		 로 헤더를 불러오고
		 프로젝트 속성 -> 구성 속성 -> Intel Performance Libraries 에서 'Use Intel MKL' 을 설정해주면 된다  

+예시<a name="cblas_ex"></a>
```C++
#include "cblas.h"
#include <stdio.h>

int main(){
/*
 *  cblas_?gemm(layout,transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
 *
 *    layout :   i) --->CblasRowMajor
 *    			   [0][1]  =  {0,1,2,3}
 *                	   [2][3]
 *
 *             ii)  |  [0][2] = {0,1,2,3}
 *                  |  [1][3]
 *                 \_/ CblasColMajor
 *
 *   
 *   C := alpha * op(A)*op(B) + beta*C
 *
 *     op(X) =  	  i) X      when transX = CblasNoTrans
 *
 *     		 	 ii) X**T     ''        = CblasTrans
 *
 *     			iii) X**H     ''        = CblasConjTrans
 *
 *      m  = the number of rows of op(A)
 *      n  = the number of columns of op(B) and C 
 *      k  = the number of columns of op(A) and rows of op(B)
 *
 *
 *      lda : the first dimension of A
 *      ldb : 		''	     B
 *      ldc :		''	     C
 *
 * 		-the first dimension : the number of columns when CblasRowMajor
 *		                 	   	''    rows   when CblasColMajor
 *
 * */
/* ex1
 * */	int i,j;
	
	float a1[]={1,1,1,1};
	float b1[]={1,2,3,4,5,6};
	float c1[6];

	int m1 = 2;
	int k1 = 2;
	int n1 = 3; 

	int lda1=k1;
	int ldb1=n1;
	int ldc1=n1;

	int alpha1 = 1;
	int beta1 = 0;
	
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m1,n1,k1,alpha1,a1,k1,b1,n1,beta1,c1,n1);

	for(i=0;i< m1; i++)
	{
		for(j=0;j<n1; j++)
			printf("%4.2f ",c1[i*n1 + j]);
		printf("\n");
	}
	printf("\n");

/* ex2
 * ---->--->--->RowMajor
 *a2 | 0.1 0.4 |
 *   | 0.2 0.3 |  lda = 2 -> CblasTrans->  | 0.1 0.2 0.3 0.4 |  m = 2
 *   | 0.3 0.2 |                           | 0.4 0.3 0.2 0.1 |  k = 4      
 *   | 0.4 0.1 |
 *
 *
 *b2 | 10 |   k=4
 *   | 10 |   n=1
 *   | 10 |   ldb = 1
 *   | 10 |
 *
 *c2 | -110 |  m=2
 *   |   90 |  n=1
 *             ldc=1
 * */

	double a2[8]={0.1, 0.4, 0.2, 0.3, 0.3, 0.2, 0.4, 0.1};
	double b2[4]={10,10,10,10};
	double c2[2]={-100,100};
	
	int m2 = 2;
	int k2 = 4;
	int n2 = 1;

	int alpha2 = -1;
	int beta2 = 1;

	int lda2=m2;
	int ldb2=n2;
	int ldc2=n2;
cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m2,n2,k2,alpha2,a2,lda2,b2,ldb2,beta2,c2,ldc2);
	for(i=0;i< m2; i++)
	{
		for(j=0;j<n2; j++)
			printf("%4.2f ",c2[i*n2 + j]);
		printf("\n");
	}	printf("\n");
/* ex3 = ex4
  *
  *a3 | 1-1i  2-2i  3-3i |
  *
  *b3 |  1+1i   4+4i |
  *   |  2+2i   5+5i |
  *   |  3+3i   6+6i |
  *
  *c3 | 28+28i  64+64i| 
  *
  *
  * */

	typedef struct fx{float r;float i;}fx;

	fx a3[3]={1,-1,2,-2,3,-3};
	fx b3[6]={1,1,2,2,3,3,4,4,5,5,6,6};
	fx c3[2]={0,0,0,0};
	
	int m3 = 1;
	int k3 = 3;
	int n3 = 2;

	fx alpha3 ={1,1};
	fx beta3 = {0,0};

	int lda3=1;
	int ldb3=3;
	int ldc3=1;
/*ex4
 * a3 | 1-1i |  ->CblasTransA | 1-1i  2-2i  3-3i |
 *    | 2-2i |
 *    | 3-3i |
 *
 *b3 |1+1i 2+2i 3+3i|    ->CblasTransB |  1+1i   4+4i |
 *   |4+4i 5+5i 6+6i|                  |  2+2i   5+5i |
 *                                     |  3+3i   6+6i |
 *
 *c3 | 28+28i  64+64i| 
 * */
cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m3,n3,k3,&alpha3,a3,lda3,b3,ldb3,&beta3,c3,ldc3);

	for(i=0;i< m3; i++)
	{	for(j=0;j<n3; j++)
			printf("%4.2f %+4.2fi ",c3[i*n3 + j].r,c3[i*n3+j].i);
		printf("\n");
	}	printf("\n");
	
	double a4[6] = {1,-1,2,-2,3,-3};
	double b4[12] = {1,1,4,4,2,2,5,5,3,3,6,6};
	double c4[4] = {0,0,0,0};

	int m4 = 1;
	int k4 = 3;
	int n4 = 2;

	double alpha4[2] = {1,1};
	double beta4[2] = {0,0};

	int lda4 = 3;
	int ldb4 = 2;
	int ldc4 = 1;
cblas_zgemm(CblasColMajor,CblasTrans,CblasTrans,m4,n4,k4,&alpha4,a4,lda4,b4,ldb4,&beta4,c4,ldc4);
	for(i=0;i< m4; i++)
	{	for(j=0;j<(n4*2); j+=2)
			printf("%2.2lf %+2.2lfi ",c4[i*n4 + j],c4[i*n4 + j+1]);
		printf("\n");
	}	printf("\n");
	return 0;
}
