#include "mkl.h"
#include <stdio.h>

int main()
{

/*
 *  cblas_?gemm(layout,transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
 *
 *    layout :   i) --->CblasRowMajor
 *    			   [0][1]  =  {0,1,2,3}
 *                 [2][3]
 *
 *             ii)  |  [0][2] = {0,1,2,3}
 *                  |  [1][3]
 *                 \_/ CblasColMajor
 *
 *   
 *   C := alpha * op(A)*op(B) + beta*C
 *
 *     op(X) =    i) X      when transX = CblasNoTrans
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
 *      ldb : 			''			 B
 *      ldc :			''			 C
 *
 * 		-the first dimension : the number of columns when CblasRowMajor
 *							   		''       rows    when CblasColMajor
 *
 * */


/* ex1
 * */
	int i,j;
	
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
 *	 | 0.2 0.3 |  lda = 2 -> CblasTrans->  | 0.1 0.2 0.3 0.4 |  m = 2
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
	}
	printf("\n");

/* ex3 = ex4
  *
  *a3 | 1-1i  2-2i  3-3i |
  *
  *b3 |  1+1i   4+4i |
  *	  |  2+2i   5+5i |
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
/*
 *
 * 
 *a3 | 1-1i |  ->CblasTransA | 1-1i  2-2i  3-3i |
 *   | 2-2i |
 *   | 3-3i |
 *
 *b3 |1+1i 2+2i 3+3i|    ->CblasTransB |  1+1i   4+4i |
 * 	 |4+4i 5+5i 6+6i|                  |  2+2i   5+5i |
 *                                     |  3+3i   6+6i |
 *
 *c3 | 28+28i  64+64i| 
 **
 *
 *
 * */
cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m3,n3,k3,&alpha3,a3,lda3,b3,ldb3,&beta3,c3,ldc3);

	for(i=0;i< m3; i++)
	{
		for(j=0;j<n3; j++)
			printf("%4.2f %+4.2fi ",c3[i*n3 + j].r,c3[i*n3+j].i);
		printf("\n");
	}
	printf("\n");
	
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
	{
		for(j=0;j<(n4*2); j+=2)
			printf("%2.2lf %+2.2lfi ",c4[i*n4 + j],c4[i*n4 + j+1]);
		printf("\n");
	}
	printf("\n");


	return 0;
}
