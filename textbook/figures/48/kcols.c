#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){return 1e-9*(double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);}
int main(void){ long ns[]={100,400,1600}; printf("n k trsv_per_col trsm (us)\n");
 for(int q=0;q<3;q++){long n=ns[q]; double*a=malloc(8*n*n),*b0=malloc(8*n*8),*b=malloc(8*n*8); srand(3);
  for(long i=0;i<n*n;i++)a[i]=rand()/(double)RAND_MAX; for(long i=0;i<n;i++)a[i+i*n]+=n; for(long i=0;i<8*n;i++)b0[i]=rand()/(double)RAND_MAX;
  for(long k=1;k<=6;k++){ double t1=1e30,t2=1e30;
   for(int r=0;r<100;r++){ memcpy(b,b0,8*n*k); double t=now(); for(long j=0;j<k;j++) cblas_dtrsv(CblasColMajor,CblasLower,CblasNoTrans,CblasNonUnit,n,a,n,b+j*n,1); t=now()-t; if(t<t1)t1=t;
     memcpy(b,b0,8*n*k); t=now(); cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,n,k,1.0,a,n,b,n); t=now()-t; if(t<t2)t2=t;}
   printf("%ld %ld %.2f %.2f\n",n,k,1e6*t1,1e6*t2);} }
}
