#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){return 1e-9*(double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);}
int main(void){ long cases[][2]={{256,256},{256,264},{512,512},{512,520},{1024,1024},{1024,1032}};
 srand(1); for(int q=0;q<6;q++){long n=cases[q][0],ld=cases[q][1],info; double*a0=malloc(8*ld*n),*a=malloc(8*ld*n); long*ip=malloc(8*n);
  for(long i=0;i<ld*n;i++)a0[i]=rand()/(double)RAND_MAX-0.5; double best=1e30;
  for(int r=0;r<9;r++){memcpy(a,a0,8*ld*n);double t=now();dgetrf_(&n,&n,a,&ld,ip,&info);t=now()-t;if(t<best)best=t;}
  printf("%ld %ld %.1f\n",n,ld,2.0/3*n*n*n/best/1e9);}
}
