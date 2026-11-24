#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
typedef __LAPACK_int lint;
static double now(void){return 1e-9*(double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);}
static unsigned long long s=88172645463325252ULL;
static double rnd(void){s^=s<<13;s^=s>>7;s^=s<<17;return (double)(s>>11)*0x1p-53*2-1;}
/* right-looking kij, column-major, partial pivoting, in place */
static int lu_unblocked(lint m, lint n, double*a, lint lda, lint*ip){
  lint kk=m<n?m:n;
  for(lint k=0;k<kk;k++){
    lint p=k; double mx=fabs(a[k+k*lda]);
    for(lint i=k+1;i<m;i++){double v=fabs(a[i+k*lda]); if(v>mx){mx=v;p=i;}}
    ip[k]=p;
    if(a[p+k*lda]==0) continue;
    if(p!=k) for(lint j=0;j<n;j++){double t=a[k+j*lda];a[k+j*lda]=a[p+j*lda];a[p+j*lda]=t;}
    double r=1.0/a[k+k*lda];
    for(lint i=k+1;i<m;i++) a[i+k*lda]*=r;
    for(lint j=k+1;j<n;j++){double ukj=a[k+j*lda]; double*cj=a+j*lda; const double*ck=a+k*lda;
      for(lint i=k+1;i<m;i++) cj[i]-=ck[i]*ukj;}
  }
  return 0;
}
/* blocked right-looking: own unblocked panel + trsm + gemm */
static void lu_blocked(lint n,double*a,lint lda,lint*ip,lint nb){
  for(lint j=0;j<n;j+=nb){
    lint jb=n-j<nb?n-j:nb;
    lu_unblocked(n-j,jb,a+j+j*lda,lda,ip+j);
    for(lint i=j;i<j+jb;i++){ ip[i]+=j; lint p=ip[i]; if(p!=i){
      for(lint c=0;c<j;c++){double t=a[i+c*lda];a[i+c*lda]=a[p+c*lda];a[p+c*lda]=t;}
      for(lint c=j+jb;c<n;c++){double t=a[i+c*lda];a[i+c*lda]=a[p+c*lda];a[p+c*lda]=t;}}}
    if(j+jb<n){
      cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,jb,n-j-jb,1.0,a+j+j*lda,lda,a+j+(j+jb)*lda,lda);
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n-j-jb,n-j-jb,jb,-1.0,a+j+jb+j*lda,lda,a+j+(j+jb)*lda,lda,1.0,a+j+jb+(j+jb)*lda,lda);
    }
  }
}
static void fill(double*a,lint n){for(lint i=0;i<n*n;i++)a[i]=rnd();}
static double time_one(int which,lint n,int reps){
  double*a0=malloc(8*n*n),*a=malloc(8*n*n); lint*ip=malloc(8*n); fill(a0,n);
  double best=1e30;
  for(int r=0;r<reps;r++){memcpy(a,a0,8*n*n); lint info=0,nn=n; double t=now();
    if(which==0) lu_unblocked(n,n,a,n,ip);
    else if(which==1) lu_blocked(n,a,n,ip,64);
    else if(which==2) dgetf2_(&nn,&nn,a,&nn,ip,&info);
    else dgetrf_(&nn,&nn,a,&nn,ip,&info);
    t=now()-t; if(t<best)best=t;}
  free(a0);free(a);free(ip); return best;
}
/* batch */
typedef struct{int which;lint n;int count;double*mats;double*work;lint*ip;} job;
static void* worker(void*v){job*j=v; lint n=j->n,info=0;
  for(int c=0;c<j->count;c++){memcpy(j->work,j->mats+(size_t)c*n*n,8*n*n);
    if(j->which==2) dgetf2_(&n,&n,j->work,&n,j->ip,&info); else dgetrf_(&n,&n,j->work,&n,j->ip,&info);}
  return 0;}
static double batch(int which,lint n,int total,int threads){
  double*mats=malloc(8*(size_t)n*n*total); for(size_t i=0;i<(size_t)n*n*total;i++)mats[i]=rnd();
  pthread_t th[16]; job jb[16]; double best=1e30;
  for(int r=0;r<5;r++){ double t=now();
    for(int k=0;k<threads;k++){int lo=total*k/threads,hi=total*(k+1)/threads;
      jb[k]=(job){which,n,hi-lo,mats+(size_t)lo*n*n,malloc(8*n*n),malloc(8*n)};
      pthread_create(&th[k],0,worker,&jb[k]);}
    for(int k=0;k<threads;k++){pthread_join(th[k],0);free(jb[k].work);free(jb[k].ip);}
    t=now()-t; if(t<best)best=t;}
  free(mats); return best/total;
}
int main(int argc,char**argv){
  if(argc>1&&!strcmp(argv[1],"flops")){
    lint ns[]={32,64,96,128,192,256,384,512,768,1024,1536,2048};
    printf("n unblocked blocked getf2 getrf (GFLOP/s)\n");
    for(int i=0;i<12;i++){lint n=ns[i]; double f=2.0/3*n*n*n; int reps=n<=256?20:(n<=1024?5:3);
      printf("%lld",n); for(int w=0;w<4;w++){ if(w==0&&n>1536){printf(" nan");continue;} printf(" %.3f",f/time_one(w,n,reps)/1e9);} printf("\n"); fflush(stdout);}
  } else if(argc>1&&!strcmp(argv[1],"batch")){
    printf("n getrf1 getf2_1 getrf10 getf2_10 (us per matrix)\n");
    for(lint n=8;n<=72;n+=4){ int tot=n<40?4000:2000;
      printf("%lld %.3f %.3f %.3f %.3f\n",n,1e6*batch(3,n,tot,1),1e6*batch(2,n,tot,1),1e6*batch(3,n,tot,10),1e6*batch(2,n,tot,10)); fflush(stdout);}
  } else if(argc>1&&!strcmp(argv[1],"trsv")){
    printf("n trsv trsm (us, one column)\n");
    lint ns[]={50,100,200,400,800,1600};
    for(int i=0;i<6;i++){lint n=ns[i]; double*a=malloc(8*n*n),*b=malloc(8*n),*b0=malloc(8*n); fill(a,n); for(lint k=0;k<n;k++){a[k+k*n]=n; b0[k]=rnd();}
      double t1=1e30,t2=1e30; for(int r=0;r<200;r++){memcpy(b,b0,8*n);double t=now();cblas_dtrsv(CblasColMajor,CblasLower,CblasNoTrans,CblasNonUnit,n,a,n,b,1);t=now()-t;if(t<t1)t1=t;
        memcpy(b,b0,8*n);t=now();cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,n,1,1.0,a,n,b,n);t=now()-t;if(t<t2)t2=t;}
      printf("%lld %.3f %.3f\n",n,1e6*t1,1e6*t2); free(a);free(b);free(b0);}
  }
  else if(argc>1&&!strcmp(argv[1],"inv")){
    printf("n getri getrs_I (us, after getrf; getrf time)\n");
    lint ns[]={8,32,64,128,256,512,1024,2048};
    for(int q=0;q<8;q++){lint n=ns[q],info=0; double*a0=malloc(8*n*n),*a=malloc(8*n*n),*x=malloc(8*n*n),*w=malloc(8*n*n*2); lint*ip=malloc(8*n); fill(a0,n); for(lint k=0;k<n;k++)a0[k+k*n]+=n;
      memcpy(a,a0,8*n*n); dgetrf_(&n,&n,a,&n,ip,&info);
      int reps=n<=256?50:5; double t1=1e30,t2=1e30,t0=1e30;
      for(int r=0;r<reps;r++){memcpy(x,a0,8*n*n);double t=now();dgetrf_(&n,&n,x,&n,ip,&info);t=now()-t;if(t<t0)t0=t;
        memcpy(x,a,8*n*n); lint lw=n*n; t=now(); dgetri_(&n,x,&n,ip,w,&lw,&info); t=now()-t; if(t<t1)t1=t;
        memset(x,0,8*n*n); for(lint k=0;k<n;k++)x[k+k*n]=1; char N='N'; t=now(); dgetrs_(&N,&n,&n,a,&n,ip,x,&n,&info); t=now()-t; if(t<t2)t2=t;}
      printf("%ld %.2f %.2f %.2f\n",n,1e6*t1,1e6*t2,1e6*t0); free(a0);free(a);free(x);free(w);free(ip);}
  } else if(argc>1&&!strcmp(argv[1],"lda")){
    printf("n lda blocked_gflops unblocked_gflops\n");
    lint cases[][2]={{512,512},{512,520},{1024,1024},{1024,1032},{256,256},{256,264}};
    for(int q=0;q<6;q++){lint n=cases[q][0],ld=cases[q][1]; double*a0=malloc(8*ld*n),*a=malloc(8*ld*n); lint*ip=malloc(8*n);
      for(lint i=0;i<ld*n;i++)a0[i]=rnd(); double f=2.0/3*n*n*n,tb=1e30,tu=1e30;
      for(int r=0;r<5;r++){memcpy(a,a0,8*ld*n);double t=now();lu_blocked(n,a,ld,ip,64);t=now()-t;if(t<tb)tb=t;
        memcpy(a,a0,8*ld*n);t=now();lu_unblocked(n,n,a,ld,ip);t=now()-t;if(t<tu)tu=t;}
      printf("%ld %ld %.2f %.2f\n",n,ld,f/tb/1e9,f/tu/1e9); free(a0);free(a);free(ip);}
  }
  return 0;
}
