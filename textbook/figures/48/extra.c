#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
typedef __LAPACK_int lint;
static double now(void){return 1e-9*(double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);}
static unsigned long long s=1234567ULL;
static double rnd(void){s^=s<<13;s^=s>>7;s^=s<<17;return (double)(s>>11)*0x1p-53*2-1;}
/* triad bandwidth */
static void bw(void){ size_t n=1<<26; double*a=malloc(8*n),*b=malloc(8*n),*c=malloc(8*n); for(size_t i=0;i<n;i++){b[i]=1;c[i]=2;a[i]=0;}
  double best=1e30; for(int r=0;r<5;r++){double t=now(); for(size_t i=0;i<n;i++)a[i]=b[i]+3.0*c[i]; t=now()-t; if(t<best)best=t;}
  printf("triad GB/s %.1f (check %g)\n",3*8.0*n/best/1e9,a[n/2]); free(a);free(b);free(c);}
/* alignment: factor+solve same matrix at aligned vs +8 byte offset; compare bits */
static void align(void){ int diff_rf=0,diff_rs=0,diff_tv=0,tot=0;
  for(lint n=3;n<=41;n+=2) for(int rep=0;rep<10;rep++){ tot++;
    double*base1=aligned_alloc(64,(8*(n*n+8)+63)/64*64),*base2=aligned_alloc(64,(8*(n*n+8)+63)/64*64); double*a1=base1,*a2=base2+1; /* a2 misaligned by 8 */
    double*m=malloc(8*n*n),*b=malloc(8*n),*x1=aligned_alloc(64,(8*(n+8)+63)/64*64),*x2=aligned_alloc(64,(8*(n+8)+63)/64*64)+0; for(lint i=0;i<n*n;i++)m[i]=rnd(); for(lint i=0;i<n;i++)b[i]=rnd();
    memcpy(a1,m,8*n*n); memcpy(a2,m,8*n*n); lint ip1[64],ip2[64],info; dgetrf_(&n,&n,a1,&n,ip1,&info); dgetrf_(&n,&n,a2,&n,ip2,&info);
    if(memcmp(a1,a2,8*n*n)) diff_rf++;
    double*y2=x2+1; memcpy(x1,b,8*n); memcpy(y2,b,8*n); lint one=1; char N='N'; dgetrs_(&N,&n,&one,a1,&n,ip1,x1,&n,&info); dgetrs_(&N,&n,&one,a1,&n,ip1,y2,&n,&info);
    if(memcmp(x1,y2,8*n)) diff_rs++;
    memcpy(x1,b,8*n); memcpy(y2,b,8*n); cblas_dtrsv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,a1,n,x1,1); cblas_dtrsv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,a1,n,y2,1);
    if(memcmp(x1,y2,8*n)) diff_tv++;
    free(base1);free(base2);free(m);free(b);free(x1);free(x2);}
  printf("alignment: of %d matrices n=3..41 odd, bits differ: getrf %d, getrs(b misaligned) %d, trsv(b misaligned) %d\n",tot,diff_rf,diff_rs,diff_tv);}
/* gesv vs getrf+getrs bits and time */
static void gesv(void){ lint ns[]={2,3,4,8,64,256,1024}; for(int q=0;q<7;q++){lint n=ns[q],info,one=1; char N='N';
  double*m=malloc(8*n*n),*a=malloc(8*n*n),*b=malloc(8*n),*x1=malloc(8*n),*x2=malloc(8*n); lint*ip=malloc(8*n); int diff=0;
  double t1=1e30,t2=1e30;
  for(int r=0;r<(n<=256?20:5);r++){ for(lint i=0;i<n*n;i++)m[i]=rnd(); for(lint i=0;i<n;i++)b[i]=rnd();
    memcpy(a,m,8*n*n); memcpy(x1,b,8*n); double t=now(); dgesv_(&n,&one,a,&n,ip,x1,&n,&info); t=now()-t; if(t<t1)t1=t;
    memcpy(a,m,8*n*n); memcpy(x2,b,8*n); t=now(); dgetrf_(&n,&n,a,&n,ip,&info); dgetrs_(&N,&n,&one,a,&n,ip,x2,&n,&info); t=now()-t; if(t<t2)t2=t;
    if(memcmp(x1,x2,8*n)) diff++; }
  printf("n=%ld gesv %.1fus getrf+getrs %.1fus  bit-differ %d/%d\n",n,1e6*t1,1e6*t2,diff,(n<=256?20:5)); free(m);free(a);free(b);free(x1);free(x2);free(ip);}}
/* residual: getrf + trsv path */
static void resid(void){ lint ns[]={10,100,1000}; int reps[]={2000,300,20}; double eps=0x1p-52;
  for(int q=0;q<3;q++){lint n=ns[q],info; double*a=malloc(8*n*n),*f=malloc(8*n*n),*b=malloc(8*n),*x=malloc(8*n),*r=malloc(8*n); lint*ip=malloc(8*n);
    double*v=malloc(8*reps[q]);
    for(int k=0;k<reps[q];k++){ for(lint i=0;i<n*n;i++)a[i]=rnd()*1.7320508; for(lint i=0;i<n;i++)b[i]=rnd()*1.7320508; /* unit variance uniform */
      memcpy(f,a,8*n*n); dgetrf_(&n,&n,f,&n,ip,&info); memcpy(x,b,8*n);
      for(lint i=0;i<n;i++) if(ip[i]-1!=i){double t=x[i];x[i]=x[ip[i]-1];x[ip[i]-1]=t;}
      cblas_dtrsv(CblasColMajor,CblasLower,CblasNoTrans,CblasUnit,n,f,n,x,1); cblas_dtrsv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,f,n,x,1);
      /* residual in long double */
      double rn=0,an=0,xn=0,bn=0; for(lint i=0;i<n;i++){ long double s=-(long double)b[i]; double rs=0; for(lint j=0;j<n;j++){s+=(long double)a[i+j*n]*x[j]; rs+=fabs(a[i+j*n]);} if(fabsl(s)>rn)rn=fabsl(s); if(rs>an)an=rs; if(fabs(x[i])>xn)xn=fabs(x[i]); if(fabs(b[i])>bn)bn=fabs(b[i]);}
      v[k]=rn/(n*eps*(an*xn+bn)); }
    double mx=0; for(int k=0;k<reps[q];k++) if(v[k]>mx)mx=v[k];
    /* median */ for(int i=0;i<reps[q];i++)for(int j=i+1;j<reps[q];j++)if(v[j]<v[i]){double t=v[i];v[i]=v[j];v[j]=t;}
    printf("resid n=%ld reps=%d median %.4f max %.4f\n",n,reps[q],v[reps[q]/2],mx); free(a);free(f);free(b);free(x);free(r);free(ip);free(v);}}
int main(int c,char**v){ if(!strcmp(v[1],"bw"))bw(); if(!strcmp(v[1],"align"))align(); if(!strcmp(v[1],"gesv"))gesv(); if(!strcmp(v[1],"resid"))resid(); return 0;}
