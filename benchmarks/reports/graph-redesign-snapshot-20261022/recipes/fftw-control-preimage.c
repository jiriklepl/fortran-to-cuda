
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef void *plan_t;
typedef double z64[2];
typedef float z32[2];
typedef struct { const char *prefix; const char *suffix; int initialized; int threads; int imported; } state_t;
static state_t d64={"fftw", "f64.fftw",0,1,0}, d32={"fftwf", "f32.fftw",0,1,0};
static pthread_mutex_t log_mutex=PTHREAD_MUTEX_INITIALIZER;
static FILE *evidence;
static void fail(const char *message) { fprintf(stderr,"FFTW_CONTROL_FATAL %s\n",message); fflush(stderr); _exit(91); }
static void *symbol(const char *name) { void *p=dlsym(RTLD_NEXT,name); if(!p)fail(name); return p; }
static void named(char *buffer,size_t count,state_t *s,const char *name) {
 if(snprintf(buffer,count,"%s_%s",s->prefix,name)<0)fail("symbol formatting failed");
}
static const char *mode(void) {
 const char *m=getenv("FORT_FFTW_CONTROL_MODE");
 if(!m || (strcmp(m,"capture") && strcmp(m,"verify")))fail("explicit capture or verify mode required");
 return m;
}
static FILE *log_stream(void) {
 if(!evidence) { const char *p=getenv("FORT_FFTW_CONTROL_LOG");if(!p)fail("fresh evidence log required");evidence=fopen(p,"wx");if(!evidence)fail("evidence log exists or cannot be created"); }
 return evidence;
}
static void json_string(FILE *f,const char *s) {
 fputc('"',f);for(const unsigned char *p=(const unsigned char*)s;*p;++p) {
  if(*p=='"'||*p=='\\') {fputc('\\',f);fputc(*p,f);}
  else if(*p=='\n')fputs("\\n",f);else if(*p=='\r')fputs("\\r",f);else if(*p=='\t')fputs("\\t",f);
  else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);
 }fputc('"',f);
}
static void wisdom_path(char *out,size_t count,state_t *s) {
 const char *p=getenv("FORT_FFTW_CONTROL_PREFIX");if(!p)fail("wisdom prefix required");
 int n=snprintf(out,count,"%s.%s",p,s->suffix);if(n<0||(size_t)n>=count)fail("wisdom path too long");
}
static unsigned flags_for(state_t *s,unsigned flags) {
 const char *m=mode();if(!s->initialized)fail("plan reached before original application thread initialization");
 if(!strcmp(m,"verify") && !s->imported) {
  char path[4096],name[96];wisdom_path(path,sizeof(path),s);named(name,sizeof(name),s,"import_wisdom_from_filename");
  int (*importer)(const char*)=symbol(name);if(!importer(path))fail("wisdom import failed");s->imported=1;
 }
 return !strcmp(m,"verify") ? flags|2097152U : flags;
}
static plan_t attest(state_t *s,const char *function,int rank,const int *dims,int sign,const int *kinds,unsigned original,unsigned applied,plan_t plan) {
 if(!plan)fail("plan creation failed; WISDOM_ONLY missing match cannot fall back");
 char name[96];named(name,sizeof(name),s,"sprint_plan");char *(*printer)(plan_t)=symbol(name);char *description=printer(plan);if(!description)fail("plan printer failed");
 pthread_mutex_lock(&log_mutex);FILE *f=log_stream();
 fprintf(f,"{\"schema_version\":1,\"event\":\"plan\",\"function\":");json_string(f,function);
 fprintf(f,",\"precision\":");json_string(f,s->prefix);fprintf(f,",\"original_thread_init_observed\":true,\"threads\":%d,\"original_flags\":%u,\"applied_flags\":%u,\"rank\":%d,\"dimensions\":[",s->threads,original,applied,rank);
 for(int i=0;i<rank;++i)fprintf(f,"%s%d",i?",":"",dims[i]);fprintf(f,"],\"sign\":%d,\"kinds\":[",sign);
 if(kinds)for(int i=0;i<rank;++i)fprintf(f,"%s%d",i?",":"",kinds[i]);fputs("],\"plan\":",f);json_string(f,description);fputs("}\n",f);fflush(f);pthread_mutex_unlock(&log_mutex);
 named(name,sizeof(name),s,"free");void (*release)(void*)=symbol(name);release(description);
 if(!strcmp(mode(),"capture")) {char path[4096];wisdom_path(path,sizeof(path),s);named(name,sizeof(name),s,"export_wisdom_to_filename");int(*exporter)(const char*)=symbol(name);if(!exporter(path))fail("wisdom export failed");}
 return plan;
}
plan_t fftw_plan_dft_1d(int n0, z64 *input, z64 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, z64 *input, z64 *output, int sign, unsigned flags)=symbol("fftw_plan_dft_1d"); int dimensions[]={ n0 };return attest(&d64,"fftw_plan_dft_1d",1,dimensions,sign,NULL,flags,applied,real(n0, input, output, sign, applied)); }
plan_t fftw_plan_r2r_1d(int n0, double *input, double *output, int k0, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, double *input, double *output, int k0, unsigned flags)=symbol("fftw_plan_r2r_1d"); int dimensions[]={ n0 };int kinds[]={ k0 };return attest(&d64,"fftw_plan_r2r_1d",1,dimensions,0,kinds,flags,applied,real(n0, input, output, k0, applied)); }
plan_t fftw_plan_dft_2d(int n0, int n1, z64 *input, z64 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, int n1, z64 *input, z64 *output, int sign, unsigned flags)=symbol("fftw_plan_dft_2d"); int dimensions[]={ n0, n1 };return attest(&d64,"fftw_plan_dft_2d",2,dimensions,sign,NULL,flags,applied,real(n0, n1, input, output, sign, applied)); }
plan_t fftw_plan_r2r_2d(int n0, int n1, double *input, double *output, int k0, int k1, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, int n1, double *input, double *output, int k0, int k1, unsigned flags)=symbol("fftw_plan_r2r_2d"); int dimensions[]={ n0, n1 };int kinds[]={ k0, k1 };return attest(&d64,"fftw_plan_r2r_2d",2,dimensions,0,kinds,flags,applied,real(n0, n1, input, output, k0, k1, applied)); }
plan_t fftw_plan_dft_3d(int n0, int n1, int n2, z64 *input, z64 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, int n1, int n2, z64 *input, z64 *output, int sign, unsigned flags)=symbol("fftw_plan_dft_3d"); int dimensions[]={ n0, n1, n2 };return attest(&d64,"fftw_plan_dft_3d",3,dimensions,sign,NULL,flags,applied,real(n0, n1, n2, input, output, sign, applied)); }
plan_t fftw_plan_r2r_3d(int n0, int n1, int n2, double *input, double *output, int k0, int k1, int k2, unsigned flags) { unsigned applied=flags_for(&d64,flags); plan_t (*real)(int n0, int n1, int n2, double *input, double *output, int k0, int k1, int k2, unsigned flags)=symbol("fftw_plan_r2r_3d"); int dimensions[]={ n0, n1, n2 };int kinds[]={ k0, k1, k2 };return attest(&d64,"fftw_plan_r2r_3d",3,dimensions,0,kinds,flags,applied,real(n0, n1, n2, input, output, k0, k1, k2, applied)); }
int fftw_init_threads(void) { int(*real)(void)=symbol("fftw_init_threads"); int result=real();if(result)d64.initialized=1; return result; }
void fftw_plan_with_nthreads(int n) { void(*real)(int)=symbol("fftw_plan_with_nthreads");real(n);d64.threads=n; }
plan_t fftwf_plan_dft_1d(int n0, z32 *input, z32 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, z32 *input, z32 *output, int sign, unsigned flags)=symbol("fftwf_plan_dft_1d"); int dimensions[]={ n0 };return attest(&d32,"fftwf_plan_dft_1d",1,dimensions,sign,NULL,flags,applied,real(n0, input, output, sign, applied)); }
plan_t fftwf_plan_r2r_1d(int n0, float *input, float *output, int k0, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, float *input, float *output, int k0, unsigned flags)=symbol("fftwf_plan_r2r_1d"); int dimensions[]={ n0 };int kinds[]={ k0 };return attest(&d32,"fftwf_plan_r2r_1d",1,dimensions,0,kinds,flags,applied,real(n0, input, output, k0, applied)); }
plan_t fftwf_plan_dft_2d(int n0, int n1, z32 *input, z32 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, int n1, z32 *input, z32 *output, int sign, unsigned flags)=symbol("fftwf_plan_dft_2d"); int dimensions[]={ n0, n1 };return attest(&d32,"fftwf_plan_dft_2d",2,dimensions,sign,NULL,flags,applied,real(n0, n1, input, output, sign, applied)); }
plan_t fftwf_plan_r2r_2d(int n0, int n1, float *input, float *output, int k0, int k1, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, int n1, float *input, float *output, int k0, int k1, unsigned flags)=symbol("fftwf_plan_r2r_2d"); int dimensions[]={ n0, n1 };int kinds[]={ k0, k1 };return attest(&d32,"fftwf_plan_r2r_2d",2,dimensions,0,kinds,flags,applied,real(n0, n1, input, output, k0, k1, applied)); }
plan_t fftwf_plan_dft_3d(int n0, int n1, int n2, z32 *input, z32 *output, int sign, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, int n1, int n2, z32 *input, z32 *output, int sign, unsigned flags)=symbol("fftwf_plan_dft_3d"); int dimensions[]={ n0, n1, n2 };return attest(&d32,"fftwf_plan_dft_3d",3,dimensions,sign,NULL,flags,applied,real(n0, n1, n2, input, output, sign, applied)); }
plan_t fftwf_plan_r2r_3d(int n0, int n1, int n2, float *input, float *output, int k0, int k1, int k2, unsigned flags) { unsigned applied=flags_for(&d32,flags); plan_t (*real)(int n0, int n1, int n2, float *input, float *output, int k0, int k1, int k2, unsigned flags)=symbol("fftwf_plan_r2r_3d"); int dimensions[]={ n0, n1, n2 };int kinds[]={ k0, k1, k2 };return attest(&d32,"fftwf_plan_r2r_3d",3,dimensions,0,kinds,flags,applied,real(n0, n1, n2, input, output, k0, k1, k2, applied)); }
int fftwf_init_threads(void) { int(*real)(void)=symbol("fftwf_init_threads"); int result=real();if(result)d32.initialized=1; return result; }
void fftwf_plan_with_nthreads(int n) { void(*real)(int)=symbol("fftwf_plan_with_nthreads");real(n);d32.threads=n; }
