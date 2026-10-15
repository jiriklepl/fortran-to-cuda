"""Emit bounded actual process-input guards for empirical selection evidence."""

import json


def workload_source(alternatives):
    """Exact argv/environment/file contents, without trusting a launch marker.

    Input paths are explicit relative paths in the measured case directory.
    Files are opened as bounded regular files and checked again after hashing.
    Executable argv[0] is excluded only because the build attestation identifies
    the executed binary; every remaining argument is compared byte for byte.
    """
    clauses = []
    for alternative in alternatives:
        workload = alternative["workload"]
        predicates = ([] if alternative.get("key") is None else [
            "key&&" + "&&".join(f"key[{i}]=={value}LL" for i, value in enumerate(alternative["key"]))])
        predicates += ["args==std::vector<std::string>{" + ",".join(json.dumps(arg) for arg in workload["argv"][1:]) + "}"]
        predicates += ["selection_environment(" + json.dumps(name) + ","
                       + ("nullptr" if value is None else json.dumps(value)) + ")"
                       for name, value in sorted(workload["environment"].items())]
        predicates += ["selection_file_hash(" + json.dumps(path) + ")==" + json.dumps(value)
                       for path, value in sorted(workload["files"].items())]
        clauses.append("(" + "&&".join(predicates) + ")")
    return r'''
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
namespace {
struct selection_sha256 {
 std::array<uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
 std::array<unsigned char,64> block{};size_t used=0;uint64_t bytes=0;
 static uint32_t rotate(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
 void compress(){
  static constexpr uint32_t k[64]={
   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t w[64];for(unsigned i=0;i<16;++i)w[i]=(uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)
    |(uint32_t(block[4*i+2])<<8)|uint32_t(block[4*i+3]);
  for(unsigned i=16;i<64;++i){uint32_t a=w[i-15],b=w[i-2];
   w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));}
  uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
  for(unsigned i=0;i<64;++i){uint32_t t1=z+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
   uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
   z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
 }
 void add(const unsigned char *data,size_t size){bytes+=size;for(size_t i=0;i<size;++i){block[used++]=data[i];
   if(used==64){compress();used=0;}}}
 std::string finish(){uint64_t bits=bytes*8;block[used++]=0x80;
  if(used>56){while(used<64)block[used++]=0;compress();used=0;}
  while(used<56)block[used++]=0;for(unsigned i=0;i<8;++i)block[63-i]=static_cast<unsigned char>(bits>>(8*i));compress();
  static constexpr char digits[]="0123456789abcdef";std::string result;result.reserve(64);
  for(uint32_t value:h)for(int shift=28;shift>=0;shift-=4)result+=digits[(value>>shift)&15];return result;
 }
};
static std::string selection_file_hash(const char *path){
 int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return "";
 struct stat before{},after{};if(fstat(fd,&before)||!S_ISREG(before.st_mode)||before.st_size<0||before.st_size>64*1024*1024){close(fd);return "";}
 selection_sha256 digest;std::array<unsigned char,8192> data{};uint64_t total=0;ssize_t size;
 while((size=read(fd,data.data(),data.size()))>0){total+=uint64_t(size);if(total>64*1024*1024){close(fd);return "";}digest.add(data.data(),size);}
 bool valid=size==0&&!fstat(fd,&after)&&before.st_dev==after.st_dev&&before.st_ino==after.st_ino
  &&before.st_size==after.st_size&&uint64_t(after.st_size)==total
  &&before.st_mtim.tv_sec==after.st_mtim.tv_sec&&before.st_mtim.tv_nsec==after.st_mtim.tv_nsec
  &&before.st_ctim.tv_sec==after.st_ctim.tv_sec&&before.st_ctim.tv_nsec==after.st_ctim.tv_nsec;
 close(fd);return valid?digest.finish():"";
}
static bool selection_environment(const char *name,const char *expected){const char *value=std::getenv(name);
 return expected?(value&&std::strcmp(value,expected)==0):!value;}
static bool selection_arguments(std::vector<std::string> &args){
 int fd=open("/proc/self/cmdline",O_RDONLY|O_CLOEXEC);if(fd<0)return false;
 std::array<char,65537> data{};size_t used=0;ssize_t size;
 while(used<data.size()&&(size=read(fd,data.data()+used,data.size()-used))>0)used+=size;
 close(fd);if(!used||used==data.size()||data[used-1]!='\0')return false;
 size_t start=0;bool first=true;for(size_t i=0;i<used;++i)if(data[i]=='\0'){
  if(!first)args.emplace_back(data.data()+start,i-start);first=false;start=i+1;}return true;
}
static bool fort_selection_workload_matches(const int64_t *key){
 std::vector<std::string> args;if(!selection_arguments(args))return false;
 return @PREDICATE@;
}
}
'''.replace("@PREDICATE@", "||".join(clauses) or "false")
