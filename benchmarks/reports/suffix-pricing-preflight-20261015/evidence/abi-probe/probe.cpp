#include <omp.h>
#include <cstdint>
extern "C" void set_c() { omp_set_schedule(static_cast<omp_sched_t>(static_cast<uint32_t>(omp_sched_static)|0x80000000U),0); }
extern "C" int64_t get_c() { omp_sched_t kind; int chunk; omp_get_schedule(&kind,&chunk); return static_cast<uint32_t>(kind); }
