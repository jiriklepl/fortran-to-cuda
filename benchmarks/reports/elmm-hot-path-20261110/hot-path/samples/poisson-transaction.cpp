extern "C" int elmm_gpu_pressure_solve_v1(fort_scope_t context,elmm_gpu_pressure_handle_v1 *h,fort_buffer_t phi,fort_buffer_t rhs,fort_scope_external_report_v1 *report) {
    report_reset(report);
    const int result=guarded([&]() -> int {
        require(h && report && phi && rhs && phi!=rhs,"missing/aliased pressure roots or provider");std::lock_guard lock(h->mutex);
        const int status=context_check(context,*h);if(status) return status;
        std::array<size_t,3> pl{2,2,2},pu{},rl{1,1,1},ru{};
        for(int i=0;i<3;++i) { pu[i]=size_t(h->config.solver.sizes[i])+2;ru[i]=size_t(h->config.solver.sizes[i])+1; }
        fort_scope_section interior{pl.data(),pu.data()},input{rl.data(),ru.data()};
        fort_scope_access write{0,0,nullptr,1,&interior,1,&interior},read{0,1,&input,0,nullptr,0,nullptr};
        fort_scope_external_binding_v1 bindings[2]={{phi,FORT_SCOPE_REAL64,3,8,write},{rhs,FORT_SCOPE_REAL64,3,8,read}};
        fort_scope_external_call_v1 call{1,0,bindings,2,h->lease,h->workspace};
        return fort_scope_external_execute_v1(context,&call,solve_worker,h,report);
    });
    if(report) report->status=result;return result;
}
