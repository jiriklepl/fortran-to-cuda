void view_check(const fort_scope_external_view_v1 &v,const elmm_gpu_pressure_handle_v1 &h,bool phi) {
    require(v.type==FORT_SCOPE_REAL64 && v.rank==3 && v.element_bytes==sizeof(double),"pressure view requires rank3 REAL64");
    require(v.extents && v.strides && v.lower_bounds && v.device,"missing pressure view geometry/storage");
    size_t bytes=sizeof(double);const int ghosts=phi?2:1;
    const auto *lower=phi?h.config.phi_lower_bounds:h.config.rhs_lower_bounds;
    for(int i=0;i<3;++i) {
        require(v.extents[i]==size_t(h.config.solver.sizes[i])+2*ghosts && v.strides[i]==bytes && v.lower_bounds[i]==lower[i],"pressure view differs from checked dense original layout");
        require(bytes<=SIZE_MAX/v.extents[i],"pressure view byte overflow");bytes*=v.extents[i];
    }
    require(v.bytes==bytes,"pressure root bytes differ from full dense layout");
}
int solve_worker(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    return guarded([&]() -> int {
        auto &h=*static_cast<elmm_gpu_pressure_handle_v1 *>(opaque);
        require(count==2 && views,"pressure transaction requires Phi and RHS");
        view_check(views[0],h,true);view_check(views[1],h,false);
        const int32_t phi_ghosts[3]={2,2,2},rhs_ghosts[3]={1,1,1};
        const int status=poisson_solver_execute_device_v1(h.solver,views[0].device,views[0].bytes,
            views[1].device,views[1].bytes,phi_ghosts,rhs_ghosts,stream,started);
        return provider_status(status);
    });
}
