                                fort_graph_0_effect_fort_v0_u.write_count = 1; fort_graph_0_effect_fort_v0_u.writes = fort_graph_0_effect_fort_v0_u_writes;
                                fort_graph_0_effect_fort_v0_u.overwrite_count = 1; fort_graph_0_effect_fort_v0_u.overwrites = fort_graph_0_effect_fort_v0_u_writes;
                                FORT_SHARED_CHECK(fort_graph_0_access.add(fort_v0_u_handle, fort_graph_0_effect_fort_v0_u));
                                FORT_SHARED_CHECK(fort_graph_0_access.export_bindings(fort_graph_0_bindings, 1, &fort_graph_0_binding_count));
                                fort_graph_0_active = true;
                                auto &fort_graph_0_task = fort_graph_tasks[fort_graph_count++];
                                fort_graph_0_task.bindings = fort_graph_0_bindings;
                                fort_graph_0_task.binding_count = fort_graph_0_binding_count;
                                fort_graph_0_task.kernel.version = FORT_SCOPE_GRAPH_ABI_VERSION;
                                fort_graph_0_task.kernel.function = reinterpret_cast<const void *>(kernel_region_0_device);
                                const std::size_t fort_graph_0_blocks = (fort_internal_total - 1) / 256 + 1;
                                fort_graph_0_task.kernel.grid[0] = static_cast<unsigned int>(fort_graph_0_blocks > 65535 ? 65535 : fort_graph_0_blocks);
                                fort_graph_0_task.kernel.grid[1] = fort_graph_0_task.kernel.grid[2] = 1;
                                fort_graph_0_task.kernel.block[0] = 256;
                                fort_graph_0_task.kernel.block[1] = fort_graph_0_task.kernel.block[2] = 1;
                                fort_graph_0_task.kernel.arguments = fort_graph_0_parameters;
                                fort_graph_0_task.kernel.argument_count = 22;
                                fort_graph_0_values.value_1 = fort_v0_u_dim1;
                                fort_graph_0_values.value_2 = fort_v0_u_dim2;
                                fort_graph_0_values.value_3 = fort_v0_u_dim3;
                                fort_graph_0_values.value_4 = fort_v1_fort_region_lb_u_1;
                                fort_graph_0_values.value_5 = fort_v2_fort_region_lb_u_2;
                                fort_graph_0_values.value_6 = fort_v3_fort_region_lb_u_3;
                                fort_graph_0_values.value_7 = fort_v4_a;
                                fort_graph_0_values.value_8 = fort_v5_pr_gradient_x;
                                fort_graph_0_values.value_9 = fort_v6_unx;
                                fort_graph_0_values.value_10 = fort_v7_uny;
                                fort_graph_0_values.value_11 = fort_v8_unz;
                                fort_graph_0_values.value_12 = fort_internal_lower0;
                                fort_graph_0_values.value_13 = fort_internal_stride0;
                                fort_graph_0_values.value_14 = fort_internal_extent0;
                                fort_graph_0_values.value_15 = fort_internal_lower1;
                                fort_graph_0_values.value_16 = fort_internal_stride1;
                                fort_graph_0_values.value_17 = fort_internal_extent1;
                                fort_graph_0_values.value_18 = fort_internal_lower2;
                                fort_graph_0_values.value_19 = fort_internal_stride2;
                                fort_graph_0_values.value_20 = fort_internal_extent2;
                                fort_graph_0_values.value_21 = fort_internal_total;
                            }
                        }
                        fort_buffer_t fort_graph_handles[1]{};
                        void *fort_graph_pointers[1]{};
                        std::size_t fort_graph_reserve_count = 0;
                        std::size_t fort_graph_slot_0 = 0;
                        if (fort_graph_0_active) {
                            fort_graph_slot_0 = fort_graph_reserve_count;
                            fort_graph_handles[fort_graph_reserve_count++] = fort_v0_u_handle;
                        }
                        FORT_SHARED_CHECK(fort_scope_device_reserve_many_v1(fort_context, fort_graph_handles, fort_graph_reserve_count, fort_graph_pointers));
                        if (fort_graph_0_active) fort_v0_u_device = static_cast<double *>(fort_graph_pointers[fort_graph_slot_0]);
                        if (fort_graph_0_active) fort_graph_0_values.value_0 = fort_v0_u_device;
                        if (!fort_graph_count) return FORT_SCOPE_OK;
                        return fort_scope_graph_enqueue_v1(fort_context, 1757138722106318298ULL, fort_graph_tasks, fort_graph_0_argument_sizes);
                    }();
                    if (!fort_graph_status) {
                        fort_graph_done = true;
                        fort_task_queued = true;
                        fort_entry_work_started = true;
                    }
                    else if (fort_graph_status != FORT_SCOPE_RESOURCE && fort_graph_status != FORT_SCOPE_BOUNDARY)
                        return fort_graph_status;
                }
                if (!fort_graph_done) {
