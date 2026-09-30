from pathlib import Path
import re

from compiler.cuda_generation.code_parts.cuda_kernel import CudaKernelGenerator
from compiler.cuda_generation.code_parts.cuda_mem import CudaMemCodeGenerator
from compiler.cuda_generation.code_parts.fortran_interface import FortranInterfaceGenerator
from compiler.cuda_generation.code_parts.host_params import ParamsGenerator
from compiler.cuda_generation.code_parts.pure_cpp_gen import PureCppGenerator
from compiler.cuda_generation.templates.template import Template
from compiler.kernel_abstraction import Kernel, KernelFunctionDefinition


class FullCodeGenerator:
    def __init__(self, kernels: list[Kernel], entry_kernel_func: KernelFunctionDefinition):

        path_to_templates = Path(__file__).resolve().parent / "templates"
        self.cu_file_template_path = path_to_templates / "kernels_interface_template.cu"
        self.fortran_interface_template_path = path_to_templates / "fortran_interface.f90"
        self.pure_cpp_template_path = path_to_templates / "cpp_impl.cpp"

        self.host_params_generator = ParamsGenerator(kernels, preceding_kernels=[])
        self.cuda_mem_code_generator = CudaMemCodeGenerator(kernels)
        self.kernel_code_generator = CudaKernelGenerator(kernels)

        self.fortran_generator = FortranInterfaceGenerator(kernels, entry_kernel_func)

        self.pure_cpp_generator = PureCppGenerator(kernels)

        self.entry_kernel_func = entry_kernel_func

    def generate_cuda_code(self) -> str:
        cu_file_template = Template(self.cu_file_template_path)

        cu_file_template.replace_placeholder("KERNEL_NAME", self.entry_kernel_func.name(), tabs=0)

        in_cpp_func_tabs = 2
        host_params = self.host_params_generator.generate_host_params()
        cu_file_template.replace_placeholder("HOST_PARAMETERS", host_params, tabs=in_cpp_func_tabs)

        pinning_of_host_buffers = self.cuda_mem_code_generator.generate_pinning_of_host_buffers()
        cu_file_template.replace_placeholder("PINNING_OF_HOST_BUFFERS", pinning_of_host_buffers, tabs=in_cpp_func_tabs)

        device_buff_decls, cuda_allocation = self.cuda_mem_code_generator.generate_cuda_alloc_code()
        cu_file_template.replace_placeholder("DEVICE_BUFF_DECLS", device_buff_decls, tabs=in_cpp_func_tabs)
        cu_file_template.replace_placeholder("MEMORY_ALLOCATIONS", cuda_allocation, tabs=in_cpp_func_tabs)

        cuda_H2D_copy = self.cuda_mem_code_generator.generate_cuda_host_to_device_copy_code()
        cu_file_template.replace_placeholder("CUDA_H2D_COPY", cuda_H2D_copy, tabs=in_cpp_func_tabs)

        total_h2d_size_calculation = self.cuda_mem_code_generator.generate_total_h2d_size_calculation()
        cu_file_template.replace_placeholder("TOTAL_H2D_SIZE_CALCULATION", total_h2d_size_calculation)

        cuda_D2H_copy = self.cuda_mem_code_generator.generate_cuda_device_to_host_copy_code()
        cu_file_template.replace_placeholder("CUDA_D2H_COPY", cuda_D2H_copy, tabs=in_cpp_func_tabs)

        total_d2h_size_calculation = self.cuda_mem_code_generator.generate_total_d2h_size_calculation()
        cu_file_template.replace_placeholder("TOTAL_D2H_SIZE_CALCULATION", total_d2h_size_calculation)

        cuda_variables_decls = self.kernel_code_generator.generate_host_local_var_decls()
        cu_file_template.replace_placeholder("LOCAL_VAR_DECLS_IN_HOST_CODE", cuda_variables_decls, tabs=in_cpp_func_tabs)

        kernel_calls = self.kernel_code_generator.generate_cuda_kernel_calls()
        cu_file_template.replace_placeholder("KERNELS_LAUNCH", kernel_calls, tabs=in_cpp_func_tabs)

        kernel_codes = self.kernel_code_generator.generate_cuda_kernels_code()
        cu_file_template.replace_placeholder("KERNEL_DEFINITIONS", kernel_codes, tabs=0)

        cuda_deallocation = self.cuda_mem_code_generator.generate_cuda_dealloc_code()
        cu_file_template.replace_placeholder("MEMORY_FREES", cuda_deallocation, tabs=in_cpp_func_tabs)

        return cu_file_template.code

    def generate_fortran_interface_code(self) -> str:
        fortran_interface_template = Template(self.fortran_interface_template_path)

        function_name = self.entry_kernel_func.name()
        module_name = self.entry_kernel_func.get_module_name()
        fortran_interface_template.replace_placeholder("MODULE_NAME", module_name, tabs=0)
        fortran_interface_template.replace_placeholder("KERNEL_NAME", function_name, tabs=0)

        fortran_interface_dummy = self.fortran_generator.generate_interface_dummies()
        fortran_interface_template.replace_placeholder("FORTRAN_INTERFACE_DUMMY", fortran_interface_dummy, tabs=3)

        fortran_interface_decls = self.fortran_generator.generate_interface_decls()
        fortran_interface_template.replace_placeholder("FORTRAN_INTERFACE_DECLS", fortran_interface_decls, tabs=2)

        fortran_cpp_kernel_args_call = self.fortran_generator.generate_cpp_kernel_call()
        fortran_interface_template.replace_placeholder("FORTRAN_CPP_KERNEL_ARGS_CALL", fortran_cpp_kernel_args_call, tabs=2)

        original_fortran_func_dummies = self.fortran_generator.generate_original_func_args_dummies()
        fortran_interface_template.replace_placeholder("ORIGINAL_FORTRAN_FUNC_DUMMY", original_fortran_func_dummies, tabs=2)

        original_fortran_func_args_decls = self.fortran_generator.generate_original_func_args_decls()
        fortran_interface_template.replace_placeholder("FORTRAN_KERNEL_ARGS_DECLS", original_fortran_func_args_decls, tabs=2)

        return fortran_interface_template.code

    def generate_pure_cpp_code(self) -> str:
        cpp_file_template = Template(self.pure_cpp_template_path)

        cpp_file_template.replace_placeholder("KERNEL_NAME", self.entry_kernel_func.name(), tabs=0)

        host_params = self.host_params_generator.generate_host_params()
        cpp_file_template.replace_placeholder("HOST_PARAMETERS", host_params, tabs=2)

        kernel_body = self.pure_cpp_generator.generate_kernel_body()
        cpp_file_template.replace_placeholder("KERNEL_BODY", kernel_body, tabs=2)

        local_var_decls = self.pure_cpp_generator.generate_local_var_decls()
        cpp_file_template.replace_placeholder("LOCAL_VAR_DECLS", local_var_decls, tabs=2)

        return cpp_file_template.code