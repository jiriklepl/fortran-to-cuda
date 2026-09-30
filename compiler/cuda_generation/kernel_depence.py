from itertools import groupby

from compiler.context import DoLoopContext
from compiler.kernel_abstraction import Kernel


class KernelGroup:
    def __init__(self,
                 kernels,
                 shared_outer_loop_contexts: list[DoLoopContext],
                 preceding_group: 'KernelGroup' = None):
        if len(kernels) == 0:
            raise Exception("Cannot create a KernelGroup with no kernels")
        self.kernels = kernels
        self.shared_outer_loop_contexts = shared_outer_loop_contexts
        self.preceding_group = preceding_group

    def get_shared_outer_loop_contexts(self) -> list[DoLoopContext]:
        return self.shared_outer_loop_contexts

    def set_preceding_group(self, preceding_group: 'KernelGroup'):
        self.preceding_group = preceding_group

    def get_all_preceding_kernels(self) -> list[Kernel]:
        if self.preceding_group is None:
            return []
        else:
            return self.preceding_group.kernels + self.preceding_group.get_all_preceding_kernels()


class DependenceResolver:
    def group_kernels(self, kernels: list[Kernel]) -> list[KernelGroup]:
        loop_contexts_from_outer_to_inner = [
            kernel_context.get_all_do_loop_contexts_from_outer_to_inner()
            for kernel_context in kernels
        ]

        no_iteration_kernels_indices = [ i for i, contexts in enumerate(loop_contexts_from_outer_to_inner) if len(contexts) == 0 ] + [None]

        if no_iteration_kernels_indices == [None]:
            return self._group_kernels_by_shared_do_loop_contexts(kernels, loop_contexts_from_outer_to_inner, [])
    
        all_groups = []

        for i in range(len(no_iteration_kernels_indices) - 1):
            start_kernel_index = no_iteration_kernels_indices[i]
            end_index = no_iteration_kernels_indices[i + 1]

            all_groups.append(KernelGroup(
                kernels=[kernels[start_kernel_index]],
                shared_outer_loop_contexts=[],
            ))

            sub_groups = self._group_kernels_by_shared_do_loop_contexts(
                kernels[start_kernel_index + 1:end_index],
                loop_contexts_from_outer_to_inner[start_kernel_index + 1:end_index],
                []
            )

            all_groups.extend(sub_groups)

        for i, group in enumerate(all_groups):
            if i == 0:
                group.set_preceding_group(None)
            else:
                group.set_preceding_group(all_groups[i - 1])

        return all_groups
        
    def _group_kernels_by_shared_do_loop_contexts(self, kernels, kernels_do_loop_contexts, shared_contexts_so_far = []):
        if any(len(contexts) == 0 for contexts in kernels_do_loop_contexts):
            return [KernelGroup(kernels, shared_contexts_so_far)]
         
        def get_first_do_loop_context(kernel_and_contexts):
            _, contexts_of_kernel = kernel_and_contexts
            return contexts_of_kernel[0] if contexts_of_kernel else None

        zipped_data = list(zip(kernels, kernels_do_loop_contexts))
        sub_groups = groupby(zipped_data, get_first_do_loop_context)
        
        result_groups = []

        for do_loop_context, group in sub_groups:
            group = list(group)
            kernels_in_group = [kernel for kernel, _ in group]
            new_contexts = [contexts[1:] for _, contexts in group]
            new_shared_contexts = shared_contexts_so_far + [do_loop_context] 

            sub_groups = self._group_kernels_by_shared_do_loop_contexts(kernels_in_group, new_contexts, new_shared_contexts)
            result_groups.extend(sub_groups)

        return result_groups
