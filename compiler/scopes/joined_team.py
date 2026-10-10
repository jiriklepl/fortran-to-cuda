"""Reached numerical workers inside one unchanged, synchronized original team.

Only mapped numerical DOs are replaced. Native units, uniform branches and
configured includes retain their lexical source and their original worksharing.
The original master owns runtime mutation; every member enters team workers.
"""

from dataclasses import replace
from copy import copy
import re

from fparser.two.utils import walk

from compiler.frontend.native_completion import _joined_completion_facts
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import (Branch, Native, Segment, StructuredScope,
                                     condition_fragment, directive, fortran_lines,
                                     fragment, original_roots, statement_span)


def prepared_span(node):
    spans = [item.item.span for item in walk(node) if getattr(item, 'item', None)]
    if not spans or any(span is None for span in spans):
        raise CompilationError('original team unit lacks prepared source order')
    return min(span[0] for span in spans), max(span[1] for span in spans)


def unit_span(unit):
    opening, loop, ending = unit
    # An original loop may own a prefix including PARALLEL or preceding
    # comments. Its associated DO directive is the exact replacement start.
    first = statement_span(opening)[0]
    last = statement_span(ending)[1] if ending is not None else statement_span(loop.content[-1])[1]
    return first, last


def declared_names(specification):
    """Generated local storage that DEFAULT(NONE) must explicitly share."""
    names = []
    for line in specification:
        if '::' not in line or re.search(r'\bparameter\b', line.split('::')[0], re.I):
            continue
        depth, start = 0, 0
        text = line.split('::', 1)[1]
        for index, character in enumerate(text + ','):
            depth += (character == '(') - (character == ')')
            if character == ',' and depth == 0:
                match = re.match(r'\s*([a-z][a-z0-9_]*)', text[start:index], re.I)
                if match:
                    names.append(match[1])
                start = index + 1
    return tuple(dict.fromkeys(names))


class JoinedTeam(StructuredScope):
    """A scope whose generated hooks surround original team source in place."""

    def __init__(self, builder, nodes):
        self.builder, self.nodes = builder, tuple(nodes)
        self.native_metadata = True
        self.calls, self.native, self.segments, self.guarded = [], [], [], {}
        self.operation_count, self.numerical_units, self.rejections = 0, [], []
        self.original_nodes = original_roots(builder.inline.original_selection(nodes))
        self.completion = builder.analysis.joined_completion(builder.entry.qualified, self.original_nodes)
        units = []
        _joined_completion_facts(builder.analysis, builder.entry.qualified, self.original_nodes, worksharing=units)
        if not units:
            raise CompilationError('mixed execution requires separate original PARALLEL and synchronized DOs')
        # Issue a native token first: it also rejects NOWAIT in an otherwise
        # unrelated sibling before any cut or generated artifact is admitted.
        builder.analysis.worksharing_native_completion(builder.entry.qualified, self.original_nodes, (units[0][1],))
        self.units_by_loop = {id(loop): unit for unit in units for loop in (unit[1],)}
        self.first = statement_span(nodes[0])[0]
        self.last = statement_span(nodes[-1])[1]
        self.tree = self._sequence(self.original_nodes)
        if not self.calls:
            raise CompilationError('original joined team contains no supported numerical worksharing unit')
        source = builder.analysis.structure(builder.entry.qualified)
        self.source_operation_count = source.operation_count
        self.reached_effect_count = max((len(operation.summary['operations']) for operation in self.native), default=0)
        # The authenticated local skeleton and each reached proof already have
        # their own budgets. Adding every independently proved segment's
        # expanded effects would recreate eager whole-owner closure rejection.
        if (len(self.calls) > 32 or len(self.calls) + len(self.native) > builder.analysis.operation_limit
                or self.reached_effect_count > builder.analysis.operation_limit):
            raise CompilationError('mixed original team exceeds bounded source-unit/reached-effect budget')
        self.extra_patches = []

    def _window(self, selected):
        """Bound a whole unmapped include run by original mapped neighbours.

        Never interpret a prepared line as an edit coordinate. The hash-bound
        monotone map must enclose exactly the selected unmapped execution.
        """
        units = [self.units_by_loop[id(loop)] for loop in selected]
        try:
            return unit_span(units[0])[0], unit_span(units[-1])[1]
        except CompilationError:
            record = self.builder.analysis.inputs.entries.get(str(self.builder.entry.scope.path))
            if record is None:
                raise
            first = units[0][0].item.span[0]
            last = (units[-1][2] or units[-1][1].content[-1]).item.span[1]
            mapping = record['line_map']
            before_index = next((index for index in range(first-2, -1, -1) if mapping[index] is not None), None)
            before = mapping[before_index] if before_index is not None else None
            after = next((value for value in mapping[last:] if value is not None), None)
            original_first = before + 1 if before is not None else None
            if before is not None:
                original_lines = self.builder.entry.scope.path.read_text().splitlines()
                prepared_lines = self.builder.analysis.inputs.path(self.builder.entry.scope.path).read_text().splitlines()
                # Configured preprocessing can retain a mapped blank marker
                # for INCLUDE followed by its unmapped expansion. Surround
                # that original directive itself, never a prepared coordinate.
                if (re.match(r'^\s*#\s*include\s+["<]', original_lines[before-1])
                        and not prepared_lines[before_index].strip()):
                    original_first = before
            if before is None or after is None or original_first >= after:
                raise CompilationError('configured native units have no bounded original source window')
            # All prepared text between those mapped neighbours must belong to
            # the selected units; a partial include cannot acquire whole authority.
            left = next(index for index in range(first-2, -1, -1) if mapping[index] is not None) + 2
            right = next(index for index in range(last, len(mapping)) if mapping[index] is not None)
            covered = {id(node) for loop in selected for node in walk(loop)}
            covered.update(id(item) for unit in units for item in (unit[0], unit[2]) if item is not None)
            for node in walk(self.builder.entry.execution):
                if getattr(node, 'item', None) is None or _kind(node) == 'Comment':
                    continue
                low, high = node.item.span
                if left <= low <= right and high <= right and id(node) not in covered:
                    raise CompilationError('configured native edit window includes unselected execution')
            return original_first, after - 1

    def _native(self, loops):
        proof = self.builder.analysis.worksharing_native_completion(
            self.builder.entry.qualified, self.original_nodes, tuple(loops))
        span = self._window(loops)
        operation = fragment(self.builder, tuple(loops), kind='original native worksharing',
                             native_metadata=True, completion=proof, span=span)
        if not operation.sections.available:
            from compiler.frontend.indirect_sections import analyze_indirect_sections
            from compiler.frontend.native_sections import NativeSections
            indirect = analyze_indirect_sections(self.builder.analysis, self.builder.entry.qualified,
                                                  tuple(loops), completion=proof, capture_locals=True)
            operation.indirect_sections = indirect
            if indirect.available:
                # Coverage is checked against the prepared exact union at the
                # original reached boundary. A failed inspector closes before
                # this native operation; it cannot establish a definition.
                operation.affine_sections = operation.sections
                operation.sections = NativeSections(True, resources=())
        # These units are never reproduced from prepared text or renamed ASTs.
        operation.preserve_original = True
        operation.completion_proof = proof
        self.native.append(operation)
        self.operation_count += len(operation.summary['operations'])
        return operation

    def _sequence(self, nodes):
        result, pending = [], []

        def flush():
            if pending:
                result.append(self._native(tuple(pending)))
                pending.clear()

        for node in nodes:
            if _kind(node) == 'Comment':
                continue
            if _kind(node) == 'If_Construct':
                flush()
                alternatives, body, header = [], [], None
                for item in node.content:
                    kind = _kind(item)
                    if kind in {'If_Then_Stmt', 'Else_If_Stmt', 'Else_Stmt', 'End_If_Stmt'}:
                        if header is not None:
                            condition = None if _kind(header) == 'Else_Stmt' else condition_fragment(self.builder, header)
                            if condition is not None and condition.effects:
                                raise CompilationError('original team branch requires payload publication')
                            if condition is not None:
                                self.native.append(condition)
                                self.operation_count += len(condition.summary['operations'])
                            alternatives.append((condition, self._sequence(body)))
                        header, body = item, []
                    else:
                        body.append(item)
                result.append(Branch(alternatives, statement_span(node)))
                continue
            unit = self.units_by_loop.get(id(node))
            if unit is None:
                raise CompilationError('mixed original team lost its associated worksharing source')
            try:
                span = unit_span(unit)
                proof = self.builder.analysis.worksharing_completion(
                    self.builder.entry.qualified, self.original_nodes, (node,))
                facade = self.builder.inline.prepare_worksharing(node, proof)
                call = self.builder.inline.call(facade)
            except CompilationError as error:
                self.rejections.append({'reason': str(error), 'prepared_span': prepared_span(node)})
                pending.append(node)
                continue
            flush()
            fallback = self._native((node,))
            self.guarded[id(call.node)] = fallback
            self.calls.append(call)
            self.numerical_units.append((call, proof, span))
            result.append(Segment([call]))
        flush()
        return result

    def public(self):
        result = super().public()
        result['joined_team'] = {'schema_version': 1, 'completion': self.completion.public(),
            'execution': 'original team; master runtime coordination; collective numerical workers',
            'numerical_units': [{'first_line': span[0], 'last_line': span[1], 'procedure': call.procedure,
                                 'completion': proof.public()} for call, proof, span in self.numerical_units],
            'native_rejections': self.rejections,
            'budgets': {'procedure_source_operations': self.source_operation_count,
                'source_operation_limit': self.builder.analysis.operation_limit,
                'planning_units': len(self.calls) + len(self.native),
                'planning_unit_limit': self.builder.analysis.operation_limit,
                'largest_reached_native_effects': self.reached_effect_count,
                'reached_effect_limit': self.builder.analysis.operation_limit,
                'expanded_owner_effects': self.operation_count,
                'numerical_unit_limit': 32,
                'effect_budget_role': 'each reached proof; expanded effects are not summed across ownership'},
            'native_abi': 'unchanged original body; absent borrowed controls are never read',
            'failure_policy': 'publish and continue once before execution; never replay a completed prefix'}
        for operation, public in zip(self.native, result['native_operations'], strict=True):
            if hasattr(operation, 'indirect_sections'):
                public['indirect_inspector'] = operation.indirect_sections.public()
                if operation.indirect_sections.available:
                    public['sections'] = operation.affine_sections.public()
                    public['effective_sections'] = {'availability': 'checked when reached',
                        'precision': 'exact runtime read/write unions from indirect_inspector.references',
                        'preflight': 'all inspection and definition validation precede numerical native work',
                        'failure': 'publish and close before original native work; no overwrite facts'}
        return result

    def shared_names(self, specification):
        return (*declared_names(specification),
                'fort_native_ready', 'fort_team_run', 'fort_team_native')

    def emit_original(self, owner, unit, handles, values, imports, specification):
        """Return additional specifications and exact original-source patches."""
        from compiler.scopes.access import build_native_accesses
        from compiler.scopes.source import _checked

        builder = self.builder
        imports.append('use omp_lib, only: fort_team_omp_level => omp_get_level, fort_team_omp_threads => omp_get_num_threads')
        # ENTRY-only dummies must not even appear in original native SHARED
        # clauses. Always-present local mirrors carry control inside the team.
        original_owner, original_handles = owner, handles
        owner = copy(owner)
        owner.context, owner.enabled, owner.control_guard = 'fort_team_context', 'fort_team_enabled', None
        handles = {root: 'fort_team_buffer_' + str(index) for index, root in enumerate(original_handles)}
        extra = ['logical :: fort_team_run, fort_team_native, fort_team_owned, fort_team_enabled',
                 'integer(c_int64_t) :: fort_team_context',
                 *[f'integer(c_int64_t) :: {handle}' for handle in handles.values()]]
        before = [f'fort_team_context = {original_owner.context}',
                  f'fort_team_enabled = {original_owner.enabled}',
                  *[f'{handles[root]} = {handle}' for root, handle in original_handles.items()],
                  'fort_team_owned = .true.']
        after = ['if (fort_team_owned) then', f'{original_owner.context} = fort_team_context',
                 f'{original_owner.enabled} = fort_team_enabled', 'endif']
        patches = []
        numeric = {id(self.guarded[id(call.node)]): call for call in self.calls}
        lowers = {root: tuple(f'lbound({values[root]}, {axis}, kind=c_int64_t)'
                              for axis in range(1, binding.rank + 1)) for root, binding in unit.arrays.items()}

        def coordinated(lines, *, reset=False):
            return ['!$omp master', *(['fort_team_run = .false.', 'fort_team_native = .false.'] if reset else []),
                    'if (fort_native_ready) then',
                    f'associate(fort_context => {owner.context})', *lines, 'end associate',
                    'endif', '!$omp end master', '!$omp barrier']

        def fail():
            return [*owner.close(), 'fort_native_ready = .false.']

        # Declaration storage is outside the team and shared. Section TARGETs
        # remain alive through host_end; preparation occurs only when reached.
        for number, operation in enumerate(self.native):
            if operation.kind == 'condition read':
                continue
            call = numeric.get(id(operation))
            block = 'fort_team_prepare_' + str(number)
            accesses = None
            try:
                parameters = {root: values[root] for root, binding in operation.bindings.items()
                              if root in values and not binding.rank and binding.dtype == 'integer'}
                indirect = getattr(operation, 'indirect_sections', None)
                if indirect is not None and indirect.available:
                    from compiler.scopes.indirect_access import build_indirect_accesses
                    accesses = build_indirect_accesses(indirect, handles, 'fort_t' + str(number),
                        resource_names={binding.root: builder.visible(builder.entry, binding.root)
                                        for binding in (*indirect.metadata, *indirect.scalars) if binding.rank},
                        parameters={**parameters, **{binding.root: builder.visible(builder.entry, binding.root)
                                                    for binding in indirect.scalars if not binding.rank}},
                        logical_lower_bounds=lowers, on_error=('exit ' + block,))
                else:
                    accesses = build_native_accesses(operation.sections, handles, 'fort_t' + str(number),
                        parameters=parameters, logical_lower_bounds=lowers, on_error=('exit ' + block,))
            except CompilationError:
                if getattr(operation, 'indirect_sections', None) is not None and operation.indirect_sections.available:
                    # Definition analysis relied on exact inspection. Emission
                    # cannot silently replace that obligation by whole access.
                    raise
            if accesses is not None:
                extra += [line for access in accesses for line in access.specification]
            prepare, begins, ends = [], [], []
            prepare += ['fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)', block + ': block']
            if accesses is not None:
                for index, access in enumerate(accesses, 1):
                    prepare += [*access.prepare, f'fort_bindings({index}) = fort_scope_plan_binding()',
                                f'fort_bindings({index})%buffer = {access.handle}',
                                f'fort_bindings({index})%access = {access.access_name}']
                    begins += _checked(f'fort_scope_host_begin(fort_context, {access.handle}, {access.access_name})')
                    ends += _checked(f'fort_scope_host_end(fort_context, {access.handle})')
                count = len(accesses)
            else:
                for index, (root, actions) in enumerate(sorted(operation.effects.items()), 1):
                    flags = (['FORT_SCOPE_READ_ALL'] if 'read' in actions else []) + (['FORT_SCOPE_WRITE_ALL'] if 'write' in actions else [])
                    if root in operation.overwrites:
                        flags.append('FORT_SCOPE_OVERWRITE_ALL')
                    prepare += [f'fort_bindings({index}) = fort_scope_plan_binding()',
                                f'fort_bindings({index})%buffer = {handles[root]}',
                                f'fort_bindings({index})%access%flags = ' + ' + '.join(flags)]
                    begins += _checked(f'fort_scope_host_begin(fort_context, {handles[root]}, fort_bindings({index})%access)')
                    ends += _checked(f'fort_scope_host_end(fort_context, {handles[root]})')
                count = len(operation.effects)
            prepare += ['if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_add( &',
                        'fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &',
                        f'{"c_loc(fort_bindings)" if count else "c_null_ptr"}, {count}_c_size_t, &',
                        '0.0_c_double, 0.0_c_double, 0_c_int)',
                        'if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)',
                        'end block ' + block, 'if (fort_status /= FORT_SCOPE_OK) then', *fail(),
                        'else', *begins, 'fort_team_native = .true.', 'endif']
            lines = []
            if call is not None:
                guards = (*call.region.runtime_guards,
                          *(("fort_scope_numerical_environment_supported() /= 0",)
                            if call.region.requires_numerical_environment else ()))
                lines += ['fort_numerical_guard = .true.']
                lines += ['if (fort_numerical_guard) fort_numerical_guard = ' + guard for guard in guards]
                lines += ['if (fort_numerical_guard) then',
                          'fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)',
                          *builder.inline.emit_call(call, handles, values, imports, query=True),
                          'if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)']
                if builder.config.policy == 'auto':
                    public, _ = builder.entry_artifacts(call.procedure)
                    imports.append(f"use {public['fortran_module']}, only: fort_team_choose_{number} => {public['planning']['fortran_selector']}")
                    lines += [f'if (fort_status == FORT_SCOPE_OK) fort_status = fort_team_choose_{number}(fort_context, fort_decision)']
                selected = 'fort_decision%gpu_units > 0' if builder.config.policy == 'auto' else '.true.'
                lines += ['if (fort_status /= FORT_SCOPE_OK) then', *fail(),
                          'else', 'fort_team_run = ' + selected, 'endif', 'endif',
                          'if (fort_native_ready .and. .not. fort_team_run) then', *prepare, 'endif']
            else:
                lines += prepare
            prefix = coordinated(lines, reset=True)
            # A collective worker may return while its numerical execution is
            # still pending in the context. Complete that execution before the
            # next reached query validates definitions, without publishing any
            # unrelated device-current fields. Native work commits first.
            suffix = coordinated(['if (fort_team_native) then', *ends, 'endif',
                                  *_checked('fort_scope_wait(fort_context)')])
            first, last = operation.span
            if call is not None:
                run = builder.inline.emit_call(call, handles, values, imports, query=False, collective=True,
                                               status='fort_returned', check_status=False)
                original = ''.join(owner.lines[first-1:last]).splitlines()
                replacement = [*prefix, 'if (fort_team_run) then', 'block', 'integer(c_int) :: fort_returned',
                               f'associate(fort_context => {owner.context})', *run,
                               '!$omp master', 'fort_status = fort_returned',
                               "if (fort_status /= FORT_SCOPE_OK) error stop 'original team numerical execution failed'",
                               '!$omp end master', '!$omp barrier', 'end associate', 'end block',
                               'else', *original, 'endif', '!$omp barrier', *suffix]
                patches.append((first, last, '\n'.join(fortran_lines(replacement)) + '\n'))
            else:
                patches += [(first, first-1, '\n'.join(fortran_lines(prefix)) + '\n'),
                            (last+1, last, '\n'.join(fortran_lines(['!$omp barrier', *suffix])) + '\n')]

        opening = next(item for node in self.original_nodes for item in walk(node)
                       if (directive(item) or '').startswith('parallel'))
        first, last = statement_span(opening)
        shared = self.shared_names([*specification, *extra])
        parallel = str(opening) + ' shared(' + ','.join(dict.fromkeys(shared)) + ')'
        qualification = coordinated([f'if (fort_team_omp_level() /= 1 .or. fort_team_omp_threads() /= {builder.config.host_threads}) then',
                                     *fail(), 'endif'])
        patches.append((first, last, '\n'.join(fortran_lines([parallel, *qualification])) + '\n'))
        return extra, patches, before, after
