"""One checked plan supplies cost inputs without granting pricing authority."""

from dataclasses import replace

import pytest

from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.ir import CompilationError
from compiler.offload.exact_window import digest
from compiler.offload.interval_applicability import analyze_interval_applicability
from compiler.tests.test_checked_execution_plan import scanned
from compiler.tests.test_exact_window import observations
from compiler.tests.test_exact_window_placement import issued


def test_candidates_include_future_resources_and_stable_control_declarations(tmp_path):
    with scanned(tmp_path) as (_owner, plan):
        result = analyze_interval_applicability(plan)
        record = result.public()
        candidates = {row['resource']: row for row in record['key_candidates']}
        assert {'argument::a', 'argument::b', 'argument::out', 'argument::n', 'argument::escape'} <= candidates.keys()
        assert candidates['argument::out']['properties'] == [
            {'property': 'size', 'axis': 1}, {'property': 'lbound', 'axis': 1}]
        assert candidates['argument::n']['kind'] == 'integer_input'
        assert candidates['argument::escape']['kind'] == 'logical_input'
        assert all(not row['safe_evaluation_authorized'] for row in candidates.values())
        assert candidates['argument::n']['missing_proofs']
        assert record['original_control_trees']
        assert record['unclassified_dependencies']  # Internal iterators are not an entry key.
        assert not record['placement_authorized']
        assert not record['execution_authorized']
        assert record['plan_identity'] == plan.identity
        identity = record.pop('identity')
        assert identity == digest(record)


def test_independent_requirements_survive_first_unavailable_native_operation(tmp_path):
    with scanned(tmp_path) as (_owner, plan):
        before = plan.automatic_native_reason()
        record = analyze_interval_applicability(plan).public()
        kinds = {item['kind'] for item in record['requirements']}
        assert {'native_workload', 'runtime_workload_key', 'same_measured_alternative',
                'application_build_attestation', 'actual_execution_outcome',
                'complete_interval_observations'} <= kinds
        assert record['native_operations']
        assert record['numerical_operations']
        assert plan.automatic_native_reason() == before
        assert record['existing_exact_window'] is None
        record['requirements'].clear()
        record['placement_authorized'] = True
        assert not analyze_interval_applicability(plan).public()['placement_authorized']
        assert plan.exact_window_placement() is None


@pytest.mark.parametrize('mutation', ['source', 'configuration', 'facts'])
def test_revalidation_rejects_changed_source_context(tmp_path, mutation):
    with scanned(tmp_path) as (owner, plan):
        result = analyze_interval_applicability(plan)
        builder, path = owner.builder, owner.routine.scope.path
        source, config = path.read_text(), builder.config
        initialized = builder.facts['captures']['argument::a']['initialized']
        try:
            if mutation == 'source':
                path.write_text(source + '\n! changed source\n')
            elif mutation == 'configuration':
                builder.config = replace(config, host_threads=2)
            else:
                builder.facts['captures']['argument::a']['initialized'] = 'none'
            with pytest.raises(CompilationError):
                result.public()
        finally:
            path.write_text(source)
            builder.config = config
            builder.facts['captures']['argument::a']['initialized'] = initialized
        assert result.validate(plan) is result


def test_equal_portable_identity_and_public_record_are_not_checked_plan_authority(tmp_path):
    with scanned(tmp_path) as (_owner, first):
        result = analyze_interval_applicability(first)
    with scanned(tmp_path) as (_owner, second):
        assert first.execution_identity == second.execution_identity
        with pytest.raises(CompilationError, match='exact checked plan'):
            result.validate(second)
        public = second.public()
        with pytest.raises(CompilationError, match='checked execution plan'):
            analyze_interval_applicability(public)


def test_policy_difference_is_retained_not_called_semantic_equivalence(tmp_path):
    with scanned(tmp_path) as (_owner, sections):
        first = analyze_interval_applicability(sections).public()
    with scanned(tmp_path, config=OffloadConfig(policy='auto', scope_execution='reached')) as (_owner, auto):
        second = analyze_interval_applicability(auto).public()
    assert first['configuration_identity'] != second['configuration_identity']
    assert first['execution_identity'] != second['execution_identity']
    assert 'no measured/production equivalence inferred' in second['identity_policy']


def test_existing_issued_exact_window_decision_remains_authoritative(tmp_path):
    with issued(tmp_path) as (_owner, _plan, capture):
        evidence = observations(capture.public()['contract'])
    with issued(tmp_path, request={'evidence': evidence}) as (_owner, plan, placement):
        assert placement.public()['selection'] == 'resident'
        result = analyze_interval_applicability(plan).public()
        assert result['existing_exact_window'] == {key: placement.public()[key] for key in result['existing_exact_window']}
        assert result['existing_placement_unchanged']
        assert not result['placement_authorized']
        assert plan.exact_window_placement() is placement
        assert plan.automatic_native_reason() is None


def test_original_joined_teams_and_multidimensional_descriptors_are_preserved(tmp_path, monkeypatch):
    from compiler.tests import test_exact_window_placement as fixture

    source = fixture.fixture_source(original_teams=True)
    monkeypatch.setattr(fixture, 'fixture_source', lambda: source)
    with issued(tmp_path) as (_owner, plan, placement):
        result = analyze_interval_applicability(plan).public()
        arrays = [row for row in result['key_candidates'] if row['kind'] == 'array_descriptor']
        assert arrays
        assert all(len(row['properties']) == 6 for row in arrays)
        assert any(row['source_control_nodes'] for row in result['original_control_trees'])
        assert placement.public()['contract']['original_participation']['kind'] == 'original_joined_teams'
        assert plan.exact_window_placement() is placement


def test_native_graph_and_future_canonical_resources_feed_one_applicability_record(tmp_path):
    from compiler.tests.test_native_demand_owner import generate

    _path, builder, _outputs, _report = generate(tmp_path)
    plans = {id(demand._plan): demand._plan for demand in builder._resource_demands.values()
             if demand._plan is not None}
    plan, = plans.values()
    result = analyze_interval_applicability(plan).public()
    operation, = (row for row in result['native_operations'] if row['native_graph'] is not None)
    assert operation['native_graph_identity'] == digest(operation['native_graph'])
    assert {'demand_owner::star', 'demand_owner::out'} <= {
        row['resource'] for row in result['key_candidates']}
    assert 'native_graph_workload' in {row['kind'] for row in result['requirements']}
    assert not result['placement_authorized']
    assert operation['runtime_workload_key'] is None
