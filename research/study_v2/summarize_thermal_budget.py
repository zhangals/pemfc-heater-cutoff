"""Summarize six completed fresh-v2 thermal budgets without reintegration."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED = [f'{p}_r{r}' for p in ('preheat', 'concurrent') for r in (1, 2, 4)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory):
    directory = Path(directory).resolve()
    directory.relative_to(HERE)
    records = []
    source_scripts = set()
    for job_id in EXPECTED:
        path = directory / f'{job_id}.thermal_budget.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['job_id'] != job_id:
            raise ValueError(f'Unexpected job identity: {path}')
        for relative, checksum in data['sources'].items():
            source = (ROOT / relative).resolve()
            source.relative_to(HERE)
            if digest(source) != checksum:
                raise ValueError(f'Input archive hash changed: {relative}')
        source_scripts.add(data['postprocessor_sha256'])
        post = data['stages']['post']
        window = post['windows']['to_coldest_mean_minimum']
        cold = post['dense_minima']['global_minimum']
        cold_cell = window['cells'][cold['cell_one_based'] - 1]
        regions = {r['region']: r for r in window['regions']}
        all_windows = [w for stage in data['stages'].values() for w in stage['windows'].values()]
        baseline_path = HERE / 'runs' / 'baseline' / f'{job_id}.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        native = baseline['postload']['dense_temperature_diagnostics']['0.01']
        records.append(dict(
            job_id=job_id, policy=data['policy'], refine=data['refine'],
            budget_json=path.relative_to(ROOT).as_posix(), budget_sha256=digest(path),
            baseline_json=baseline_path.relative_to(ROOT).as_posix(), baseline_sha256=digest(baseline_path),
            first_threshold_time_s=baseline['time_s'],
            post_minimum_C=cold['temperature_C'], post_minimum_time_s=cold['time_s'],
            post_minimum_cell_one_based=cold['cell_one_based'],
            coldest_cell_temperature_change_K=cold_cell['mean_temperature_change_K'],
            coldest_cell_drop_K=-cold_cell['mean_temperature_change_K'],
            coldest_cell_contributions_K=cold_cell['contributions_K'],
            storage_change_J={k:regions[k]['sensible_change_J'] for k in ('all','MEA','BP','EP')},
            mea_contributions_J=regions['MEA']['contributions_J'],
            phase_components_J=regions['MEA']['phase_components_J'],
            whole_stack_contributions_J=regions['all']['contributions_J'],
            signed_exchange_J=window['signed_exchange_J'],
            max_enthalpy_absolute_residual_J=max(abs(w['whole_stack_enthalpy']['residual_J']) for w in all_windows),
            max_enthalpy_relative_residual=max(w['whole_stack_enthalpy']['relative_residual'] for w in all_windows),
            max_regional_sensible_absolute_residual_J=max(abs(r['closure_residual_J']) for w in all_windows for r in w['regions']),
            max_regional_sensible_relative_residual=max(r['closure_relative'] for w in all_windows for r in w['regions']),
            max_cell_temperature_residual_K=max(abs(c['closure_residual_K']) for w in all_windows for c in w['cells']),
            max_pointwise_power_identity_error_W=max(s['pointwise_checks']['max_power_identity_error_W'] for s in data['stages'].values()),
            max_full_state_reconstruction_error_scaled=max(s['full_state_reconstruction_max_error_scaled'] for s in data['stages'].values()),
            independent_dense_minimum_difference_K=abs(cold['temperature_C']-native['minimum_mea_mean_C']),
            independent_dense_minimum_time_difference_s=abs(cold['time_s']-native['minimum_time_s']),
            quadrature_converged=data['quadrature_converged'],
            pointwise_identity_passed=data['pointwise_identity_passed'],
            closure_requires_review=data['closure_requires_review']))
    if len(source_scripts) != 1:
        raise ValueError('Mixed postprocessor versions; review before aggregation')
    source_script_sha256 = source_scripts.pop()
    if source_script_sha256 != digest(HERE / 'thermal_budget.py'):
        raise ValueError('Postprocessor changed since the budgets were generated')
    result = dict(study_id='sustained_heating_v2_20261106',
                  source_script_sha256=source_script_sha256,
                  summary_script_sha256=digest(__file__),
                  interval_definition='Storage and heat contributions: post-load start to the global minimum of the five fixed-cell MEA means. QA maxima: all pre/post integration windows.',
                  sign_convention='Signed contributions add heat; coldest_cell_drop_K is positive for cooling. all=MEA+BP+EP; do not add all to its partition.',
                  interpretation='Integrated terms share a coupled trajectory; their magnitudes are not independent intervention effects.',
                  all_qa_passed=all(r['quadrature_converged'] and r['pointwise_identity_passed'] and not r['closure_requires_review'] for r in records),
                  records=records)
    targets = [directory / 'summary.json', directory / 'summary.csv']
    if any(p.exists() for p in targets):
        raise FileExistsError('Summary outputs already exist; do not overwrite reviewed results')
    with targets[0].open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    rows = []
    for record in records:
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                flat.update({key + '_' + subkey:subvalue for subkey,subvalue in value.items()})
            else:
                flat[key] = value
        rows.append(flat)
    with targets[1].open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(outputs=[p.relative_to(ROOT).as_posix() for p in targets], all_qa_passed=result['all_qa_passed']), ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE / 'mechanism' / 'final')
    arguments = parser.parse_args()
    outcome = summarize(arguments.directory)
    if not outcome['all_qa_passed']:
        raise SystemExit(1)
