"""Fresh review-triggered transition diagnostics; original v2 files stay immutable."""
from __future__ import annotations

import engine as e  # fixes numerical threading before NumPy is imported
from transition_diagnostics import interval_diagnostic
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'transition'
RECIPE = OUT / 'diagnostic_recipe.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=e.encode, allow_nan=False)


def source_hashes():
    files = [Path(__file__), BASE / 'transition_diagnostics.py', BASE / 'engine.py',
             BASE / 'protocol.json', BASE / 'protocol_freeze.json', BASE / 'representatives.json',
             ROOT / '代码/模型物理.py', ROOT / '代码/数值求解.py']
    return {e.relative(p): e.digest(p) for p in files}


def original_records():
    rows = []
    for rep in read(BASE / 'representatives.json')['representatives']:
        for job_id in rep['validation_jobs']:
            path = BASE / 'runs/validation' / (job_id + '.json')
            record = read(path)
            rows.append(dict(source_json=e.relative(path), source_json_sha256=e.digest(path),
                source_npz=record['states_archive'], source_npz_sha256=record['states_archive_sha256'],
                original_job_id=job_id, diagnostic_job_id='transition_' + job_id,
                family=rep['family'], reserve_C=rep['reserve_C'], refine=record['refine'],
                job=record['job']))
    if len(rows) != 12:
        raise ValueError('Expected six representatives at two meshes')
    return sorted(rows, key=lambda d: (d['reserve_C'] != 1, d['family'], d['reserve_C'], d['refine']))


def freeze():
    record = dict(frozen_utc=e.utc_now(), study_id=e.PROTOCOL['study_id'],
        original_protocol_sha256=e.FROZEN_PROTOCOL_SHA256,
        scope='Review-triggered supplementary analysis, not part of the original preregistered main analysis.',
        trigger='Original validation archives contain dense temperature polynomials only around 120–180 s; exact transition extrema cannot be reconstructed from those archives.',
        immutable_main_analysis='No edits to original protocol, engine, original runs, selected controls or feasibility labels.',
        experiment='Twelve new integrations: six original representatives at r2 and r4; no control retuning.',
        change='Only temperature-polynomial capture expands to [original cutoff,180 s]; full-state dense storage remains disabled.',
        unchanged=['Original job physical inputs and fresh initial-state construction', 'Original current ramp and rectangular heater commands',
                   '180 s horizon, voltage floor, Jacobian, tolerances, breakpoints, sample times and wall limit'],
        diagnostics=dict(dense='Endpoint/derivative-root extrema and last upward crossing of minimum cell-mean temperature through 0.01 C over [cutoff,120].',
            separate_late_window='Reproduce the unchanged [120,180] s minimum separately.',
            physical_state_samples='At archived solver output times only: local CCL minimum temperature, maximum pore ice saturation, minimum cell voltage, anode pressure diagnostic and node/time location. These are sampled extrema, not continuous dense extrema.',
            intervals=['[0,180]', '[cutoff,120]', '[120,180]', '[cutoff,180]']),
        numerical_agreement_tolerances=dict(auxiliary_energy_J=1e-6, electrical_output_J=1e-4, late_window_minimum_K=1e-7),
        reporting='All new runs and failures retained separately; no silent retries or substitution. No upward crossing is null with an explicit status.',
        source_hashes=source_hashes(), jobs=original_records())
    save_new(RECIPE, record)
    print(json.dumps(dict(recipe=e.relative(RECIPE), sha256=e.digest(RECIPE), jobs=len(record['jobs'])), ensure_ascii=False))


def validate_recipe():
    e.read_protocol()
    recipe = read(RECIPE)
    if recipe['source_hashes'] != source_hashes() or recipe['jobs'] != original_records():
        raise ValueError('Frozen diagnostic recipe/input sources changed')
    for job in recipe['jobs']:
        if e.digest(ROOT / job['source_npz']) != job['source_npz_sha256']:
            raise ValueError('Original archive changed')
    return recipe


def sampled_diagnostics(model, raw):
    """Physical variables at saved solver output times, never dense extrema."""
    times = np.asarray(raw['series']['time_s'], dtype=float)
    scaled = np.asarray(raw['states_scaled'])
    if scaled.shape != (len(times), model.size):
        raise ValueError('Unexpected sampled-state orientation')
    physical = scaled * model.scale[None, :]
    temperature = physical[:, model.sl['T']] - e.TM
    ccl = np.flatnonzero(np.asarray(model.mesh.kind) == 'CCL')
    local_ccl = temperature[:, ccl]
    ice = physical[:, model.sl['mi']] / (920. * model.eps0[:, 0])
    voltage = np.asarray(raw['series']['voltage_V'])
    pressure = np.empty((len(times), int(np.sum(model.an))))
    anode_nodes = model.pidx[model.an]
    for i, (t, y) in enumerate(zip(times, scaled)):
        d, z, _ = model.quantities(y, e.current_ramp(t))
        eps = model.eps0 - d['ml'] / 990. - d['mi'] / 920.
        pgas = z['c'] * e.solver.R * z['Tp']
        vapor = np.maximum(d['mv'], 0.) * e.solver.R * z['Tp'] / (e.solver.MW * np.maximum(eps, 1e-12))
        pressure[i] = ((pgas + vapor) / e.solver.PREF - 1.)[model.an, 0]
    series = dict(time_s=times, local_CCL_minimum_C=np.min(local_ccl, axis=1),
                  maximum_ice_saturation=np.max(ice, axis=1), minimum_cell_voltage_V=np.min(voltage, axis=1),
                  maximum_absolute_anode_pressure_relative_deviation=np.max(np.abs(pressure), axis=1))

    def window(interval):
        indices = np.flatnonzero((times >= interval[0] - 1e-8) & (times <= interval[1] + 1e-8))
        if not len(indices):
            return dict(status='NO_NATIVE_SAMPLES', requested_interval_s=interval)
        def locate(values, nodes=None, maximum=False):
            data = values[indices]
            ii, jj = np.unravel_index(np.argmax(data) if maximum else np.argmin(data), data.shape)
            row = int(indices[ii])
            result = dict(value=float(data[ii, jj]), time_s=float(times[row]))
            if nodes is not None:
                node = int(nodes[jj])
                result.update(thermal_node_zero_based=node, cell_one_based=int(model.mesh.cell_ids[node]) + 1,
                              layer=str(model.mesh.kind[node]), x_m=float(model.mesh.x[node]))
            else:
                result['cell_one_based'] = int(jj) + 1
            return result
        return dict(status='NATIVE_STATE_SAMPLED', requested_interval_s=list(map(float, interval)),
                    sampled_range_s=[float(times[indices[0]]), float(times[indices[-1]])], sample_count=len(indices),
                    minimum_local_CCL_temperature_C=locate(local_ccl, ccl),
                    maximum_ice_saturation=locate(ice, model.pidx, maximum=True),
                    minimum_cell_voltage_V=locate(voltage),
                    maximum_absolute_anode_pressure_relative_deviation=locate(np.abs(pressure), anode_nodes, maximum=True))
    return series, window


def worker(entry):
    started = time.perf_counter()
    recipe = validate_recipe()
    job = e.normalize_job(entry['job'])
    job_id = entry['diagnostic_job_id']
    directory = OUT / 'runs'
    directory.mkdir(parents=True, exist_ok=True)
    path, archive = directory / (job_id + '.json'), directory / (job_id + '.npz')
    with (directory / (job_id + '.started.json')).open('x', encoding='utf-8') as stream:
        json.dump(dict(started_utc=e.utc_now(), recipe_sha256=e.digest(RECIPE)), stream)
    result = dict(job_id=job_id, phase='transition_diagnostic', started_utc=e.utc_now(),
        original_job=job, family=entry['family'], reserve_C=entry['reserve_C'], refine=entry['refine'],
        original_protocol_sha256=e.FROZEN_PROTOCOL_SHA256, diagnostic_recipe_sha256=e.digest(RECIPE),
        sources=recipe['source_hashes'], source_record=entry, fresh_integration=True, reused_prior_result=False,
        original_main_analysis_unchanged=True)
    arrays = {}
    try:
        original = read(ROOT / entry['source_json'])
        model, initial_C, ambient_C, realization = e.make_model(job)
        initial = model.initial(initial_C)
        with np.load(ROOT / entry['source_npz'], allow_pickle=False) as old:
            initial_equal = np.array_equal(initial, old['initial_state_scaled'])
            original_final = old['final_state_scaled'].copy()
        if not initial_equal or realization != 1.:
            raise ValueError('Fresh initial condition or heater realization differs from original')
        command, dose, knots, delta = e.waveform(job['powers'], job['cutoff_s'], job['profile'])
        raw, final, segments, integration_recipe = e.integrate(model, 180., initial_C, ambient_C,
            e.current_ramp, command, breakpoints=knots, dense_interval=[job['cutoff_s'], 180.], full_dense=False)
        if integration_recipe != original['solver_kwargs']:
            raise ValueError('Solver settings differ from the original validation')
        e.add_accounting(raw, model, initial, final, e.current_ramp)
        complete = raw['status'] == 'COMPLETE' and abs(raw['time_s'] - 180.) <= 1e-8
        series, sampled_window = sampled_diagnostics(model, raw)
        intervals = dict(full=[0.,180.], transition=[job['cutoff_s'],120.], late_window=[120.,180.], postheater=[job['cutoff_s'],180.])
        sampled = {name:sampled_window(interval) for name, interval in intervals.items()}
        arrays = e.model_arrays(model)
        arrays.update(e.archive_stage(raw, final, initial, segments))
        times, cc, orders = (arrays[k] for k in ['dense_segment_times_s','dense_temperature_coefficients_C','dense_orders'])
        dense = {name:interval_diagnostic(times, cc, orders, interval) for name, interval in intervals.items() if name != 'full'}
        tolerances = recipe['numerical_agreement_tolerances']
        differences = dict(auxiliary_energy_J=abs(raw['aux_energy_J'] - original['aux_energy_J']),
                           electrical_output_J=abs(raw['electrical_output_J'] - original['electrical_output_J']),
                           late_window_minimum_K=abs(dense['late_window']['minimum_mea_mean_C'] - original['dense_dwell']['minimum_mea_mean_C']) if complete else None)
        agreement = complete and all(differences[k] is not None and differences[k] <= v for k,v in tolerances.items())
        result.update(raw)
        result.update(complete_180s=complete, audit=e.solver.validate_constraints(raw, require_success=False),
            diagnostic_status='COMPLETE_AND_MATCHED' if agreement else 'REQUIRES_REVIEW',
            dense_diagnostics=dense, sampled_physical_diagnostics=sampled, sampled_physical_series=series,
            agreement_with_original=dict(passed=bool(agreement), absolute_differences=differences,
                tolerances=tolerances, fresh_initial_state_bitwise_equal=initial_equal,
                final_scaled_state_max_abs_difference=float(np.max(np.abs(final-original_final)))),
            original_feasible=original['feasible'], cutoff_s=job['cutoff_s'], powers=job['powers'],
            solver_kwargs=integration_recipe, dense_capture_interval_s=[job['cutoff_s'],180.], full_state_dense_saved=False,
            versions=dict(python=e.sys.version, numpy=np.__version__, scipy=e.scipy.__version__, numba=e.numba.__version__))
    except Exception as exc:
        result.update(status='NUMERICAL_FAILURE', complete_180s=False, diagnostic_status='FAILED_RETAINED',
                      failure_reason=f'{type(exc).__name__}: {exc}', exception_traceback=traceback.format_exc())
    return e.finish(path, archive, result, arrays, started)


def summarize(recipe):
    rows, sources = [], {e.relative(RECIPE):e.digest(RECIPE)}
    for entry in recipe['jobs']:
        path = OUT / 'runs' / (entry['diagnostic_job_id'] + '.json')
        record = read(path)
        sources[e.relative(path)] = e.digest(path)
        if record.get('states_archive'):
            sources[record['states_archive']] = record['states_archive_sha256']
        t = record.get('dense_diagnostics', {}).get('transition', {})
        late = record.get('dense_diagnostics', {}).get('late_window', {})
        sampled = record.get('sampled_physical_diagnostics', {}).get('transition', {})
        rows.append(dict(job_id=record['job_id'], family=entry['family'], reserve_C=entry['reserve_C'], refine=entry['refine'],
            status=record['status'], diagnostic_status=record['diagnostic_status'],
            cutoff_s=entry['job']['cutoff_s'], transition_minimum_C=t.get('minimum_mea_mean_C'),
            transition_minimum_time_s=t.get('minimum_time_s'), transition_minimum_cell_one_based=t.get('minimum_cell_one_based'),
            last_upcrossing_0p01_C_s=t.get('last_upcrossing_0p01_C_s'), crossing_status=t.get('crossing_status'),
            late_window_minimum_C=late.get('minimum_mea_mean_C'), late_window_minimum_time_s=late.get('minimum_time_s'),
            sampled_transition_local_CCL_min_C=sampled.get('minimum_local_CCL_temperature_C',{}).get('value'),
            sampled_transition_max_ice_saturation=sampled.get('maximum_ice_saturation',{}).get('value'),
            sampled_transition_min_voltage_V=sampled.get('minimum_cell_voltage_V',{}).get('value'),
            sampled_transition_max_anode_pressure_relative_deviation=sampled.get('maximum_absolute_anode_pressure_relative_deviation',{}).get('value'),
            agreement=record.get('agreement_with_original'), source_json=e.relative(path), source_json_sha256=e.digest(path)))
    summary = dict(generated_utc=e.utc_now(), original_protocol_sha256=e.FROZEN_PROTOCOL_SHA256,
                   diagnostic_recipe_sha256=e.digest(RECIPE), planned_supplementary_integrations=12,
                   original_main_stack_case_records=482, original_main_records_unchanged=True,
                   complete_and_matched=sum(r['diagnostic_status']=='COMPLETE_AND_MATCHED' for r in rows),
                   interpretation='Late-window feasibility does not establish temperature retention throughout the earlier heater-off transition.',
                   dense_vs_sampled='MEA extrema and crossings use accepted temperature polynomials. Local CCL temperature, ice, voltage and pressure extrema use saved full-state samples only.',
                   sources=sources, rows=rows)
    save_new(OUT / 'summary.json', summary)
    with (OUT / 'summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in r.items()} for r in rows)
    text = ['# Review-triggered transition diagnostics / 审稿触发的过渡段诊断', '',
            'This supplementary set has 12 new integrations and is separate from the original 482 stack case records. '
            'Original physical controls, initial state construction, load, solver and late-window criterion are unchanged; only dense recording coverage is expanded.', '',
            f'Completed and matched to original validation within frozen diagnostic tolerances: {summary["complete_and_matched"]}/12.', '',
            '|Family|Requirement / °C|Mesh|Cutoff / s|Transition minimum / °C|Minimum time / s|Last upward 0.01 °C crossing / s|Late-window minimum / °C|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda v:'NA' if v is None else f'{v:.6f}'
        text.append('|'+ '|'.join([r['family'],str(r['reserve_C']),'r'+str(r['refine'])]+[fmt(r[k]) for k in ['cutoff_s','transition_minimum_C','transition_minimum_time_s','last_upcrossing_0p01_C_s','late_window_minimum_C']])+'|')
    text += ['', '温度极值与上穿时刻来自新记录的接受步温度多项式；这是数值插值极值，并非物理精确解。'
             'CCL局部温度、冰饱和度、电压与气体近似诊断来自原生全状态输出时刻采样，不能与连续极值混称。'
             '全部逐节点位置/时间及四种区间结果保存在各案例JSON；原可行性标签不改判。', '',
             '三级核对：事实来源为冻结诊断配方及新积分，原档案哈希保持；逻辑上分别列过渡区间和后期约束窗口；'
             '不将新增补充分析称为原预注册结果，不把采样量宣称为连续极值。']
    with (OUT / 'report.md').open('x', encoding='utf-8') as stream: stream.write('\n'.join(text)+'\n')
    print(json.dumps(dict(summary=e.relative(OUT/'summary.json'), complete_and_matched=summary['complete_and_matched'], rows=rows), ensure_ascii=False), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true'); action.add_argument('--run', action='store_true')
    action.add_argument('--summarize', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args=parser.parse_args()
    if not 1 <= args.workers <= 4: parser.error('Use 1–4 process workers')
    if args.freeze: freeze(); return
    recipe=validate_recipe()
    if args.summarize: summarize(recipe); return
    for job in recipe['jobs']:
        if list((OUT/'runs').glob(job['diagnostic_job_id']+'.*')):
            raise FileExistsError('Existing diagnostic ID; do not read or overwrite a cached run')
    save_new(OUT/'execution_started.json', dict(started_utc=e.utc_now(), recipe_sha256=e.digest(RECIPE), workers=args.workers, jobs=12))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,job):job for job in recipe['jobs']}
        for future in as_completed(futures):
            r=future.result()
            print(json.dumps(dict(job_id=r['job_id'],status=r['status'],diagnostic_status=r['diagnostic_status'],elapsed_s=r['job_elapsed_s']),ensure_ascii=False),flush=True)
    summarize(recipe)


if __name__=='__main__': main()
