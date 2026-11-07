"""Read-only transition-window diagnostics from archived v2 temperature polynomials.

No integration, model import, sampled-state interpolation, or legacy data access.
Missing accepted-step coverage is reported as unavailable, never extrapolated.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import polynomial as poly

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PROTOCOL_HASH = '9c8d446971a1fa6085cbc74aa8565e97151b3c2266a2db7005be292176a042e5'
TIME_TOL_S = 1e-8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def roots(coefficients, lo, hi):
    cc = np.trim_zeros(np.asarray(coefficients, dtype=float), 'b')
    rr = poly.polyroots(cc) if len(cc) > 1 else []
    return [float(np.clip(z.real, lo, hi)) for z in rr
            if abs(z.imag) < 1e-7 and lo - 1e-10 <= z.real <= hi + 1e-10]


def interval_diagnostic(times, coefficients, orders, interval, threshold_C=.01):
    """Find extrema/threshold crossings of the stored accepted-step interpolants.

    Each coefficient vector is ascending-power in z=(t-t0)/(t1-t0).
    Full interval coverage is mandatory for every reported requested statistic.
    """
    begin, end = map(float, interval)
    minimum, when, cell = math.inf, None, None
    coverage, positive = [], []
    for (t0, t1), block, order in zip(times, coefficients, orders):
        t0, t1 = float(t0), float(t1)
        if t1 <= begin or t0 >= end:
            continue
        lo = max(0., (begin - t0) / (t1 - t0))
        hi = min(1., (end - t0) / (t1 - t0))
        if hi <= lo:
            continue
        cc = block[:int(order) + 1]
        coverage.append([float(t0 + lo * (t1 - t0)), float(t0 + hi * (t1 - t0))])
        cuts = [lo, hi]
        for k in range(5):
            candidates = [lo, hi] + roots(poly.polyder(cc[:, k]), lo, hi)
            values = poly.polyval(candidates, cc[:, k])
            index = int(np.argmin(values))
            if values[index] < minimum:
                minimum = float(values[index])
                when = float(t0 + candidates[index] * (t1 - t0))
                cell = k + 1
            shifted = cc[:, k].copy()
            shifted[0] -= threshold_C
            cuts.extend(roots(shifted, lo, hi))
        cuts = np.unique(np.clip(cuts, lo, hi))
        for left, right in zip(cuts[:-1], cuts[1:]):
            if np.min(poly.polyval((left + right) / 2., cc)) >= threshold_C:
                a, b = float(t0 + left * (t1 - t0)), float(t0 + right * (t1 - t0))
                if positive and abs(a - positive[-1][1]) <= TIME_TOL_S:
                    positive[-1][1] = b
                else:
                    positive.append([a, b])
    full = bool(coverage and abs(coverage[0][0] - begin) <= TIME_TOL_S
                and abs(coverage[-1][1] - end) <= TIME_TOL_S
                and all(abs(a[1] - b[0]) <= TIME_TOL_S for a, b in zip(coverage, coverage[1:])))
    if not full:
        return dict(status='UNAVAILABLE_DENSE_COVERAGE', interval_s=[begin, end],
                    full_dense_coverage=False, covered_segments=len(coverage),
                    covered_start_s=coverage[0][0] if coverage else None,
                    covered_end_s=coverage[-1][1] if coverage else None,
                    minimum_mea_mean_C=None, minimum_time_s=None, minimum_cell_one_based=None,
                    last_upcrossing_0p01_C_s=None, all_means_ge_threshold_intervals_s=None,
                    crossing_status='UNAVAILABLE_DENSE_COVERAGE')
    upcrossings = [a for a, _ in positive if a > begin + TIME_TOL_S]
    return dict(status='AVAILABLE', interval_s=[begin, end], full_dense_coverage=True,
                covered_segments=len(coverage), covered_start_s=coverage[0][0], covered_end_s=coverage[-1][1],
                minimum_mea_mean_C=minimum, minimum_time_s=when, minimum_cell_one_based=cell,
                last_upcrossing_0p01_C_s=max(upcrossings) if upcrossings else None,
                all_means_ge_threshold_intervals_s=positive,
                crossing_status='UPWARD_CROSSING_FOUND' if upcrossings else 'NO_UPWARD_CROSSING_IN_INTERVAL')


def self_test():
    # A synthetic algebra check only; never saved as an experimental trajectory.
    times = np.array([[0., 2.]])
    coefficients = np.zeros((1, 6, 5))
    coefficients[0, :3] = np.array([.24, -1., 1.])[:, None]
    orders = np.array([2])
    d = interval_diagnostic(times, coefficients, orders, [0., 2.])
    assert d['full_dense_coverage'] and abs(d['minimum_mea_mean_C'] + .01) < 1e-12
    assert abs(d['minimum_time_s'] - 1.) < 1e-12
    assert abs(d['last_upcrossing_0p01_C_s'] - (1. + 2. * math.sqrt(.02))) < 1e-12
    partial = interval_diagnostic(times, coefficients, orders, [-1., 2.])
    assert not partial['full_dense_coverage'] and partial['minimum_mea_mean_C'] is None
    assert partial['last_upcrossing_0p01_C_s'] is None
    print('Analytic extrema/crossing and missing-coverage checks PASS; no trajectory was integrated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    targets = [BASE / ('transition_diagnostics.' + suffix) for suffix in ['json', 'csv', 'md']]
    if any(p.exists() for p in targets):
        raise FileExistsError('Diagnostic outputs already exist; do not overwrite archived postprocessing.')
    if sha(BASE / 'protocol.json') != PROTOCOL_HASH:
        raise ValueError('Frozen protocol mismatch')
    sources = {relative(p): sha(p) for p in [Path(__file__), BASE / 'protocol.json',
               BASE / 'protocol_freeze.json', BASE / 'engine.py', BASE / 'representatives.json']}
    representative_data = read(BASE / 'representatives.json')
    if representative_data['protocol_sha256'] != PROTOCOL_HASH:
        raise ValueError('Representative selection protocol mismatch')
    rows = []
    for representative in representative_data['representatives']:
        for job_id in representative['validation_jobs']:
            path = BASE / 'runs/validation' / (job_id + '.json')
            record = read(path)
            if not (record['complete_180s'] and record['fresh_integration']
                    and not record['reused_prior_result'] and record['frozen_protocol_sha256'] == PROTOCOL_HASH):
                raise ValueError('Not a complete fresh frozen-protocol record: ' + job_id)
            archive = (ROOT / record['states_archive']).resolve()
            archive.relative_to(BASE / 'runs/validation')
            if sha(archive) != record['states_archive_sha256']:
                raise ValueError('Archive SHA256 mismatch: ' + job_id)
            for source, checksum in record['sources'].items():
                if sha(ROOT / source) != checksum:
                    raise ValueError('Original integration source mismatch: ' + source)
            sources[relative(path)] = sha(path)
            sources[relative(archive)] = sha(archive)
            with np.load(archive, allow_pickle=False) as arrays:
                times = arrays['dense_segment_times_s'].copy()
                coefficients = arrays['dense_temperature_coefficients_C'].copy()
                orders = arrays['dense_orders'].copy()
                reconstruction = arrays['dense_reconstruction_error_K'].copy()
            if times.shape != (len(orders), 2) or coefficients.shape != (len(orders), 6, 5):
                raise ValueError('Unexpected temperature polynomial schema')
            if not (np.isfinite(times).all() and np.isfinite(coefficients).all()
                    and np.all(times[:, 1] > times[:, 0]) and np.all(np.diff(times[:, 0]) >= 0)
                    and np.all((orders >= 1) & (orders <= 5))):
                raise ValueError('Invalid dense polynomial data')
            transition = interval_diagnostic(times, coefficients, orders, [record['cutoff_s'], 120.])
            late = interval_diagnostic(times, coefficients, orders, [120., 180.])
            if not late['full_dense_coverage']:
                raise ValueError('Archived late-window coverage unexpectedly incomplete')
            error = abs(late['minimum_mea_mean_C'] - record['dense_dwell']['minimum_mea_mean_C'])
            if error > 1e-10:
                raise ValueError('Independent late-window minimum differs from archived result')
            rows.append(dict(job_id=job_id, family=representative['family'], reserve_C=representative['reserve_C'],
                primary=representative['reserve_C'] == 1, refine=record['refine'], cutoff_s=record['cutoff_s'],
                source_json=relative(path), source_json_sha256=sources[relative(path)],
                source_npz=relative(archive), source_npz_sha256=sources[relative(archive)],
                dense_archive_start_s=float(times[0, 0]), dense_archive_end_s=float(times[-1, 1]),
                dense_segment_count=len(orders), polynomial_reconstruction_max_error_K=float(np.max(reconstruction)),
                transition=transition, late_window=late, archived_late_window_minimum_C=record['dense_dwell']['minimum_mea_mean_C'],
                late_window_minimum_reproduction_error_K=error))
    if len(rows) != 12 or sum(r['primary'] for r in rows) != 4:
        raise ValueError('Expected six representatives times two meshes, including four primary records')
    payload = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=PROTOCOL_HASH, postprocessing_scope='Post hoc diagnostic only; no new integrations or changes to the frozen feasibility criterion.',
        method=dict(temperature='Minimum over five thickness-weighted MEA means in degrees Celsius.',
            polynomial='Ascending coefficients in z=(t-t0)/(t1-t0); order-specific accepted BDF interpolants.',
            extrema='Endpoints and all real roots of each cell-polynomial derivative in the clipped interval.',
            crossing='Last below-to-at/above crossing of min_k Tbar_k=0.01 C in the fully covered requested interval; all-cell polynomial roots partition its sign intervals.',
            transition_interval='[heater cutoff,120 s]', prespecified_late_interval='[120 s,180 s]',
            missing_rule='No extrapolation and no coarse-sample fallback. Any coverage gap makes the requested transition minimum, time and crossing null.',
            meaning='Exact extrema of the saved numerical interpolation within floating-point root tolerance, not an exact continuous-physics solution.',
            current_archive_limitation='The original validation engine recorded dense temperature polynomials only for the late window, including any accepted step straddling 120 s.'),
        counts=dict(representative_mesh_records=len(rows), primary_records=4,
                    available_transition_records=sum(r['transition']['full_dense_coverage'] for r in rows),
                    late_window_records_verified=len(rows)),
        sources=sources, rows=rows,
        review=dict(facts='Original fresh JSON/NPZ and current model/solver/engine source hashes checked; late-window extrema independently reproduced.',
                    logic='The late operating window is not identified with the entire heater-off interval.',
                    unverified='No precise transition minimum or crossing is claimed without saved full interval coverage.'))
    flat = []
    for r in rows:
        flat.append(dict(job_id=r['job_id'], family=r['family'], reserve_C=r['reserve_C'], primary=r['primary'],
            refine=r['refine'], cutoff_s=r['cutoff_s'], dense_archive_start_s=r['dense_archive_start_s'],
            dense_archive_end_s=r['dense_archive_end_s'], transition_status=r['transition']['status'],
            transition_minimum_C=r['transition']['minimum_mea_mean_C'],
            transition_minimum_time_s=r['transition']['minimum_time_s'],
            transition_last_upcrossing_0p01_C_s=r['transition']['last_upcrossing_0p01_C_s'],
            late_window_minimum_C=r['late_window']['minimum_mea_mean_C'],
            late_window_minimum_time_s=r['late_window']['minimum_time_s'],
            late_window_reproduction_error_K=r['late_window_minimum_reproduction_error_K'],
            source_json=r['source_json'], source_json_sha256=r['source_json_sha256'],
            source_npz=r['source_npz'], source_npz_sha256=r['source_npz_sha256']))
    note = ['# 停热过渡段与预设后期窗口：档案诊断', '',
        '本后处理仅读取 v2 已保存的温度多项式，不重新积分，不修改原协议、可行性标签或结果。', '',
        f'共核对 {len(rows)} 个代表×网格记录；[120,180] s 的最低均温全部独立复现。',
        f'具备完整 [cutoff,120] s 稠密覆盖的记录为 {payload["counts"]["available_transition_records"]} 个。', '',
        '当前档案只含后期评估窗口的接受步多项式，通常从略早于120 s的跨界步开始。'
        '因此不能由这些档案精确确定停热后到120 s的最低均温、其时刻或最后上穿0.01 ℃的时刻；表内不可得项保持NA。'
        '没有使用0.5 s状态样本补插值，也没有将缺数据解释为温度违规或判据通过。', '',
        '原可行性结论仅约束120–180 s这一预先规定的后期运行窗口，并不保证停热后全过程都高于0 ℃。'
        '图中窗口之前的样本曲线可提示过渡回冷，但本档案诊断不为其给出未经保存的精确极值。', '',
        '|策略|要求/℃|网格|停热/s|稠密档案起点/s|过渡最低温|过渡最后上穿0.01℃|后期最低温/℃|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t = r['transition']
        fm = lambda v: 'NA' if v is None else f'{v:.9f}'
        note.append(f'|{r["family"]}|{r["reserve_C"]:g}|r{r["refine"]}|{r["cutoff_s"]:.9f}|{r["dense_archive_start_s"]:.9f}|{fm(t["minimum_mea_mean_C"])}|{fm(t["last_upcrossing_0p01_C_s"])}|{r["late_window"]["minimum_mea_mean_C"]:.9f}|')
    note += ['', '证据层级：仅为已保存数值插值的多项式诊断，不是物理精确解或新增实体测量。'
             '逐文件SHA256、生成时间、区间覆盖、求根方式及三级审核说明保存在同名JSON。', '',
             'English: The predeclared criterion applies only to the late operating window, 120–180 s. '
             'Saved validation dense polynomials do not cover the complete heater-off transition to 120 s. '
             'Transition extrema and last upward crossings are therefore unavailable; no extrapolation, '
             'sampled-state substitute or new integration has been used. All twelve late-window minima '
             'were independently reproduced from the saved polynomials.']
    with targets[0].open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with targets[1].open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with targets[2].open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(note) + '\n')
    print(json.dumps(dict(outputs=[relative(p) for p in targets], counts=payload['counts'],
        maximum_late_window_reproduction_error_K=max(r['late_window_minimum_reproduction_error_K'] for r in rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
