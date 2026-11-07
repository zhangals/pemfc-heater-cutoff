"""Recompute virtual-probe, heat-path and mesh summaries from simulation results."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from numpy.polynomial import polynomial as poly

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/review_checks'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def plate_storage():
    nodes = read_csv(DATA / 'simulated_endplate_profiles.csv')
    flux = read_csv(DATA / 'simulated_boundary_powers.csv')
    summaries = []
    for mesh in (2, 4):
        exact = midpoint = two_depth = total_capacity = 0.
        for plate in (1, 2):
            subset = sorted((r for r in nodes if int(r['mesh']) == mesh and int(r['plate']) == plate), key=lambda r: float(r['x_over_L']))
            x = np.array([float(r['x_over_L']) for r in subset])
            capacities = np.array([float(r['node_heat_capacity_J_K']) for r in subset])
            rise = np.array([float(r['temperature_at_55p5s_C']) - float(r['temperature_at_38s_C']) for r in subset])
            c = sum(capacities)
            exact += capacities @ rise
            total_capacity += c
            midpoint += c * np.interp(0.5, x, rise)
            probe_x = (1 + np.array([-1., 1.]) / np.sqrt(3.)) / 2.
            two_depth += c * np.interp(probe_x, x, rise).mean()
        q = [r for r in flux if int(r['mesh']) == mesh]
        times = [float(r['time_s']) for r in q]
        assert times[0] == 38. and times[-1] == 55.5
        qin = np.trapezoid([float(r['interface_into_plates_W']) for r in q], times)
        qloss = np.trapezoid([float(r['external_loss_W']) for r in q], times)
        summaries.append(dict(mesh=mesh,capacity_J_K=total_capacity,rise_K=exact/total_capacity,
                              full_storage_J=exact,midpoint_estimate_J=midpoint,
                              two_depth_estimate_J=two_depth,two_depth_bias_J=two_depth-exact,
                              interface_heat_J_trapezoid=qin,external_loss_J_trapezoid=qloss,
                              sampled_quadrature_difference_J=qin-qloss-exact))
    return summaries


def spatial_spreads():
    rows = [r for r in read_csv(ROOT/'results/spatial/all_cases.csv') if r['refine']=='4' and float(r['shortening_s'])==0.]
    assert len(rows) == 5
    spans = {}
    for field in ('cutoff_end_minus_center_K','120s_end_minus_center_K','late_minimum_mea_mean_C'):
        values = [float(r[field]) for r in rows]
        spans[field] = dict(min=min(values),max=max(values),span=max(values)-min(values))
    return dict(n=len(rows),active_cells=sorted(set(int(r['late_minimum_cell_one_based']) for r in rows)),
                active_time_s=sorted(set(float(r['late_minimum_time_s']) for r in rows)),ranges=spans)


def dense_minimum(z):
    best = np.inf
    for (left,right),coefficients,order in zip(z['dense_times_s'],z['properties'],z['orders']):
        if right<=120 or left>=180:
            continue
        lo, hi = max(0.,(120.-left)/(right-left)), min(1.,(180.-left)/(right-left))
        for cell in range(coefficients.shape[1]):
            c = coefficients[:int(order)+1,cell]
            extrema = poly.polyroots(poly.polyder(c))
            points=[lo,hi]+[t.real for t in extrema if abs(t.imag)<1e-8 and lo<t.real<hi]
            best=min(best,float(poly.polyval(points,c).min()))
    return best


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('recalculated_review_results.json'))
    args=ap.parse_args()
    profiles = plate_storage()
    expected=read_csv(DATA/'plate_storage_reference.csv')
    for row,ref in zip(profiles,expected):
        for key,value in row.items():
            assert np.isclose(value,float(ref[key]),rtol=1e-12,atol=1e-8),(key,value,ref[key])
    fine=[row for row in read_csv(DATA/'mesh_minima.csv') if row['mesh']=='8']
    computed={}
    for row in fine:
        with np.load(DATA/'refined_traces'/(row['name']+'.npz'),allow_pickle=False) as z:
            minimum=dense_minimum(z)
        assert abs(minimum-float(row['minimum_C']))<1e-8, row['name']
        computed[row['name']]=minimum
    assert len(computed)==10
    result=dict(plate_storage=profiles,spatial_distribution=spatial_spreads(),finest_mesh_window_minima_C=computed)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"Verified plate storage on two meshes, five heat distributions and {len(computed)} finest-mesh window minima.")


if __name__ == '__main__':
    main()

