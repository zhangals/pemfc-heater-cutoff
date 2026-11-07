"""Clip start intervals to full load and heater-off operation; tabulate limiting-cell heat/bias."""
from pathlib import Path
import argparse,csv,json
import numpy as np

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table_block(n,cap,headers,rows,widths):
    return {'type':'table','id':f'Table S{n}','caption':dict(zip(('en','zh'),cap)),
            'headers':dict(zip(('en','zh'),headers)),'rows':rows,'widths':widths}

def compute(src,out):
    with (src/'window_start_input.csv').open(encoding='utf-8-sig') as f: orig=list(csv.DictReader(f))
    out.mkdir(parents=True,exist_ok=True)
    schedules={};clipped=[]
    for row in orig:
        intervals=json.loads(row['start_intervals_s'])
        lower=max(60.,float(row['cutoff_s']))
        kept=[[max(a,lower),min(b,120.)] for a,b in intervals if max(a,lower)<=min(b,120.)]
        earliest=kept[0][0] if kept else None
        clipped.append({**row,'ready_operation_floor_s':lower,'eligible_start_intervals_s':json.dumps(kept),
                        'earliest_eligible_start_s':earliest,'latest_eligible_start_s':kept[-1][1] if kept else None})
        family,reserve,profile=row['family'],float(row['nominal_design_target_C']),row['profile']
        key=(family,reserve,profile);schedules.setdefault(key,{})[float(row['threshold_C'])]=earliest
    assert len(schedules)==10 and len(orig)==60
    write_csv(out/'window_start_clipped.csv',clipped)
    rows=[]
    for (fam,res,wave),r in schedules.items():
        name={'uniform':'Uniform','spatial':'Spatial'}[fam]
        zh={'uniform':'均匀','spatial':'空间'}[fam]
        wave_en={'constant':'','front_loaded':', front-loaded','back_loaded':', back-loaded'}[wave]
        wave_zh={'constant':'','front_loaded':'，前置','back_loaded':'，后置'}[wave]
        rows.append([{'en':f'{name}, {res:g}{wave_en}','zh':f'{zh}，{res:g}{wave_zh}'}]+
                    [f'{r[t]:.2f}' if r[t] is not None else '—' for t in (0,.01,.5,1,2,3)])
    thresholds=['0','0.01','0.5','1','2','3']
    s40=table_block(40,('Earliest eligible start (s) of a continuous 60 s interval above each cell-mean threshold (°C) on the finest mesh. The interval starts after or at both heater cutoff and the 60 s current plateau. At a given threshold an em dash denotes an empty set of eligible starts. Labels give the nominal design reserve in °C and any waveform modification.',
                       '最细网格上各片均温连续60 s达到各列温度阈值（℃）的最早可行起点（s）。窗口起点须已到达60 s电流平台，且该方案已经关断加热。破折号表示该阈值下没有可行起点。调度名称给出名义设计储备（℃）及波形变化。'),
                      (['Schedule']+thresholds,['调度']+thresholds),rows,[1.9,.85,.85,.85,.85,.85,.85])
    terms=['conduction','reaction','phase','auxiliary','convection_signed']
    with (src/'limiting_cell_integrals.csv').open(encoding='utf-8') as f: budgets=list(csv.DictReader(f))
    jrows=[];tbl=[]
    with np.load(src/'limiting_cell_endpoint_states.npz',allow_pickle=False) as arrays:
        for r in budgets:
            prefix=r['policy']+'_r'+r['mesh']
            field=lambda name:arrays[prefix+'_'+name]
            delta=field('Tend_K')-field('Tstart_K')
            raw_j=np.dot(field('capacity_J_K'),delta)
            raw_k=np.average(delta,weights=field('dx_m'))
            assert np.isclose(raw_j,float(r['storage_J']),rtol=0,atol=1e-8)
            assert np.isclose(raw_k,float(r['mean_change_K']),rtol=0,atol=1e-9)
            cj=raw_j-sum(float(r[key+'_J']) for key in terms);ck=raw_k-sum(float(r[key+'_K']) for key in terms)
            assert abs(cj)<1e-5 and abs(ck)<5e-5,(r['policy'],cj,ck)
            assert all(abs(float(r[k+'_'+u]))<1e-11 for k in ('auxiliary','convection_signed') for u in ('J','K'))
            jrows.append({**r,'independently_reconstructed_dU_J':raw_j,'independently_reconstructed_dT_K':raw_k,
                          'recomputed_closure_J':cj,'recomputed_closure_K':ck})
            lab={'preheat':{'en':'Preheat','zh':'预热'},'concurrent':{'en':'Concurrent','zh':'同步加热'}}[r['policy']]
            tbl.append([lab,r['mesh'],r['limiting_cell'],*[
                '{:.2f}\n{:.2f}'.format(float(r[t+'_J']),float(r[t+'_K'])) for t in ('conduction','reaction','phase')],
                f'{raw_j:.3f}\n{raw_k:.3f}',f'{cj:.1e}\n{ck:.1e}'])
    write_csv(out/'limiting_cell_budgets.csv',jrows)
    s43=table_block(43,('Post-cutoff balances for the limiting MEA from heater cutoff to its minimum temperature. Each paired entry reports sensible energy (J, upper line) and the thickness-weighted mean-temperature contribution (K, lower line). The latter weights each nodal power by its layer thickness fraction divided by its own thermal capacity. Closure is the observed change minus summed contributions. Direct heater and boundary-convection inputs to this MEA are zero during the interval.',
        '从停热到最低温时刻，限制片膜电极的局部热收支。成对数值上行为显热项（J），下行为厚度加权均温项（K）。均温项按各节点厚度占比除以该节点热容对功率加权。闭合差为实际变化减去各贡献之和，此区间膜电极的直接加热与边界对流项均为零。'),
        (['Policy','Mesh','Cell','Conduction','Reaction','Phase','Change','Closure'],['工况','网格','片号','导热','反应','水相转移','变化','闭合差']),tbl,[1.25,.57,.51,1.08,1.08,1.02,1,1.18])
    (out/'si_blocks.json').write_text(json.dumps({'table_s40':s40,'table_s43':s43},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return clipped,jrows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--out',type=Path,default=Path('recalculated_diagnostics'))
    a=p.parse_args();compute(a.inputs,a.out)

