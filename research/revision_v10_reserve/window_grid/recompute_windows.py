"""Window and mesh sensitivity on saved five-cell temperature states."""
from pathlib import Path
import hashlib, json, csv
import numpy as np
from numpy.polynomial import polynomial as poly

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'research/study_v2'
LEVELS=(0.,.01,.5,1.,2.,3.)
WINDOWS=((90.,150.),(105.,165.),(120.,180.),(135.,180.))
STARTS=np.arange(0.,120.5,.5)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, rows):
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sources():
    for r in read(BASE/'representatives.json')['representatives']:
        for g,name in zip((2,4),r['validation_jobs']):
            yield r['family'],float(r['reserve_C']),'constant',g,BASE/'runs/validation'/name
    for fam in ('uniform','spatial'):
        for wave in ('front_loaded','back_loaded'):
            for g in (2,4):yield fam,1.,wave,g,BASE/'runs/waveform'/f'wave_{fam}_{wave}_r{g}'

def passing_starts(t,T,threshold):
    """Intersect all five affine cell inequalities, then subtract 60 s from each qualifying end."""
    spans=[]
    for t0,t1,a,b in zip(t[:-1],t[1:],T[:-1],T[1:]):
        lo,hi=0.,1.
        for value,slope in zip(a,b-a):
            if slope>0:lo=max(lo,(threshold-value)/slope)
            elif slope<0:hi=min(hi,(threshold-value)/slope)
            elif value<threshold:lo=2.;break
        if lo<=hi:
            begin,end=float(t0+lo*(t1-t0)),float(t0+hi*(t1-t0))
            if spans and begin<=spans[-1][1]+1e-10:spans[-1][1]=end
            else:spans.append([begin,end])
    return [[a,b-60.] for a,b in spans if b-a>=60.]

def contains(spans,t):return any(a<=t<=b for a,b in spans)

def sample(t,T,a,b):
    lo,hi=int(2*a),int(2*b); assert t[lo]==a and t[hi]==b
    cells=T[lo:hi+1]; i,j=np.unravel_index(np.argmin(cells),cells.shape)
    return float(cells[i,j]),float(t[lo+i]),int(j+1)

def dense_minimum(z,a,b):
    times=z['dense_segment_times_s']; coeffs=z['dense_temperature_coefficients_C']; orders=z['dense_orders']
    assert times[0,0]<=a and times[-1,1]>=b-1e-10 and np.abs(times[1:,0]-times[:-1,1]).max()<1e-9
    best=(np.inf,None,None)
    for k in np.flatnonzero((times[:,1]>a)&(times[:,0]<b)):
        t0,t1=times[k];lo=max(0.,(a-t0)/(t1-t0));hi=min(1.,(b-t0)/(t1-t0))
        for j in range(5):
            c=coeffs[k,:orders[k]+1,j]; d=np.arange(1,len(c))*c[1:]; xs=[lo,hi]
            # The leading derivative term bounds the sign on [0,1] when it exceeds the sum of the others.
            if len(d)>1 and abs(d[0])<=np.abs(d[1:]).sum():
                xs.extend(float(r.real) for r in poly.polyroots(d) if abs(r.imag)<1e-8 and lo<r.real<hi)
            vals=poly.polyval(xs,c);i=int(np.argmin(vals))
            if vals[i]<best[0]:best=float(vals[i]),float(t0+xs[i]*(t1-t0)),j+1
    return best

def pair(rows):
    groups={}
    for r in rows:groups.setdefault((r['design'],r['start_s'],r['end_s']),{})[r['grid']]=r
    paired=[]
    for (name,a,b),g in groups.items():
        c,f=g[2],g[4]; cm,fm=c['minimum_C'],f['minimum_C']
        paired.append(dict(design=name,start_s=a,end_s=b,duration_s=b-a,r2_C=cm,r4_C=fm,r4_minus_r2_K=fm-cm,
            r2_limiting_time_s=c['active_time_s'],r4_limiting_time_s=f['active_time_s'],r2_limiting_cell=c['active_cell'],r4_limiting_cell=f['active_cell'],
            flipped_thresholds_C='|'.join(f'{v:g}' for v in LEVELS if (fm>=v)!=(cm>=v))))
    return paired

def main():
    named=[];scan=[];ranges=[];dense=[];interval_map={};err=0.
    cfg=BASE/'representatives.json';inputs={cfg.relative_to(ROOT).as_posix():sha(cfg)}
    for family,reserve,wave,g,stem in sources():
        key=f'{family}_{reserve:g}_{wave}';p=stem.with_suffix('.npz');q=stem.with_suffix('.json');j=read(q)
        for f in (p,q):inputs[f.relative_to(ROOT).as_posix()]=sha(f)
        assert j['states_archive_sha256']==sha(p) and j['status']=='COMPLETE' and j['audit']['passed'] and j['time_s']==180 and j['refine']==g
        with np.load(p,allow_pickle=False) as z:
            t=z['time_s'];np.testing.assert_array_equal(t,np.arange(0.,180.5,.5))
            a,b=j['state_slices']['T'];nodes=z['states_scaled'][:,a:b]*z['state_scale'][a:b]-273.15;dx=z['mesh_dx_m']
            T=np.column_stack([np.average(nodes[:,i],axis=1,weights=dx[i]) for i in [z[f'mea_indices_{i}'] for i in range(5)]])
            err=max(err,float(np.abs(T-np.asarray(j['series']['temperature_C'])).max()));assert err<5e-10
            common=dict(design=key,family=family,nominal_design_target_C=reserve,profile=wave,grid=g,source_job=j['job_id'])
            def assess(a,b):
                m,when,c=sample(t,T,a,b)
                return dict(**common,start_s=a,end_s=b,duration_s=b-a,minimum_C=m,active_time_s=when,active_cell=c,
                    active_cells_within_1e_minus8_K='|'.join(str(i+1) for i in np.flatnonzero(abs(T[int(2*when)]-m)<=1e-8)),
                    **{f'passes_{v:g}_C':int(m>=v) for v in LEVELS})
            for a,b in WINDOWS:
                row=assess(a,b);named.append(row)
                if a>=120:
                    m,when,c=dense_minimum(z,a,b)
                    dense.append(dict(**common,start_s=a,end_s=b,state_minimum_C=row['minimum_C'],BDF_minimum_C=m,
                        dense_minus_state_minimum_K=m-row['minimum_C'],BDF_active_time_s=when,BDF_active_cell=c,
                        threshold_flips_state_vs_dense='|'.join(str(v) for v in LEVELS if (m>=v)!=(row['minimum_C']>=v))))
                    if a==120:assert abs(m-j['dense_dwell']['minimum_mea_mean_C'])<1e-8
            windows=[assess(float(a),float(a+60)) for a in STARTS];scan.extend(windows)
            for v in LEVELS:
                spans=passing_starts(t,T,v);interval_map[key,g,v]=spans
                assert all(contains(spans,r['start_s'])==bool(r[f'passes_{v:g}_C']) for r in windows)
                ranges.append(dict(**common,threshold_C=v,earliest_start_s=spans[0][0] if spans else None,
                    latest_start_s=spans[-1][1] if spans else None,start_intervals_s=json.dumps(spans),
                    total_duration_of_start_intervals_s=sum(b-a for a,b in spans)))
    flips=[]
    for key in sorted({r['design'] for r in scan}):
        for v in LEVELS:
            c,f=interval_map[key,2,v],interval_map[key,4,v];cuts=sorted(set([0.,120.]+[x for s in c+f for x in s]))
            for a,b in zip(cuts,cuts[1:]):
                m=(a+b)/2;coarse,fine=contains(c,m),contains(f,m)
                if coarse!=fine:
                    flips.append(dict(design=key,threshold_C=v,lower_start_s=a,upper_start_s=b,width_s=b-a,
                        r2_passes_interior=int(coarse),r4_passes_interior=int(fine),
                        include_lower=int((contains(c,a),contains(f,a))==(coarse,fine)),include_upper=int((contains(c,b),contains(f,b))==(coarse,fine))))
    paired=pair(named); s_pairs=pair(scan)
    for n,r in [('named_windows',named),('rolling_60s',scan),('continuous_start_intervals',ranges),('named_grid_pairs',paired),
                ('rolling_grid_pairs',s_pairs),('continuous_grid_flips',flips),('dense_checks',dense)]:save(n,r)
    audit=dict(status='PASS',source_trajectories=len(list(sources())),named_window_rows=len(named),rolling_window_rows=len(scan),
               continuous_pass_interval_sets=len(ranges),dense_cross_checks=len(dense),maximum_reconstructed_mean_error_K=err,
               maximum_abs_dense_vs_sampled_K=max(abs(r['dense_minus_state_minimum_K']) for r in dense),
               maximum_abs_named_mesh_delta_K=max(abs(r['r4_minus_r2_K']) for r in paired),
               named_pairs_with_grid_flip=sum(bool(r['flipped_thresholds_C']) for r in paired),named_pairs_total=len(paired),
               dense_vs_state_flips=sum(bool(r['threshold_flips_state_vs_dense']) for r in dense),
               continuous_flip_intervals=len(flips),largest_width_of_continuous_grid_disagreement_s=max(r['width_s'] for r in flips))
    print(json.dumps(audit,indent=2));audit['source_sha256']=inputs
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()

