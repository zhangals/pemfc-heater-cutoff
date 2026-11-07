"""Design a plate-storage test using the published matched-cutoff simulation.
Extract simulated endplate profiles and powers from regenerated stack-model states.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]

def write_rows(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=rows[0]);out.writeheader();out.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'research/revision_v10_reserve/matched_cutoff/runs')
    parser.add_argument('--output', type=Path, default=Path('reruns/simulated_plate_profiles'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    moments=(38.,55.5)
    grid_summary=[]
    nodes=[]
    flux=[]
    for r in [2,4]:
        src=args.input/f'first_1C_r{r}.npz'
        with np.load(src,allow_pickle=False) as z:
            t=z['time_s'];kind=z['mesh_kind'];C=z['mesh_capacity_J_K'];x=z['x_m'];dx=z['mesh_dx_m']
            # Physical state vectors store the thermal nodes first, in kelvin.
            T=z['sampled_states_physical'][:,:len(kind)]
            idx=[int(np.flatnonzero(np.isclose(t,m,rtol=0,atol=1e-9))[0]) for m in moments]
            ep=np.flatnonzero(kind=='EP')
            plates=np.split(ep,np.where(np.diff(ep)>1)[0]+1)
            assert len(plates)==2
            exact=one=two=0.
            for n,inds in enumerate(plates):
                left=x[inds[0]]-dx[inds[0]]/2
                xi=(x[inds]-left)/dx[inds].sum()
                dt=T[idx[1],inds]-T[idx[0],inds]
                cap=C[inds].sum()
                exact+=C[inds]@dt
                one+=cap*np.interp(.5,xi,dt)
                sites=(1+np.array([-1.,1.])/np.sqrt(3))/2
                two+=cap*np.mean(np.interp(sites,xi,dt))
                for j,i in enumerate(inds):
                    nodes.append(dict(mesh=r,plate=n+1,x_over_L=xi[j],node_heat_capacity_J_K=C[i],
                                      temperature_at_38s_C=T[idx[0],i]-273.15,temperature_at_55p5s_C=T[idx[1],i]-273.15))
            change=z['sensible_EP_J'][idx[1]]-z['sensible_EP_J'][idx[0]]
            assert abs(exact-change)<1e-7
            first,last=np.flatnonzero(kind!='EP')[[0,-1]];g=z['mesh_conductance_W_K'];boundary=z['mesh_boundary_conductance_W_K']
            assert kind[first]==kind[last]=='BP'
            # Both interface powers are positive for heat entering an endplate.
            inlet=g[first-1]*(T[:,first]-T[:,first-1])+g[last]*(T[:,last]-T[:,last+1])
            loss=boundary[0]*(T[:,0]-243.15)+boundary[1]*(T[:,-1]-243.15)
            sl=slice(idx[0],idx[1]+1)
            qin=float(np.trapezoid(inlet[sl],t[sl]));qout=float(np.trapezoid(loss[sl],t[sl]))
            for time,qi,qo in zip(t[sl],inlet[sl],loss[sl]):flux.append(dict(mesh=r,time_s=time,interface_into_plates_W=qi,external_loss_W=qo))
            grid_summary.append(dict(mesh=r,capacity_J_K=C[ep].sum(),rise_K=exact/C[ep].sum(),full_storage_J=exact,
                                     midpoint_estimate_J=one,two_depth_estimate_J=two,two_depth_bias_J=two-exact,
                                     interface_heat_J_trapezoid=qin,external_loss_J_trapezoid=qout,
                                     sampled_quadrature_difference_J=qin-qout-exact,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    write_rows(args.output/'simulated_endplate_profiles.csv',nodes)
    write_rows(args.output/'simulated_boundary_powers.csv',flux)
    write_rows(args.output/'simulated_plate_storage.csv',grid_summary)
    (args.output/'summary.json').write_text(json.dumps({'status':'PASS','interval_s':moments,'scope':'Numerical prediction and virtual probe placement only','results':grid_summary},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

