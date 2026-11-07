"""Read-only thermal-budget postprocessing of fresh v2 baseline archives.

No ODE integration, legacy-result loading, refitting or artifact mutation.
Regional sensible-energy balances and whole-stack enthalpy are kept separate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TERMS = ('conduction', 'auxiliary', 'reaction', 'phase', 'convection_signed')
PHASE_TERMS = ('vapor_bound', 'condensation_evaporation', 'deposition_sublimation', 'freezing_melting')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside_v2(path):
    path = Path(path).resolve()
    path.relative_to(HERE.resolve())
    return path


def load_runtime():
    global np, Polynomial, engine, solver, LV, LF, TM
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    # The engine sets BLAS thread limits before NumPy is imported.
    import engine
    import numpy as np
    from numpy.polynomial import Polynomial
    from 模型物理 import LV, LF, TM
    solver = engine.solver


class DenseStage:
    def __init__(self, archive, prefix, endpoint, model):
        self.prefix, self.endpoint, self.model = prefix, float(endpoint), model
        self.times = archive[prefix + 'dense_segment_times_s']
        self.orders = archive[prefix + 'dense_orders']
        self.D = archive[prefix + 'full_dense_D_scaled']
        self.shift = archive[prefix + 'full_dense_t_shift_s']
        self.denom = archive[prefix + 'full_dense_denom_s']
        self.initial = archive[prefix + 'initial_state_scaled']
        self.final = archive[prefix + 'final_state_scaled']
        self.indices = [i for i, (a, b) in enumerate(self.times) if a < self.endpoint and b > 0]
        if not self.indices:
            raise ValueError('No full-state dense coverage for ' + prefix)
        cursor = 0.
        for i in self.indices:
            lo, hi = self.times[i]
            if abs(float(lo) - cursor) > 1e-8:
                raise ValueError('Gap or overlap in accepted dense segments')
            cursor = min(float(hi), self.endpoint)
            order = int(self.orders[i])
            if not 1 <= order <= 5 or np.any(self.denom[i, :order] == 0):
                raise ValueError('Invalid BDF dense order or denominator')
        if abs(cursor-self.endpoint) > 1e-8:
            raise ValueError('Dense archive does not cover actual endpoint')
        sampled_times = archive[prefix + 'time_s']
        sampled_states = archive[prefix + 'states_scaled']
        if sampled_states.shape != (len(sampled_times), model.size):
            raise ValueError('Unexpected sampled-state orientation')
        self.reconstruction_error_scaled = max(
            float(np.max(np.abs(self.state_at(float(t))-y)))
            for t, y in zip(sampled_times, sampled_states))
        self.reconstruction_error_scaled = max(
            self.reconstruction_error_scaled,
            float(np.max(np.abs(self.state(self.indices[0], 0.)-self.initial))),
            float(np.max(np.abs(self.state(self.indices[-1], self.endpoint)-self.final))))
        if self.reconstruction_error_scaled > 1e-7:
            raise ValueError('Full-state dense reconstruction disagrees with saved states')

    def state(self, i, t):
        order = int(self.orders[i])
        products = np.cumprod((float(t)-self.shift[i, :order])/self.denom[i, :order])
        return self.D[i, 0] + products @ self.D[i, 1:order+1]

    def state_at(self, t):
        if abs(t) <= 1e-12:
            return self.initial.copy()
        if abs(t-self.endpoint) <= 1e-10:
            return self.final.copy()
        i = int(np.searchsorted(self.times[:, 1], t, side='left'))
        if i >= len(self.times) or t < self.times[i, 0]-1e-9 or t > self.endpoint+1e-9:
            raise ValueError('Time outside accepted dense domain')
        return self.state(i, t)

    def minima(self, start=0., stop=None):
        stop = self.endpoint if stop is None else min(float(stop), self.endpoint)
        m = self.model
        best = [dict(cell_one_based=k+1, temperature_C=float('inf'), time_s=None) for k in range(m.nc)]
        for i in self.indices:
            left, right = map(float, self.times[i]); span = right-left
            lo, hi = max(start,left), min(stop,right)
            if hi < lo:
                continue
            order = int(self.orders[i])
            dT = self.D[i, :order+1, m.sl['T']] * m.scale[m.sl['T']][None, :]
            means = m.mesh.mea_means(dT)
            for k in range(m.nc):
                p = Polynomial([means[0,k]-TM]); product = Polynomial([1.])
                for j in range(order):
                    product *= Polynomial([(left-self.shift[i,j])/self.denom[i,j], span/self.denom[i,j]])
                    p += means[j+1,k]*product
                u0, u1 = (lo-left)/span, (hi-left)/span
                candidates = [u0,u1]
                for root in p.deriv().roots():
                    if abs(root.imag) < 1e-9 and u0 <= root.real <= u1:
                        candidates.append(float(root.real))
                for u in candidates:
                    value = float(p(u))
                    if value < best[k]['temperature_C']:
                        best[k].update(temperature_C=value, time_s=float(left+span*u))
        global_min = min(best, key=lambda d:d['temperature_C'])
        return dict(per_cell=best, global_minimum=global_min)


class PowerDecomposition:
    def __init__(self, model, current, heater, ambient):
        self.m, self.current, self.heater, self.ambient = model,current,heater,float(ambient)
        mesh = model.mesh
        masks = {name:np.isin(mesh.kind, kinds) for name,kinds in
                 [('MEA',['AGDL','ACL','PEM','CCL','CGDL']),('BP',['BP']),('EP',['EP'])]}
        if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
            raise ValueError('MEA/BP/EP partition is not exhaustive and disjoint')
        self.names = ['all','MEA','BP','EP']+[f'MEA_cell_{k+1}' for k in range(model.nc)]
        self.G = np.zeros((len(self.names),model.nt)); self.G[0] = 1.
        for row,name in enumerate(['MEA','BP','EP'],1):self.G[row] = masks[name]
        self.W = np.zeros((model.nc,model.nt))
        for k,ids in enumerate(mesh.mea_indices):
            self.G[4+k,ids] = 1.
            self.W[k,ids] = mesh.dx[ids]/mesh.dx[ids].sum()/mesh.capacity_J_K[ids]
        self.face_names = ['MEA_to_BP','BP_to_EP','MEA_to_EP']
        self.face_weights = []
        for a,b in [('MEA','BP'),('BP','EP'),('MEA','EP')]:
            self.face_weights.append((masks[a][:-1] & masks[b][1:]).astype(int)
                                     -(masks[b][:-1] & masks[a][1:]).astype(int))
        self.face_weights = np.array(self.face_weights)
        self.n_energy = len(self.names)*len(TERMS)
        self.n_temperature = model.nc*len(TERMS)
        self.n_phase = len(self.names)*len(PHASE_TERMS)
        self.size = self.n_energy+self.n_temperature+self.n_phase+3
        self.max_power_identity_error_W = 0.
        self.max_thermal_operator_error_W = 0.
        self.max_internal_conduction_sum_W = 0.
        self.rate_evaluations = 0

    def node_terms(self, t, y, check=False):
        m = self.m; mesh=m.mesh; j=float(self.current(t)); q=np.asarray(self.heater(t),dtype=float)
        d,z,_=m.quantities(y,j); T=d['T'][:,0]; V=z['V'][:,0]
        (vb,lb,vl,li,vi),_,_=solver._phase(m,y,j,masks=None)
        phase_density=np.array([LV*vb[:,0],LV*vl[:,0],(LV+LF)*vi[:,0],LF*li[:,0]])
        phase=np.zeros((4,m.nt));phase[:,m.pidx]=phase_density*mesh.area_m2*m.dx_p[:,0]
        powers=np.zeros((5,m.nt))
        flux=mesh.conductance_W_K*(T[:-1]-T[1:])
        powers[0,:-1]-=flux;powers[0,1:]+=flux
        powers[1]=m.heat_b@(m.cfg.area_cm2*q)
        powers[2]=m.heat_c@(mesh.area_m2*1e4*j*(1.48-V))
        powers[3]=phase.sum(axis=0)
        powers[4,0]=-mesh.boundary_conductance_W_K[0]*(T[0]-(self.ambient+TM))
        powers[4,-1]=-mesh.boundary_conductance_W_K[1]*(T[-1]-(self.ambient+TM))
        self.rate_evaluations+=1
        self.max_internal_conduction_sum_W=max(self.max_internal_conduction_sum_W,abs(float(powers[0].sum())))
        if check:
            dy=m.rhs(t,y,j,q,self.ambient)*m.scale
            rhs_W=mesh.capacity_J_K*dy[m.sl['T']]
            self.max_power_identity_error_W=max(self.max_power_identity_error_W,float(np.max(np.abs(rhs_W-powers.sum(axis=0)))))
            operator=-mesh.K_W_K@(T-(self.ambient+TM))
            self.max_thermal_operator_error_W=max(self.max_thermal_operator_error_W,float(np.max(abs(operator-powers[0]-powers[4]))))
        return powers,phase,flux

    def projected(self,t,y,check=False):
        powers,phase,flux=self.node_terms(t,y,check)
        return np.r_[(self.G@powers.T).ravel(),(self.W@powers.T).ravel(),
                     (self.G@phase.T).ravel(),self.face_weights@flux]

    def split(self,vector):
        a=self.n_energy;b=a+self.n_temperature;c=b+self.n_phase
        return (vector[:a].reshape(len(self.names),5),
                vector[a:b].reshape(self.m.nc,5),vector[b:c].reshape(len(self.names),4),vector[c:])


def integrate_budget(stage,power,start,stop,args):
    nodes_low,weights_low=np.polynomial.legendre.leggauss(args.quad_low)
    nodes_high,weights_high=np.polynomial.legendre.leggauss(args.quad_high)
    atol=np.full(power.size,args.atol_J)
    atol[power.n_energy:power.n_energy+power.n_temperature]=args.atol_K
    total=np.zeros(power.size);error=np.zeros(power.size);unresolved=[]
    if stop <= start:
        return total,dict(estimated_error=error.tolist(),unresolved_subintervals=[],accepted_segments=0)
    span=stop-start

    def quadrature(i,a,b,nodes,weights):
        times=(a+b)/2+(b-a)/2*nodes
        values=np.array([power.projected(float(t),stage.state(i,float(t))) for t in times])
        return (b-a)/2*(weights@values)

    def adaptive(i,a,b,depth):
        low=quadrature(i,a,b,nodes_low,weights_low);high=quadrature(i,a,b,nodes_high,weights_high)
        difference=np.abs(high-low)
        tolerance=atol*((b-a)/span)+args.quad_rtol*np.maximum(abs(low),abs(high))
        if np.all(difference<=tolerance):return high,difference
        if depth>=args.max_depth:
            unresolved.append(dict(t0_s=a,t1_s=b,maximum_error_ratio=float(np.max(difference/np.maximum(tolerance,1e-30)))))
            return high,difference
        mid=(a+b)/2
        x,xe=adaptive(i,a,mid,depth+1);y,ye=adaptive(i,mid,b,depth+1)
        return x+y,xe+ye

    accepted=0
    for i in stage.indices:
        a=max(start,float(stage.times[i,0]));b=min(stop,float(stage.times[i,1]))
        if b<=a:continue
        power.projected((a+b)/2,stage.state(i,(a+b)/2),check=True)
        value,err=adaptive(i,a,b,0);total+=value;error+=err;accepted+=1
    return total,dict(estimated_error=error.tolist(),unresolved_subintervals=unresolved,accepted_segments=accepted)


def state_inventory(model,y):
    d,_=model.unpack(y);T=d['T'][:,0]
    sensible=model.mesh.capacity_J_K*(T-TM)
    latent=model.mesh.area_m2*np.sum(model.dx_p[:,0]*(LV*d['mv'][:,0]-LF*d['mi'][:,0]))
    return sensible,float(sensible.sum()+latent),{key:float(d[key].sum()) for key in engine.COUNTERS}


def window_budget(stage,power,start,stop,args):
    values,qa=integrate_budget(stage,power,start,stop,args)
    energies,temperatures,phase,face=power.split(values)
    y0=stage.state_at(start);y1=stage.state_at(stop);m=stage.model
    s0,h0,c0=state_inventory(m,y0);s1,h1,c1=state_inventory(m,y1)
    deltaS=power.G@(s1-s0)
    T0=y0[m.sl['T']]*m.scale[m.sl['T']];T1=y1[m.sl['T']]*m.scale[m.sl['T']]
    deltaT=m.mesh.mea_means(T1)-m.mesh.mea_means(T0)
    dc={key:c1[key]-c0[key] for key in c0}
    rH=h1-h0+dc['conv']+LV*dc['wout']-dc['rxn']-dc['aux']
    hscale=max(1.,abs(h1-h0)+abs(dc['conv'])+abs(LV*dc['wout'])+abs(dc['rxn'])+abs(dc['aux']))
    region=[]
    for i,name in enumerate(power.names):
        residual=float(deltaS[i]-energies[i].sum())
        scale=max(1.,abs(float(deltaS[i])),float(np.abs(energies[i]).sum()))
        region.append(dict(region=name,sensible_change_J=float(deltaS[i]),
                           contributions_J=dict(zip(TERMS,energies[i].tolist())),
                           phase_components_J=dict(zip(PHASE_TERMS,phase[i].tolist())),
                           closure_residual_J=residual,closure_relative=abs(residual)/scale))
    cell=[]
    for k in range(m.nc):
        cell.append(dict(cell_one_based=k+1,mean_temperature_change_K=float(deltaT[k]),
                         contributions_K=dict(zip(TERMS,temperatures[k].tolist())),
                         closure_residual_K=float(deltaT[k]-temperatures[k].sum())))
    native_cross_checks={'auxiliary_J':dc['aux']-energies[0,1],
                         'reaction_J':dc['rxn']-energies[0,2],
                         'convection_loss_J':dc['conv']+energies[0,4]}
    return dict(start_s=start,stop_s=stop,regions=region,cells=cell,
                signed_exchange_J=dict(zip(power.face_names,face.tolist())),quadrature=qa,
                whole_stack_enthalpy=dict(change_J=h1-h0,counter_changes=dc,
                                          vapor_outflow_enthalpy_J=LV*dc['wout'],
                                          residual_J=rH,relative_residual=abs(rH)/hscale,
                                          phase_heat_included_again=False),
                counter_vs_power_quadrature_residual_J=native_cross_checks)


def verify_model_arrays(model,archive):
    expected=engine.model_arrays(model)
    for key,value in expected.items():
        if key not in archive:raise ValueError('Missing model array: '+key)
        actual=archive[key]
        if actual.shape!=value.shape:raise ValueError('Model array shape mismatch: '+key)
        if actual.dtype.kind in 'US' or value.dtype.kind in 'US':
            okay=np.array_equal(actual,value)
        else:okay=np.allclose(actual,value,rtol=1e-13,atol=1e-15)
        if not okay:raise ValueError('Rebuilt model differs from archive: '+key)


def process(path,args):
    path=inside_v2(path);record=json.loads(path.read_text(encoding='utf-8'))
    if record.get('phase')!='baseline' or not record.get('fresh_integration') or record.get('reused_prior_result'):
        raise ValueError('Input is not a fresh v2 baseline')
    if record.get('frozen_protocol_sha256')!=engine.FROZEN_PROTOCOL_SHA256:
        raise ValueError('Baseline protocol identity mismatch')
    if record.get('protocol',{}).get('study_id')!=engine.PROTOCOL['study_id']:
        raise ValueError('Study ID mismatch')
    for relative,checksum in record['sources'].items():
        if digest(ROOT/relative)!=checksum:
            raise ValueError('Source hash changed; explicit compatible-version review required: '+relative)
    archive_path=inside_v2(ROOT/record['states_archive'])
    if digest(archive_path)!=record['states_archive_sha256']:raise ValueError('NPZ checksum mismatch')
    model,initial,ambient,factor=engine.make_model(record['job'])
    if factor!=1.:raise ValueError('Nominal baseline expected')
    if record['state_slices']!={k:[v.start,v.stop] for k,v in model.sl.items()}:
        raise ValueError('State slice mismatch')
    output=dict(study_id=engine.PROTOCOL['study_id'],job_id=record['job_id'],policy=record['policy'],
                refine=record['refine'],sources={str(path.relative_to(ROOT)):digest(path),str(archive_path.relative_to(ROOT)):digest(archive_path)},
                source_code_hashes=record['sources'],postprocessor_sha256=digest(__file__),
                interpretation='Numerical power decomposition of existing fresh states, not new ODE integrations or physical validation.',
                sign_convention='Positive adds sensible heat; signed exchange A_to_B is positive from A into B.',
                regional_partition='MEA+BP+EP are disjoint; all and per-cell MEA rows are additional summaries and must not be summed with that partition.',
                quadrature_settings={key:getattr(args,key) for key in ['quad_low','quad_high','quad_rtol','atol_J','atol_K','max_depth']},stages={})
    snapshots=[]
    with np.load(archive_path,allow_pickle=False) as archive:
        verify_model_arrays(model,archive)
        for stage_name,prefix,raw in [('pre','pre_',record),('post','post_',record.get('postload'))]:
            if not raw:continue
            stage=DenseStage(archive,prefix,raw['time_s'],model)
            current=(lambda t:0.) if stage_name=='pre' and record['policy']=='preheat' else engine.current_ramp if stage_name=='pre' else (lambda t:.3)
            heater=(lambda t:np.ones(model.nc)) if stage_name=='pre' else (lambda t:np.zeros(model.nc))
            power=PowerDecomposition(model,current,heater,ambient)
            minima=stage.minima()
            windows={'full_interval':window_budget(stage,power,0.,stage.endpoint,args)}
            if stage_name=='post':
                tmin=minima['global_minimum']['time_s']
                windows['to_coldest_mean_minimum']=window_budget(stage,power,0.,tmin,args)
            for label,t in [('initial',0.),('final',stage.endpoint),('coldest_mean_minimum',minima['global_minimum']['time_s'])]:
                y=stage.state_at(t);T=y[model.sl['T']]*model.scale[model.sl['T']]-TM
                for i in range(model.nt):
                    snapshots.append(dict(stage=stage_name,snapshot=label,time_s=t,node=i,x_m=float(model.mesh.x[i]),
                                          material=str(model.mesh.kind[i]),cell_id_zero_based=int(model.mesh.cell_ids[i]),temperature_C=float(T[i])))
            output['stages'][stage_name]=dict(status=raw['status'],actual_endpoint_s=stage.endpoint,dense_minima=minima,
                                               full_state_reconstruction_max_error_scaled=stage.reconstruction_error_scaled,
                                               pointwise_checks=dict(max_power_identity_error_W=power.max_power_identity_error_W,
                                               max_thermal_operator_error_W=power.max_thermal_operator_error_W,
                                               max_internal_conduction_sum_W=power.max_internal_conduction_sum_W,
                                               rate_evaluations=power.rate_evaluations),windows=windows)
        if record.get('postload'):
            pre=archive['pre_final_state_scaled'];post=archive['post_initial_state_scaled']
            mask=np.ones(model.size,dtype=bool)
            for name in engine.COUNTERS:mask[model.sl[name]]=False
            output['physical_state_continuity_max_error_scaled']=float(np.max(np.abs(pre[mask]-post[mask])))
            if output['physical_state_continuity_max_error_scaled']!=0.:raise ValueError('Physical state changed between stages')
    output['quadrature_converged']=all(not w['quadrature']['unresolved_subintervals'] for s in output['stages'].values() for w in s['windows'].values())
    output['pointwise_identity_passed']=all(
        s['pointwise_checks']['max_power_identity_error_W'] <= 1e-6
        and s['pointwise_checks']['max_thermal_operator_error_W'] <= 1e-6
        and s['pointwise_checks']['max_internal_conduction_sum_W'] <= 1e-8
        for s in output['stages'].values())
    output['closure_requires_review']=any(abs(w['whole_stack_enthalpy']['relative_residual'])>1e-7 or any(r['closure_relative']>1e-4 for r in w['regions'])
                                         for s in output['stages'].values() for w in s['windows'].values())
    output['closure_review_rule']='Numerical diagnostic flags only: total-enthalpy relative residual > 1e-7 or regional sensible residual > 1e-4. Raw residuals retained; no physical safety interpretation.'
    destination=inside_v2(args.output_dir);destination.mkdir(parents=True,exist_ok=True)
    stem=destination/(record['job_id']+'.thermal_budget')
    targets=[stem.with_suffix(stem.suffix+ext) for ext in ['.json','.csv','.snapshots.csv']]
    if any(p.exists() for p in targets):raise FileExistsError('Use a new output directory; existing mechanism output will not be overwritten')
    with targets[0].open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    rows=[]
    for sname,s in output['stages'].items():
        for wname,w in s['windows'].items():
            for r in w['regions']:
                for term,value in r['contributions_J'].items():
                    rows.append(dict(stage=sname,window=wname,start_s=w['start_s'],stop_s=w['stop_s'],quantity='regional_sensible_energy',region=r['region'],term=term,value=value,unit='J',observed_change=r['sensible_change_J'],closure_residual=r['closure_residual_J']))
            for c in w['cells']:
                for term,value in c['contributions_K'].items():
                    rows.append(dict(stage=sname,window=wname,start_s=w['start_s'],stop_s=w['stop_s'],quantity='cell_mean_temperature',region=f'MEA_cell_{c["cell_one_based"]}',term=term,value=value,unit='K',observed_change=c['mean_temperature_change_K'],closure_residual=c['closure_residual_K']))
    for target,items in [(targets[1],rows),(targets[2],snapshots)]:
        with target.open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    print(json.dumps(dict(job_id=record['job_id'],outputs=[str(p.relative_to(ROOT)) for p in targets],quadrature_converged=output['quadrature_converged'],closure_requires_review=output['closure_requires_review']),ensure_ascii=False),flush=True)
    return output


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline_json',nargs='+',help='Fresh study_v2/runs/baseline/*.json; all input archives remain read-only.')
    p.add_argument('--output-dir',type=Path,default=HERE/'mechanism',help='New v2 output directory; existing outputs are never overwritten.')
    p.add_argument('--quad-low',type=int,default=4)
    p.add_argument('--quad-high',type=int,default=8)
    p.add_argument('--quad-rtol',type=float,default=1e-6)
    p.add_argument('--atol-J',dest='atol_J',type=float,default=1e-5)
    p.add_argument('--atol-K',dest='atol_K',type=float,default=1e-7)
    p.add_argument('--max-depth',type=int,default=6)
    args=p.parse_args()
    if not 3<=args.quad_low<args.quad_high or args.max_depth<0 or min(args.quad_rtol,args.atol_J,args.atol_K)<=0:
        p.error('Require 3 <= low < high, nonnegative depth and positive quadrature tolerances.')
    return args


if __name__=='__main__':
    arguments=parse_args()
    load_runtime()
    reports=[process(path,arguments) for path in arguments.baseline_json]
    if any(not d['quadrature_converged'] or not d['pointwise_identity_passed'] or d['closure_requires_review'] for d in reports):
        raise SystemExit(1)
