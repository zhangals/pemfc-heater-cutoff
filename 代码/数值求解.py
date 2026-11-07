# 共享求解模块
"""数值积分求解"""


from __future__ import annotations

import time
from weakref import WeakKeyDictionary

import numpy as np
from scipy.integrate import BDF
from scipy.optimize import brentq
from scipy.optimize._numdiff import group_columns
from scipy.sparse import csr_matrix

from 模型物理 import F, MW, TM, LV, LF, CLAMBDA, PREF, R, FuelCellModel, saturation

_GROUP_CACHE = WeakKeyDictionary()


# 相变分支计算
def _phase(m, y, j_c, masks=None):
    """相变分支判断"""
    cfg = m.cfg
    d, z, one = m.quantities(y, j_c)
    T, eg, lam = z['Tp'], z['eg'], z['lam']
    decisions = {} if masks is None else masks

    def positive(name, value):
        if masks is None:
            decisions[name] = value > 0.
        return np.where(decisions[name], value, 0.)


    def donor(name):
        value = d[name]
        key = 'donor_' + name
        if masks is None:
            decisions[key] = value >= 0.
        return np.where(decisions[key], value, 0.)

    mv, ml, mi = donor('mv'), donor('ml'), donor('mi')
    lp = np.zeros_like(eg)
    lp[m.pcl] = lam[m.bcl]
    ps, pi = saturation(T)
    pv = mv*R*T/(MW*eg)
    ratio = pv/ps
    if masks is None:
        decisions['activity_unsaturated'] = ratio < 1.
    a = np.where(decisions['activity_unsaturated'], ratio, 1.)
    le = .043 + 17.81*a - 39.85*a*a + 36*a**3
    fm = lambda value: value/(value+cfg.mstar)
    vb = m.pphi*CLAMBDA*(positive('vb_ads', le-lp)/cfg.tau_ads*fm(mv)
            - positive('vb_des', lp-le)/cfg.tau_des)*eg/m.eps0
    lb = m.pphi*CLAMBDA*(positive('lb_ads', 14.003-lp)/cfg.tau_lb*fm(ml)
            - positive('lb_des', lp-14.003)/cfg.tau_lb_des)
    vl = positive('vl_condense', mv-eg*MW*ps/(R*T))/cfg.tau_c \
            - ml/cfg.tau_e*positive('vl_evaporate', 1.-ratio)
    gf = positive('temperature_freeze', (TM-T)/cfg.delta_phase_K)
    gm = positive('temperature_melt', (T-TM)/cfg.delta_phase_K)
    li = cfg.gamma_f*(gf*ml/cfg.tau_f-gm*mi/cfg.tau_m)
    vi = gf/(1.+gf)*(fm(mi)*positive('vi_deposit', mv-eg*MW*pi/(R*T))/cfg.tau_di
            - mi/cfg.tau_su*positive('vi_sublimate', 1.-pv/pi))
    return (vb, lb, vl, li, vi), decisions, one


def frozen_rhs(m, t, y, j_c, q, ambient_C, reference_state, *, branches='Rvl'):
    """分支延拓计算"""
    if branches == 'Rvl':
        mv, _, ms = _rvl_data(m, np.asarray(reference_state)[:, None])
        return _with_rvl_mask(m, t, y, j_c, q, ambient_C, mv >= ms)
    if branches != 'all_phase':
        raise ValueError("branches must be 'Rvl' or 'all_phase'")
    _, masks, _ = _phase(m, reference_state, j_c)
    return _with_masks(m, t, y, j_c, q, ambient_C, masks)


def _with_masks(m, t, y, j_c, q, ambient_C, masks):
    original, _, one = _phase(m, y, j_c)
    frozen, _, _ = _phase(m, y, j_c, masks)
    vb, lb, vl, li, vi = [b-a for a, b in zip(original, frozen)]


    value = np.asarray(FuelCellModel.rhs(m, t, y, j_c, q, ambient_C))
    if one:
        value = value[:, None]
    correction = np.zeros_like(value)
    correction[m.sl['mb'].start+m.bcl] += vb[m.pcl]+lb[m.pcl]
    correction[m.sl['mv']] = -vb-vl-vi
    correction[m.sl['ml']] = vl-li-lb
    correction[m.sl['mi']] = li+vi
    heat = LV*(vb+vl)+(LV+LF)*vi+LF*li
    correction[m.pidx] = heat*m.dx_p*m.mesh.area_m2/m.mesh.capacity_J_K[m.pidx, None]
    value = value + correction/m.scale[:, None]
    return value[:, 0] if one else value


def _pattern(m):
    if m not in _GROUP_CACHE:
        pattern = m.jac_sparsity.tocoo()
        groups = group_columns(m.jac_sparsity)
        _GROUP_CACHE[m] = (pattern.row, pattern.col, groups, int(groups.max()+1))
    return _GROUP_CACHE[m]


def _rvl_data(m, scaled):
    T = scaled[m.sl['T']]*TM
    Tp = T[m.pidx]
    mv = np.maximum(scaled[m.sl['mv']]*100., 0.)
    ml = np.maximum(scaled[m.sl['ml']]*100., 0.)
    mi = np.maximum(scaled[m.sl['mi']]*100., 0.)
    eg = np.maximum(m.eps0-ml/990.-mi/920., 1e-12)
    ps, _ = saturation(Tp)
    ms = eg*MW*ps/(R*Tp)
    return mv, ml, ms


def _with_rvl_mask(m, t, y, j_c, q, ambient_C, active_condensation):
    """相变修正计算"""
    scaled = np.asarray(y)
    one = scaled.ndim == 1
    if one:
        scaled = scaled[:, None]
    mv, ml, ms = _rvl_data(m, scaled)
    actual = np.maximum(mv-ms, 0.)/m.cfg.tau_c - ml/m.cfg.tau_e*np.maximum(1.-mv/ms, 0.)
    frozen = np.where(active_condensation, (mv-ms)/m.cfg.tau_c,
                      -ml/m.cfg.tau_e*(1.-mv/ms))
    delta = frozen-actual


    value = np.asarray(m.rhs(t, scaled, j_c, q, ambient_C)).copy()
    value[m.sl['mv']] -= delta/100.
    value[m.sl['ml']] += delta/100.
    value[m.pidx] += LV*delta*m.dx_p*m.mesh.area_m2/m.mesh.capacity_J_K[m.pidx, None]/TM
    return value[:, 0] if one else value


# 雅可比计算
def active_jacobian(m, tt, y, current, q, ambient, *, branches='Rvl', batch_size=64):
    """稀疏矩阵计算"""
    state = np.asarray(y, dtype=float)
    j_c = float(current(tt) if callable(current) else current)
    power = q(tt) if callable(q) else q
    ambient_C = float(ambient(tt) if callable(ambient) else ambient)
    row, col, groups, count = _pattern(m)
    step = 1e-6*np.maximum(np.abs(state), 1e-6)
    temp = m.sl['T']
    step[temp] *= np.where(state[temp] < 1., -1., 1.)
    for name in ('mv', 'ml', 'mi'):
        sl = m.sl[name]
        step[sl] = 1e-6*np.maximum(np.abs(state[sl]), m.cfg.mstar/100*1e-3)
    if branches == 'Rvl':
        mv, _, ms = _rvl_data(m, state[:, None])
        masks = mv >= ms
        evaluate = _with_rvl_mask
    elif branches == 'all_phase':
        _, masks, _ = _phase(m, state, j_c)
        evaluate = _with_masks
    else:
        raise ValueError("branches must be 'Rvl' or 'all_phase'")
    base = evaluate(m, tt, state, j_c, power, ambient_C, masks)


    width=count if batch_size is None else max(1,int(batch_size))
    entries=np.empty(row.size,dtype=float)
    entry_groups=groups[col]
    for first in range(0,count,width):
        last=min(first+width,count)
        selected=np.flatnonzero((groups>=first)&(groups<last))
        perturbation=np.zeros((m.size,last-first))
        perturbation[selected,groups[selected]-first]=step[selected]
        delta=evaluate(m,tt,state[:,None]+perturbation,j_c,power,ambient_C,masks)-base[:,None]
        positions=np.flatnonzero((entry_groups>=first)&(entry_groups<last))
        entries[positions]=delta[row[positions],entry_groups[positions]-first]/step[col[positions]]
    return csr_matrix((entries, (row, col)), shape=(m.size, m.size))


def branch_diagnostics(m, y, current_density):
    """分支状态诊断"""
    d, z, _ = m.quantities(y, current_density)
    ps, pi = saturation(z['Tp'])
    ratio = np.maximum(d['mv'], 0)*R*z['Tp']/(MW*z['eg']*ps)
    return dict(min_abs_relative_liquid_saturation_distance=float(np.min(np.abs(ratio-1.))),
                liquid_saturation_ratio=ratio[:, 0], pore_x_m=m.mesh.x[m.pidx],
                pore_cell_id=m.pcell, liquid_inventory_kg_m3=d['ml'][:, 0])


def compact_jacobian(m,tt,y,current,q,ambient,*,branches='Rvl',batch_size=64):
    state=np.asarray(y,dtype=float)
    j_c=float(current(tt) if callable(current) else current)
    power=q(tt) if callable(q) else q
    ambient_C=float(ambient(tt) if callable(ambient) else ambient)
    row,col,groups,count=_pattern(m)
    step=1e-6*np.maximum(np.abs(state),1e-6)
    temp=m.sl['T'];step[temp]*=np.where(state[temp]<1.,-1.,1.)
    for name in ('mv','ml','mi'):
        sl=m.sl[name];step[sl]=1e-6*np.maximum(np.abs(state[sl]),m.cfg.mstar/100*1e-3)
    if branches=='Rvl':
        mv,_,ms=_rvl_data(m,state[:,None]);masks=mv>=ms;evaluate=_with_rvl_mask
    elif branches=='all_phase':
        _,masks,_=_phase(m,state,j_c);evaluate=_with_masks
    else:raise ValueError("branches must be 'Rvl' or 'all_phase'")
    base=evaluate(m,tt,state,j_c,power,ambient_C,masks)
    width=count if batch_size is None else max(1,int(batch_size))

    group_type=np.int16 if count<32767 else np.int32
    entry_groups=np.asarray(groups,dtype=group_type)[col]
    values_out=[];rows_out=[];cols_out=[]
    for first in range(0,count,width):
        last=min(first+width,count);selected=np.flatnonzero((groups>=first)&(groups<last))
        perturbation=np.zeros((m.size,last-first));perturbation[selected,groups[selected]-first]=step[selected]
        delta=evaluate(m,tt,state[:,None]+perturbation,j_c,power,ambient_C,masks)-base[:,None]
        positions=np.flatnonzero((entry_groups>=first)&(entry_groups<last))
        entries=delta[row[positions],entry_groups[positions]-first]/step[col[positions]]
        keep=entries!=0.
        values_out.append(entries[keep]);rows_out.append(row[positions[keep]]);cols_out.append(col[positions[keep]])
    return csr_matrix((np.concatenate(values_out),(np.concatenate(rows_out),np.concatenate(cols_out))),shape=(m.size,m.size))


phase_branch_jacobian = active_jacobian


def state_metrics(m,y,jc):
    d,z,_=m.quantities(y,jc)
    D={k:v[:,0] for k,v in d.items()};Z={k:v[:,0] for k,v in z.items()}
    T=D['T'];eps=m.eps0[:,0]-D['ml']/990-D['mi']/920
    lam=D['mb']/(m.phi[:,0]*CLAMBDA)
    ice=D['mi']/(920*m.eps0[:,0]);liquid=D['ml']/(990*m.eps0[:,0])
    mass=m.mesh.area_m2*(np.sum(D['mb']*m.dx_b[:,0])+np.sum((D['mv']+D['ml']+D['mi'])*m.dx_p[:,0]))
    H=np.sum(m.mesh.capacity_J_K*(T-TM))+m.mesh.area_m2*np.sum((LV*D['mv']-LF*D['mi'])*m.dx_p[:,0])
    wgen=m.nc*m.cfg.area_cm2*D['charge'][0]*MW/(2*F)
    out=dict(temperature_C=m.mesh.mea_means(T)-TM,voltage_V=Z['V'],lambda_min=float(lam[m.bk=='PEM'].min()),
      ionomer_lambda_min=float(lam.min()),
      lambda_max=float(lam.max()),porosity_ratio_min=float(np.min(eps/m.eps0[:,0])),
      inventory_min=float(min(D[k].min() for k in ['mb','mv','ml','mi'])),gas_min=float(D['ng'].min()),
      gas_min_ratio=float(np.min(D['ng']/(np.maximum(eps,1e-12)*m.gas_y[:,0]*PREF/(R*Z['Tp'])))),
      ice_max=float(ice.max()),ice_volume_max=float((D['mi']/920).max()),liquid_max=float(liquid.max()),
      ice_by_cell=np.array([ice[m.pcell==c].max() for c in range(m.nc)]),
      lambda_by_cell=np.array([np.mean(lam[m.bcell==c]) for c in range(m.nc)]),
      aux_energy_J=float(D['aux'].sum()),aux_energy_cells_J=D['aux'],rxn_energy_J=float(D['rxn'].sum()),
      electrical_output_J=float(D['elec'].sum()),charge_C_cm2=float(D['charge'][0]),
      conv_energy_J=float(D['conv'][0]),water_out_kg=float(D['wout'][0]),
      water_total_kg=float(mass),water_generated_kg=float(wgen),enthalpy_J=float(H),
      activation_V=Z['act'],ohmic_V=Z['ohm'],concentration_V=Z['con'])
    pgas=Z['c']*R*Z['Tp'];pv=np.maximum(D['mv'],0)*R*Z['Tp']/(MW*np.maximum(eps,1e-12))
    out['anode_pressure_deviation']=float(np.max(np.abs((pgas[m.an]+pv[m.an])/PREF-1)))
    return out


def domain_failure(m,met):
    if not np.isfinite(met['voltage_V']).all():return 'NONFINITE_VOLTAGE'
    if met['inventory_min'] < -1e-9:return 'NEGATIVE_INVENTORY'
    if met['porosity_ratio_min'] <= m.cfg.delta_g:return 'PORE_BLOCKAGE'
    if met['gas_min_ratio']<=m.cfg.delta_c:return 'REACTANT_DEPLETION'
    if met['lambda_min'] <= .326/.5139:return 'MEMBRANE_DRYOUT'
    if m.cfg.cl_resistance_factor and met['ionomer_lambda_min']<=.326/.5139:return 'CL_IONOMER_DRYOUT'
    if met['lambda_max']>=26.1119:return 'DIFFUSIVITY_DOMAIN'
    return ''


# 状态积分求解
def simulate(m,t_end,initial_C,ambient_C,current,heating=None,*,sample_dt=.2,breakpoints=(),
             stop_success=False,voltage_floor=None,charge_cap=None,controller=None,initial_state=None,
             save_fields=False,progress=False,controller_dt=.2,max_wall_s=90.,reuse_jacobian=False):
    start=time.perf_counter();y0=m.initial(initial_C) if initial_state is None else np.array(initial_state,copy=True)
    y=y0.copy();t=0.;q=np.zeros(m.nc) if heating is None else np.broadcast_to(heating(0.),(m.nc,)).copy()
    base=state_metrics(m,y,current(0));samples=[];ys=[];status='COMPLETE';failure='';steps=0;restarts=0
    minV=float(np.min(base['voltage_V']));minlam=base['lambda_min'];mineps=base['porosity_ratio_min'];maxice=base['ice_max']
    maxgaserr=base['anode_pressure_deviation'];maxwm=0.;maxeh=0.;lastprint=start
    source_scale=max(base['water_total_kg'],1e-12)
    knots=np.unique(np.r_[0.,np.asarray(breakpoints),t_end]);knots=knots[(knots>=0)&(knots<=t_end)]
    if controller_dt<=0:raise ValueError('controller_dt must be positive')
    if controller is not None:
        knots=np.unique(np.r_[knots,np.arange(0,t_end+controller_dt*.5,controller_dt)])
        knots=knots[(knots>=0)&(knots<=t_end)]
    targets=np.arange(0,t_end+sample_dt*.5,sample_dt);targets=targets[targets<=t_end+1e-9];nextsample=0
    def annotate(tt,yy,qq):
        nonlocal maxwm,maxeh
        met=state_metrics(m,yy,current(tt));met['time_s']=float(tt);met['j_A_cm2']=float(current(tt));met['q_W_cm2']=np.array(qq)
        rw=met['water_total_kg']-base['water_total_kg']+met['water_out_kg']-met['water_generated_kg']
        re=met['enthalpy_J']-base['enthalpy_J']+met['conv_energy_J']+LV*met['water_out_kg']-met['rxn_energy_J']-met['aux_energy_J']
        mw=abs(rw)/(source_scale+abs(met['water_generated_kg'])+abs(met['water_out_kg']))
        eh=abs(re)/(1+abs(met['enthalpy_J']-base['enthalpy_J'])+abs(met['conv_energy_J'])+abs(LV*met['water_out_kg'])+abs(met['rxn_energy_J'])+met['aux_energy_J'])
        met['water_residual_relative']=mw;met['energy_residual_relative']=eh
        maxwm=max(maxwm,mw);maxeh=max(maxeh,eh)
        return met
    def margin(met):return float(np.min(met['temperature_C'])-m.cfg.success_margin_K)
    fail0=domain_failure(m,base)
    if fail0:status='DOMAIN_FAILURE';failure=fail0
    if voltage_floor is not None and minV<voltage_floor:status='SAFETY_FAILURE';failure='VOLTAGE'
    if status=='COMPLETE' and stop_success and margin(base)>=0:
        status='SUCCESS';q=np.zeros(m.nc)
    prev=base;Jcache=None
    pattern=m.jac_sparsity.tocoo();groups=group_columns(m.jac_sparsity)
    ngroups=groups.max()+1
    def jacobian(tt,yy):
        return phase_branch_jacobian(m,tt,yy,current,q,ambient_C)
    def legacy_jacobian(tt,yy):


        step=1e-6*np.maximum(np.abs(yy),1e-6)


        ti=m.sl['T'];step[ti]*=np.where(yy[ti]<1.,-1.,1.)
        for name in ('mv','ml','mi'):
            ss=m.sl[name];step[ss]=1e-6*np.maximum(np.abs(yy[ss]),m.cfg.mstar/100*1e-3)
        pert=np.zeros((m.size,ngroups));pert[np.arange(m.size),groups]=step
        df=fun(tt,yy[:,None]+pert)-fun(tt,yy)[:,None]
        values=df[pattern.row,groups[pattern.col]]/step[pattern.col]
        return csr_matrix((values,(pattern.row,pattern.col)),shape=(m.size,m.size))
    for bound in knots[1:]:
        if status!='COMPLETE':break
        if controller is not None:
            if abs(t/controller_dt-round(t/controller_dt))<1e-7:
                q=np.asarray(controller(t,prev),float)
        elif heating is not None:q=np.broadcast_to(heating(t+1e-10),(m.nc,)).copy()
        if np.any(q<0) or np.any(q>1):raise ValueError('Heating outside [0,1]')
        fun=lambda tt,yy:m.rhs(tt,yy,current(tt),q,ambient_C)


        first_jac_call=True
        def interval_jacobian(tt,yy):
            nonlocal first_jac_call
            if first_jac_call:
                first_jac_call=False
                if reuse_jacobian and Jcache is not None:
                    return Jcache
            return jacobian(tt,yy)
        solver=BDF(fun,t,y,float(bound),rtol=m.cfg.rtol,atol=m.atol,max_step=min(.01 if t<.1 else m.cfg.max_step,bound-t),
                   jac_sparsity=m.jac_sparsity,vectorized=True,jac=interval_jacobian)
        local_retries=0
        recent_times=[]
        while solver.status=='running':
            if max_wall_s is not None and time.perf_counter()-start>max_wall_s:
                status='NUMERICAL_LIMIT';failure='numerical_wall_time_limit; requires refinement or longer numerical budget';break
            if steps>=20000:
                status='NUMERICAL_LIMIT';failure='20000 accepted steps; requires numerical refinement';break
            oldt=solver.t;oldy=solver.y.copy()
            msg=solver.step();steps+=1
            if solver.status=='failed':status='NUMERICAL_FAILURE';failure=str(msg);break
            yy=solver.y;tt=solver.t
            recent_times.append(tt)
            if len(recent_times)>256:
                recent_times.pop(0)
                if tt>.1 and tt-recent_times[0]<1e-7:
                    status='NUMERICAL_LIMIT';failure='numerical_stagnation: 256 steps advance less than 1e-7 s';break
            met=state_metrics(m,yy,current(tt));df=domain_failure(m,met)
            if df=='NEGATIVE_INVENTORY' and local_retries<20:
                local_retries+=1;restarts+=1
                solver=BDF(fun,oldt,oldy,float(bound),rtol=m.cfg.rtol*.5**min(local_retries,4),atol=m.atol*.5**min(local_retries,4),
                           max_step=max((tt-oldt)*.5,1e-14),jac_sparsity=m.jac_sparsity,vectorized=True,jac=jacobian)
                continue
            dense=solver.dense_output()

            boundaries={
                'NEGATIVE_INVENTORY':lambda mm:mm['inventory_min']+1e-9,
                'MEMBRANE_DRYOUT':lambda mm:mm['lambda_min']-.326/.5139,
                'PORE_BLOCKAGE':lambda mm:mm['porosity_ratio_min']-m.cfg.delta_g,
                'REACTANT_DEPLETION':lambda mm:mm['gas_min_ratio']-m.cfg.delta_c,
                'DIFFUSIVITY_DOMAIN':lambda mm:26.1119-mm['lambda_max']}
            if m.cfg.cl_resistance_factor:
                boundaries['CL_IONOMER_DRYOUT']=lambda mm:mm['ionomer_lambda_min']-.326/.5139
            if voltage_floor is not None:boundaries['VOLTAGE']=lambda mm:np.min(mm['voltage_V'])-voltage_floor
            if charge_cap is not None:boundaries['CHARGE']=lambda mm:charge_cap-mm['charge_C_cm2']
            events=[]
            for reason,eventfun in boundaries.items():
                if eventfun(met)<0:
                    try:
                        et=brentq(lambda s:eventfun(state_metrics(m,dense(s),current(s))),oldt,tt,xtol=1e-8)
                    except ValueError:et=oldt
                    events.append((et,0,reason))
            if df=='NONFINITE_VOLTAGE':events.append((oldt,0,df))
            if stop_success and margin(met)>=0:
                et=brentq(lambda s:margin(state_metrics(m,dense(s),current(s))),oldt,tt,xtol=1e-8) if margin(prev)<0 else oldt
                events.append((et,1,'SUCCESS'))
            if events:
                tt,_,reason=min(events)
                yy=dense(tt);met=state_metrics(m,yy,current(tt))
                status='SUCCESS' if reason=='SUCCESS' else ('SAFETY_FAILURE' if reason in ('VOLTAGE','CHARGE') else 'DOMAIN_FAILURE')
                failure='' if reason=='SUCCESS' else reason
            while nextsample<len(targets) and targets[nextsample]<=tt+1e-10:
                ts=float(targets[nextsample]);sv=y0 if ts==0 else dense(ts)
                samples.append(annotate(ts,sv,q));
                if save_fields:ys.append(sv.copy())
                nextsample+=1
            minV=min(minV,float(np.min(met['voltage_V'])));minlam=min(minlam,met['lambda_min']);mineps=min(mineps,met['porosity_ratio_min']);maxice=max(maxice,met['ice_max'])
            maxgaserr=max(maxgaserr,met['anode_pressure_deviation']);prev=met;t=tt;y=yy.copy()
            if progress and time.perf_counter()-lastprint>30:
                print(f'integrate t={t:.2f}/{t_end:.1f} s Vmin={minV:.4f} T={met["temperature_C"]}',flush=True);lastprint=time.perf_counter()
            if status!='COMPLETE':break
            solver.max_step=.01 if t<.1 else m.cfg.max_step
        Jcache=solver.J
    final=annotate(t,y,q)
    if not samples or abs(samples[-1]['time_s']-t)>1e-8:
        samples.append(final)
        if save_fields:ys.append(y.copy())
    result={'status':status,'failure_reason':failure,'time_s':float(t),'requested_time_s':float(t_end),
      'wall_s':time.perf_counter()-start,'steps':steps,'restarts':restarts,'config':m.cfg.to_dict(),
      'min_voltage_V':minV,'min_lambda':minlam,'min_porosity_ratio':mineps,'max_ice_saturation':maxice,
      'max_anode_pressure_deviation':maxgaserr,'gas_approximation_warning':maxgaserr>m.cfg.gas_tolerance,
      'reuse_previous_interval_jacobian':bool(reuse_jacobian),
      'water_residual_relative':maxwm,'energy_residual_relative':maxeh,
      **{k:final[k] for k in ['aux_energy_J','aux_energy_cells_J','rxn_energy_J','electrical_output_J','charge_C_cm2']},
      'final_temperature_C':final['temperature_C'],
      'hydrogen_consumed_mol':m.nc*m.cfg.area_cm2*final['charge_C_cm2']/(2*F),
      'hydrogen_thermoneutral_energy_J':m.nc*m.cfg.area_cm2*final['charge_C_cm2']*1.48,
      'series':{k:np.array([s[k] for s in samples]) for k in ['time_s','j_A_cm2','q_W_cm2','temperature_C','voltage_V',
        'lambda_by_cell','ice_by_cell','aux_energy_J','charge_C_cm2','activation_V','ohmic_V','concentration_V']}}
    if save_fields:result['states_scaled']=np.array(ys);result['state_scale']=m.scale;result['x_m']=m.mesh.x
    return result,y


# 反馈积分求解
def simulate_scheduled(m,t_end,initial_C,ambient_C,current,heating=None,*,sample_dt=.2,breakpoints=(),
             stop_success=False,voltage_floor=None,charge_cap=None,controller=None,initial_state=None,
             save_fields=False,progress=False,controller_dt=.2,max_wall_s=90.,reuse_jacobian=False,controller_stop_s=None):
    start=time.perf_counter();y0=m.initial(initial_C) if initial_state is None else np.array(initial_state,copy=True)
    y=y0.copy();t=0.;q=np.zeros(m.nc) if heating is None else np.broadcast_to(heating(0.),(m.nc,)).copy()
    base=state_metrics(m,y,current(0));samples=[];ys=[];status='COMPLETE';failure='';steps=0;restarts=0
    minV=float(np.min(base['voltage_V']));minlam=base['lambda_min'];mineps=base['porosity_ratio_min'];maxice=base['ice_max']
    maxgaserr=base['anode_pressure_deviation'];maxwm=0.;maxeh=0.;lastprint=start
    source_scale=max(base['water_total_kg'],1e-12)
    knots=np.unique(np.r_[0.,np.asarray(breakpoints),t_end]);knots=knots[(knots>=0)&(knots<=t_end)]
    if controller_dt<=0:raise ValueError('controller_dt must be positive')
    if controller is not None:
        knots=np.unique(np.r_[knots,np.arange(0,(t_end if controller_stop_s is None else min(t_end,controller_stop_s))+controller_dt*.5,controller_dt)])
        knots=knots[(knots>=0)&(knots<=t_end)]
    targets=np.arange(0,t_end+sample_dt*.5,sample_dt);targets=targets[targets<=t_end+1e-9];nextsample=0
    def annotate(tt,yy,qq):
        nonlocal maxwm,maxeh
        met=state_metrics(m,yy,current(tt));met['time_s']=float(tt);met['j_A_cm2']=float(current(tt));met['q_W_cm2']=np.array(qq)
        rw=met['water_total_kg']-base['water_total_kg']+met['water_out_kg']-met['water_generated_kg']
        re=met['enthalpy_J']-base['enthalpy_J']+met['conv_energy_J']+LV*met['water_out_kg']-met['rxn_energy_J']-met['aux_energy_J']
        mw=abs(rw)/(source_scale+abs(met['water_generated_kg'])+abs(met['water_out_kg']))
        eh=abs(re)/(1+abs(met['enthalpy_J']-base['enthalpy_J'])+abs(met['conv_energy_J'])+abs(LV*met['water_out_kg'])+abs(met['rxn_energy_J'])+met['aux_energy_J'])
        met['water_residual_relative']=mw;met['energy_residual_relative']=eh
        maxwm=max(maxwm,mw);maxeh=max(maxeh,eh)
        return met
    def margin(met):return float(np.min(met['temperature_C'])-m.cfg.success_margin_K)
    fail0=domain_failure(m,base)
    if fail0:status='DOMAIN_FAILURE';failure=fail0
    if voltage_floor is not None and minV<voltage_floor:status='SAFETY_FAILURE';failure='VOLTAGE'
    if status=='COMPLETE' and stop_success and margin(base)>=0:
        status='SUCCESS';q=np.zeros(m.nc)
    prev=base;Jcache=None
    pattern=m.jac_sparsity.tocoo();groups=group_columns(m.jac_sparsity)
    ngroups=groups.max()+1
    def jacobian(tt,yy):
        return phase_branch_jacobian(m,tt,yy,current,q,ambient_C)
    def legacy_jacobian(tt,yy):


        step=1e-6*np.maximum(np.abs(yy),1e-6)


        ti=m.sl['T'];step[ti]*=np.where(yy[ti]<1.,-1.,1.)
        for name in ('mv','ml','mi'):
            ss=m.sl[name];step[ss]=1e-6*np.maximum(np.abs(yy[ss]),m.cfg.mstar/100*1e-3)
        pert=np.zeros((m.size,ngroups));pert[np.arange(m.size),groups]=step
        df=fun(tt,yy[:,None]+pert)-fun(tt,yy)[:,None]
        values=df[pattern.row,groups[pattern.col]]/step[pattern.col]
        return csr_matrix((values,(pattern.row,pattern.col)),shape=(m.size,m.size))
    for bound in knots[1:]:
        if status!='COMPLETE':break
        if controller is not None:
            if abs(t/controller_dt-round(t/controller_dt))<1e-7:
                q=np.asarray(controller(t,prev),float)
        elif heating is not None:q=np.broadcast_to(heating(t+1e-10),(m.nc,)).copy()
        if np.any(q<0) or np.any(q>1):raise ValueError('Heating outside [0,1]')
        fun=lambda tt,yy:m.rhs(tt,yy,current(tt),q,ambient_C)


        first_jac_call=True
        def interval_jacobian(tt,yy):
            nonlocal first_jac_call
            if first_jac_call:
                first_jac_call=False
                if reuse_jacobian and Jcache is not None:
                    return Jcache
            return jacobian(tt,yy)
        solver=BDF(fun,t,y,float(bound),rtol=m.cfg.rtol,atol=m.atol,max_step=min(.01 if t<.1 else m.cfg.max_step,bound-t),
                   jac_sparsity=m.jac_sparsity,vectorized=True,jac=interval_jacobian)
        local_retries=0
        recent_times=[]
        while solver.status=='running':
            if max_wall_s is not None and time.perf_counter()-start>max_wall_s:
                status='NUMERICAL_LIMIT';failure='numerical_wall_time_limit; requires refinement or longer numerical budget';break
            if steps>=20000:
                status='NUMERICAL_LIMIT';failure='20000 accepted steps; requires numerical refinement';break
            oldt=solver.t;oldy=solver.y.copy()
            msg=solver.step();steps+=1
            if solver.status=='failed':status='NUMERICAL_FAILURE';failure=str(msg);break
            yy=solver.y;tt=solver.t
            recent_times.append(tt)
            if len(recent_times)>256:
                recent_times.pop(0)
                if tt>.1 and tt-recent_times[0]<1e-7:
                    status='NUMERICAL_LIMIT';failure='numerical_stagnation: 256 steps advance less than 1e-7 s';break
            met=state_metrics(m,yy,current(tt));df=domain_failure(m,met)
            if df=='NEGATIVE_INVENTORY' and local_retries<20:
                local_retries+=1;restarts+=1
                solver=BDF(fun,oldt,oldy,float(bound),rtol=m.cfg.rtol*.5**min(local_retries,4),atol=m.atol*.5**min(local_retries,4),
                           max_step=max((tt-oldt)*.5,1e-14),jac_sparsity=m.jac_sparsity,vectorized=True,jac=jacobian)
                continue
            dense=solver.dense_output()

            boundaries={
                'NEGATIVE_INVENTORY':lambda mm:mm['inventory_min']+1e-9,
                'MEMBRANE_DRYOUT':lambda mm:mm['lambda_min']-.326/.5139,
                'PORE_BLOCKAGE':lambda mm:mm['porosity_ratio_min']-m.cfg.delta_g,
                'REACTANT_DEPLETION':lambda mm:mm['gas_min_ratio']-m.cfg.delta_c,
                'DIFFUSIVITY_DOMAIN':lambda mm:26.1119-mm['lambda_max']}
            if m.cfg.cl_resistance_factor:
                boundaries['CL_IONOMER_DRYOUT']=lambda mm:mm['ionomer_lambda_min']-.326/.5139
            if voltage_floor is not None:boundaries['VOLTAGE']=lambda mm:np.min(mm['voltage_V'])-voltage_floor
            if charge_cap is not None:boundaries['CHARGE']=lambda mm:charge_cap-mm['charge_C_cm2']
            events=[]
            for reason,eventfun in boundaries.items():
                if eventfun(met)<0:
                    try:
                        et=brentq(lambda s:eventfun(state_metrics(m,dense(s),current(s))),oldt,tt,xtol=1e-8)
                    except ValueError:et=oldt
                    events.append((et,0,reason))
            if df=='NONFINITE_VOLTAGE':events.append((oldt,0,df))
            if stop_success and margin(met)>=0:
                et=brentq(lambda s:margin(state_metrics(m,dense(s),current(s))),oldt,tt,xtol=1e-8) if margin(prev)<0 else oldt
                events.append((et,1,'SUCCESS'))
            if events:
                tt,_,reason=min(events)
                yy=dense(tt);met=state_metrics(m,yy,current(tt))
                status='SUCCESS' if reason=='SUCCESS' else ('SAFETY_FAILURE' if reason in ('VOLTAGE','CHARGE') else 'DOMAIN_FAILURE')
                failure='' if reason=='SUCCESS' else reason
            while nextsample<len(targets) and targets[nextsample]<=tt+1e-10:
                ts=float(targets[nextsample]);sv=y0 if ts==0 else dense(ts)
                samples.append(annotate(ts,sv,q));
                if save_fields:ys.append(sv.copy())
                nextsample+=1
            minV=min(minV,float(np.min(met['voltage_V'])));minlam=min(minlam,met['lambda_min']);mineps=min(mineps,met['porosity_ratio_min']);maxice=max(maxice,met['ice_max'])
            maxgaserr=max(maxgaserr,met['anode_pressure_deviation']);prev=met;t=tt;y=yy.copy()
            if progress and time.perf_counter()-lastprint>30:
                print(f'integrate t={t:.2f}/{t_end:.1f} s Vmin={minV:.4f} T={met["temperature_C"]}',flush=True);lastprint=time.perf_counter()
            if status!='COMPLETE':break
            solver.max_step=.01 if t<.1 else m.cfg.max_step
        Jcache=solver.J
    final=annotate(t,y,q)
    if not samples or abs(samples[-1]['time_s']-t)>1e-8:
        samples.append(final)
        if save_fields:ys.append(y.copy())
    result={'status':status,'failure_reason':failure,'time_s':float(t),'requested_time_s':float(t_end),
      'wall_s':time.perf_counter()-start,'steps':steps,'restarts':restarts,'config':m.cfg.to_dict(),
      'min_voltage_V':minV,'min_lambda':minlam,'min_porosity_ratio':mineps,'max_ice_saturation':maxice,
      'max_anode_pressure_deviation':maxgaserr,'gas_approximation_warning':maxgaserr>m.cfg.gas_tolerance,
      'reuse_previous_interval_jacobian':bool(reuse_jacobian),
      'water_residual_relative':maxwm,'energy_residual_relative':maxeh,
      **{k:final[k] for k in ['aux_energy_J','aux_energy_cells_J','rxn_energy_J','electrical_output_J','charge_C_cm2']},
      'final_temperature_C':final['temperature_C'],
      'hydrogen_consumed_mol':m.nc*m.cfg.area_cm2*final['charge_C_cm2']/(2*F),
      'hydrogen_thermoneutral_energy_J':m.nc*m.cfg.area_cm2*final['charge_C_cm2']*1.48,
      'series':{k:np.array([s[k] for s in samples]) for k in ['time_s','j_A_cm2','q_W_cm2','temperature_C','voltage_V',
        'lambda_by_cell','ice_by_cell','aux_energy_J','charge_C_cm2','activation_V','ohmic_V','concentration_V']}}
    if save_fields:result['states_scaled']=np.array(ys);result['state_scale']=m.scale;result['x_m']=m.mesh.x
    return result,y


# 约束条件核查
def validate_constraints(result,require_success=True,charge_cap=None):
    """约束结果核查"""
    errors=[]
    if result['status'] not in ('SUCCESS','COMPLETE'):
        errors.append(result['status'])
    if require_success and result['status']!='SUCCESS': errors.append('success_not_reached')
    if result.get('min_voltage_V',1)<.30-1e-6: errors.append('voltage_below_0.30')
    if result.get('min_porosity_ratio',1)<=1e-6: errors.append('pore_domain')
    if result.get('min_lambda',3)<=.326/.5139: errors.append('conductivity_domain')
    if result.get('water_residual_relative',0)>1e-5: errors.append('water_balance')
    if result.get('energy_residual_relative',0)>1e-4: errors.append('energy_balance')
    if charge_cap is not None and result.get('charge_C_cm2',0)>charge_cap+1e-6: errors.append('charge_budget')
    if result.get('aux_energy_J',0)<-1e-8: errors.append('negative_energy')
    return {'passed':not errors,'violations':errors}
