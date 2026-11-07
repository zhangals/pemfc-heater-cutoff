"""Recover auditable post-hoc single-temperature fitting and cross-temperature prediction."""
from pathlib import Path
import sys
import argparse
import json
import time
from dataclasses import replace

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'research/study_v2'))
import engine as eng
import numpy as np
import openpyxl
from scipy.optimize import least_squares
from 模型物理 import Config,FROZEN_CONFIG,FuelCellModel,build_mesh

NAMES=['gamma0','cl_resistance_factor','hydration_exponent','tau_lb','activation_energy','gamma_f','gamma_Db']
LOW=np.array([.1,.01,0.,.2,20000.,.001,.5])
HIGH=np.array([150.,10.,5.,200.,90000.,1.,5.])
PAPER=np.array([40.6048,2.9296,3.2382,199.,73190.,.00101,2.])
LOG=np.array([True,True,False,True,False,True,True])
def transform(p):return np.where(LOG,np.log(np.maximum(p,1e-99)),p)
LO,HI=transform(LOW),transform(HIGH)
def encode(p):return (transform(p)-LO)/(HI-LO)
def decode(x):
    z=LO+(HI-LO)*x
    return np.where(LOG,np.exp(np.minimum(z,700)),z)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def new(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=eng.encode,allow_nan=False)
def simulate(p,data,refine=2,convention='recorded',fields=False):
    cfg=replace(Config(**FROZEN_CONFIG),**dict(zip(NAMES,map(float,p))),rtol=1e-7,max_step=.05)
    m=FuelCellModel(1,refine=refine,endplates=False,config=cfg,backend='numba',mesh_builder=build_mesh)
    assert convention=='recorded'
    ts=data[:,1];j=data[:,5]
    current=lambda t:float(np.interp(t,ts,j))
    raw,final=eng.solver.simulate(m,36.6,float(data[0,4]),float(data[0,0]),current,
      sample_dt=.2,voltage_floor=.2,save_fields=fields,max_wall_s=90)
    complete=raw['status']=='COMPLETE' and raw['time_s']>=36.6-1e-8
    predicted=None;residual=None;metric=None
    if complete:
        st=raw['series']['time_s']
        v=np.interp(ts,st,raw['series']['voltage_V'][:,0]);t=np.interp(ts,st,raw['series']['temperature_C'][:,0])
        predicted=np.column_stack([v,t]);dv=v-data[:,3];dt=t-data[:,4]
        residual=np.r_[dv/.02,dt/.5]/np.sqrt(len(ts))
        metric={'voltage_rmse_V':float(np.sqrt(np.mean(dv**2))),'temperature_rmse_K':float(np.sqrt(np.mean(dt**2))),
          'voltage_mae_V':float(np.mean(abs(dv))),'temperature_mae_K':float(np.mean(abs(dt))),'loss':float(residual@residual)}
    meta={k:raw[k] for k in ['status','failure_reason','time_s','wall_s','steps','restarts','water_residual_relative','energy_residual_relative']}
    meta.update(parameters=dict(zip(NAMES,map(float,p))),metrics=metric,complete=complete,refine=refine,convention=convention)
    return residual,predicted,meta,raw,final,m
