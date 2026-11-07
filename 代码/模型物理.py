# 共享物理模块
"""物理模型计算"""


from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import os

import numpy as np
from numba import njit
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import lil_matrix, csr_matrix


# 参数配置定义
@dataclass
class Config:
    area_cm2: float = 25.0
    h: float = 40.0
    lambda0: float = 3.0
    gamma0: float = 24.931749774684768
    gamma_R: float = 1.0
    gamma_f: float = 0.0016261408980128213
    gamma_Db: float = 1.77809333372637
    gamma_drag: float = 1.0
    exponential_water_flux: bool = False
    cl_resistance_factor: float = 2.5989647664491833
    hydration_exponent: float = 2.6331129534450834
    activation_energy: float = 64901.4784121162
    alpha: float = 0.5
    tau_ads: float = 1.0
    tau_des: float = 1000.0
    tau_lb: float = 178.23618258809626
    tau_lb_des: float = 2.0
    tau_c: float = 1e-4
    tau_e: float = 1e-4
    tau_f: float = 1.0
    tau_m: float = 1.0
    tau_di: float = 1e4
    tau_su: float = 1e4
    mstar: float = 1e-8
    delta_phase_K: float = 1.0
    delta_g: float = 1e-6
    delta_c: float = 1e-6
    gas_tolerance: float = .01
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = .1
    success_margin_K: float = .01
    source_hypothesis: str = 'H303_TRANSFER_CONDITIONAL'
    def to_dict(self):
        return asdict(self)

F=96485.0
R=8.314
MW=.018
CLAMBDA=38.7
PREF=101325.0
TM=273.15
LV=2.5e6
LF=333600.0
YO=(.233/.032)/(.233/.032+.767/.028)


MATERIAL_CELLS = {'C3': '0.01',
 'C4': 0.002,
 'C5': 0.002,
 'C6': 0.00015,
 'C7': 0.00015,
 'C8': 3.4e-06,
 'C9': 1.13e-05,
 'C10': 1.2e-05,
 'C36': '2150; 1050; 0.24',
 'C37': '1980; 766; 95; 83000',
 'C38': '185; 545; 0.3; 375',
 'C39': '970; 240; 0.27',
 'C40': '7900',
 'C41': '500\xa0',
 'C42': '15'}


FROZEN_CONFIG = {'area_cm2': 25.0,
 'h': 40.0,
 'lambda0': 3.0,
 'gamma0': 24.931749774684768,
 'gamma_R': 1.0,
 'gamma_f': 0.0016261408980128213,
 'gamma_Db': 1.77809333372637,
 'gamma_drag': 1.0,
 'exponential_water_flux': False,
 'cl_resistance_factor': 2.5989647664491833,
 'hydration_exponent': 2.6331129534450834,
 'activation_energy': 64901.4784121162,
 'alpha': 0.5,
 'tau_ads': 1.0,
 'tau_des': 1000.0,
 'tau_lb': 178.23618258809626,
 'tau_lb_des': 2.0,
 'tau_c': 0.0001,
 'tau_e': 0.0001,
 'tau_f': 1.0,
 'tau_m': 1.0,
 'tau_di': 10000.0,
 'tau_su': 10000.0,
 'mstar': 1e-08,
 'delta_phase_K': 1.0,
 'delta_g': 1e-06,
 'delta_c': 1e-06,
 'gas_tolerance': 0.01,
 'rtol': 1e-06,
 'atol': 1e-09,
 'max_step': 0.1,
 'success_margin_K': 0.01,
 'source_hypothesis': 'H303_TRANSFER_CONDITIONAL'}

def calibrated_config() -> Config:
    return Config(**FROZEN_CONFIG)


# 热网格计算
@dataclass
class ThermalMesh:
    n_cells: int
    refine: int
    area_cm2: float
    h: float
    dx: np.ndarray
    rhocp: np.ndarray
    k: np.ndarray
    kind: np.ndarray
    cell_ids: np.ndarray
    edges: np.ndarray
    x: np.ndarray
    capacity_J_K: np.ndarray
    conductance_W_K: np.ndarray
    boundary_conductance_W_K: np.ndarray
    K_W_K: sparse.csr_matrix
    A_s_inv: sparse.csr_matrix
    mea_indices: tuple[np.ndarray, ...]
    ccl_indices: tuple[np.ndarray, ...]
    bp_indices: tuple[np.ndarray, ...]
    parameter_sources: dict

    @property
    def area_m2(self):
        return self.area_cm2 * 1e-4

    @property
    def size(self):
        return self.dx.size

    def rhs(self, T, ambient_K, power_W=None):
        """热量变化计算"""
        temperature = np.asarray(T, dtype=float)
        net_power = -self.K_W_K @ (temperature - ambient_K)
        if power_W is not None:
            net_power += np.asarray(power_W, dtype=float)
        return net_power / self.capacity_J_K

    def mea_means(self, T):
        """温度均值计算"""
        temperature = np.asarray(T, dtype=float)
        return np.stack([
            np.sum(temperature[..., ind] * self.dx[ind], axis=-1)
            / self.dx[ind].sum() for ind in self.mea_indices
        ], axis=-1)

    def source_from_cell_power(self, P, where="BP"):
        """热源功率分配"""
        power = np.broadcast_to(np.asarray(P, dtype=float), (self.n_cells,))
        if where == "BP":
            selections = self.bp_indices
        elif where == "CCL":
            selections = self.ccl_indices
        else:
            raise ValueError("where must be 'BP' or 'CCL'")
        result = np.zeros(self.size)
        for value, ind in zip(power, selections):
            result[ind] = value * self.dx[ind] / self.dx[ind].sum()
        return result

    def outward_heat_W(self, T, ambient_K):
        temperature = np.asarray(T, dtype=float)
        return np.sum((temperature[..., [0, -1]] - ambient_K)
                      * self.boundary_conductance_W_K, axis=-1)


def _official_parameters():
    raw = dict(MATERIAL_CELLS)
    def scalar(cell):
        return float(str(raw[cell]).strip())
    def triple(cell):
        return tuple(float(s.strip()) for s in str(raw[cell]).split(";")[:3])
    mats = {"EP": (scalar("C40"), scalar("C41"), scalar("C42")),
            "BP": triple("C37"), "GDL": triple("C38"),
            "CL": triple("C39"), "PEM": triple("C36")}
    layers = [("BP", scalar("C4"), 2, "BP"),
              ("AGDL", scalar("C6"), 4, "GDL"),
              ("ACL", scalar("C8"), 2, "CL"),
              ("PEM", scalar("C10"), 4, "PEM"),
              ("CCL", scalar("C9"), 2, "CL"),
              ("CGDL", scalar("C7"), 4, "GDL"),
              ("BP", scalar("C5"), 2, "BP")]
    return mats, layers, scalar("C3"), raw


def build_mesh(n_cells=5, refine=1, endplates=True, area_cm2=25., h=40.):
    """电堆网格构造"""
    if int(n_cells) != n_cells or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    if int(refine) != refine or refine < 1 or area_cm2 <= 0 or h < 0:
        raise ValueError("Invalid refinement, area, or heat transfer coefficient")
    mats, layers, ep_length, raw = _official_parameters()
    dx, rhocp, conductivity, kinds, cells = [], [], [], [], []
    def add(kind, length, count, material, cell):
        rho, cp, conduct = mats[material]
        count *= refine
        dx.extend([length / count] * count)
        rhocp.extend([rho * cp] * count)
        conductivity.extend([conduct] * count)
        kinds.extend([kind] * count)
        cells.extend([cell] * count)
    if endplates:
        add("EP", ep_length, 8, "EP", -1)
    for cell in range(n_cells):
        for kind, length, count, material in layers:
            add(kind, length, count, material, cell)
    if endplates:
        add("EP", ep_length, 8, "EP", -1)
    dx, rhocp, conductivity = map(np.asarray, (dx, rhocp, conductivity))
    kinds, cells = np.asarray(kinds), np.asarray(cells)
    area = area_cm2 * 1e-4
    capacity = rhocp * dx * area
    g = area / (dx[:-1] / (2 * conductivity[:-1])
                + dx[1:] / (2 * conductivity[1:]))

    if h == 0:
        gb = np.zeros(2)
    else:
        gb = area / (1 / h + dx[[0, -1]] / (2 * conductivity[[0, -1]]))
    diagonal = np.r_[g, 0.] + np.r_[0., g]
    diagonal[[0, -1]] += gb
    stiffness = sparse.diags((-g, diagonal, -g), (-1, 0, 1), format="csr")
    dynamics = -sparse.diags(1 / capacity) @ stiffness
    edges = np.r_[0., np.cumsum(dx)]
    selections = lambda kinds_selected: tuple(
        np.flatnonzero((cells == c) & np.isin(kinds, kinds_selected))
        for c in range(n_cells))
    return ThermalMesh(
        n_cells, refine, float(area_cm2), float(h), dx, rhocp, conductivity,
        kinds, cells, edges, (edges[:-1] + edges[1:]) / 2, capacity, g, gb,
        stiffness, dynamics.tocsr(), selections(["AGDL", "ACL", "PEM", "CCL", "CGDL"]),
        selections(["CCL"]), selections(["BP"]),
        {"file": "附件1.xlsx", "cells": {key: raw[key] for key in
         ("C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
          "C36", "C37", "C38", "C39", "C40", "C41", "C42")}})


def modal_basis(mesh):
    """热模态分解"""
    root_capacity = np.sqrt(mesh.capacity_J_K)
    diagonal = mesh.K_W_K.diagonal() / mesh.capacity_J_K
    off = -mesh.conductance_W_K / (root_capacity[:-1] * root_capacity[1:])
    eigenvalues, vectors = eigh_tridiagonal(diagonal, off, lapack_driver="stebz")
    return eigenvalues, vectors, root_capacity


def modal_response(mesh, times, T0, ambient_K):
    """预冷温度求解"""
    times = np.atleast_1d(times).astype(float)
    if np.any(times < 0):
        raise ValueError("Response times must be nonnegative")
    initial = np.broadcast_to(np.asarray(T0, dtype=float), (mesh.size,))
    values, vectors, root_capacity = modal_basis(mesh)
    weights = vectors.T @ (root_capacity * (initial - ambient_K))
    fields = (np.exp(-times[:, None] * values) * weights) @ vectors.T
    return ambient_K + fields / root_capacity


def build_shared(n_cells=5,refine=1,endplates=True,area_cm2=25.,h=40.):
    m=build_mesh(n_cells,refine,endplates,area_cm2,h)
    if n_cells<=1:return m
    dx=m.dx.copy()
    for c,ids in enumerate(m.bp_indices):
        n=len(ids)//2
        if c>0:dx[ids[:n]]*=.5
        if c<n_cells-1:dx[ids[n:]]*=.5
    A=m.area_m2;capacity=m.rhocp*dx*A
    g=A/(dx[:-1]/(2*m.k[:-1])+dx[1:]/(2*m.k[1:]))
    gb=A/(1/h+dx[[0,-1]]/(2*m.k[[0,-1]])) if h else np.zeros(2)
    diag=np.r_[g,0]+np.r_[0,g];diag[[0,-1]]+=gb
    K=sparse.diags((-g,diag,-g),(-1,0,1),format='csr')
    edges=np.r_[0,np.cumsum(dx)]
    return replace(m,dx=dx,edges=edges,x=(edges[:-1]+edges[1:])/2,capacity_J_K=capacity,
        conductance_W_K=g,boundary_conductance_W_K=gb,K_W_K=K,A_s_inv=-sparse.diags(1/capacity)@K,
        parameter_sources=dict(m.parameter_sources,layout='N+1 physical 2 mm plates; internal plate partitioned into two 1 mm ownership halves'))


def saturation(T):
    liquid=np.exp(54.842763-6763.22/T-4.210*np.log(T)+.000367*T
        +np.tanh(.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+.014025*T))
    ice=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-.00728332*T)
    return liquid,ice


def bernoulli(x):
    """界面通量计算"""
    x=np.asarray(x,dtype=float); result=np.empty_like(x)
    small=abs(x)<1e-4; positive=x>50.; negative=x< -50.
    middle=~(small|positive|negative)
    z=x[small];z2=z*z
    result[small]=1-z/2+z2/12-z2*z2/720+z2*z2*z2/30240
    z=x[positive];ez=np.exp(-z)
    result[positive]=z*ez/(1-ez)
    z=x[negative]
    result[negative]=-z/(1-np.exp(z))
    result[middle]=x[middle]/np.expm1(x[middle])
    return result


# 电池方程计算
class FuelCellModel:
    def __init__(self,n_cells=1,refine=1,endplates=None,config=None,backend='auto',*,mesh_builder=None):
        self.cfg=config or Config(); self.nc=n_cells
        builder=build_mesh if mesh_builder is None else mesh_builder
        self.mesh=builder(n_cells,refine,n_cells>1 if endplates is None else endplates,
                          self.cfg.area_cm2,self.cfg.h)
        m=self.mesh; self.nt=len(m.dx); kind=np.asarray(m.kind)
        self.pidx=np.where(np.isin(kind,['AGDL','ACL','CCL','CGDL']))[0]
        self.bidx=np.where(np.isin(kind,['ACL','PEM','CCL']))[0]
        self.np=len(self.pidx); self.nb=len(self.bidx)
        self.pk=kind[self.pidx]; self.bk=kind[self.bidx]
        self.pcell=m.cell_ids[self.pidx]; self.bcell=m.cell_ids[self.bidx]
        self.dx_p=m.dx[self.pidx,None];self.dx_b=m.dx[self.bidx,None]
        self.eps0=np.where(self.pk=='ACL',.3916,np.where(self.pk=='CCL',.4207,.8))[:,None]
        self.phi=np.where(self.bk=='PEM',1.,.3)[:,None]
        self.pphi=np.where(np.isin(self.pk,['ACL','CCL']),.3,0.)[:,None]
        self.an=np.isin(self.pk,['AGDL','ACL'])
        self.gas_y=np.where(self.an,1.,YO)[:,None]
        self.gas_d=np.where(self.an,1.1e-4,2.2e-5)[:,None]
        self.vap_d=np.where(self.an,8.69e-5,2.48e-5)[:,None]
        self.pl=np.where(np.diff(self.pidx)==1)[0];self.pr=self.pl+1
        self.bl=np.where(np.diff(self.bidx)==1)[0];self.br=self.bl+1
        self.boundp=[];self.imean=[]
        for c in range(n_cells):
            pc=np.where(self.pcell==c)[0];bc=np.where(self.bcell==c)[0]
            self.boundp.extend([pc[0],pc[-1]])
        self.boundp=np.array(self.boundp)
        pos={v:i for i,v in enumerate(self.bidx)}
        self.pcl=np.where(np.isin(self.pk,['ACL','CCL']))[0]
        self.bcl=np.array([pos[self.pidx[i]] for i in self.pcl])
        self.ifa=np.zeros(len(self.bl))
        self.imnode=np.ones(self.nb)
        self.im2avg=np.ones(self.nb)
        for c in range(n_cells):
            bc=np.where(self.bcell==c)[0]; faces=np.cumsum(m.dx[self.bidx[bc]])[:-1]
            fr=np.where(self.bcell[self.bl]==c)[0]
            self.ifa[fr]=np.minimum(np.minimum(faces/3.4e-6,1.),(26.7e-6-faces)/11.3e-6)
            mids=np.cumsum(m.dx[self.bidx[bc]])-m.dx[self.bidx[bc]]/2
            self.imnode[bc]=np.minimum(np.minimum(mids/3.4e-6,1.),(26.7e-6-mids)/11.3e-6)
            edge=np.r_[0.,np.cumsum(m.dx[self.bidx[bc]])]
            fi=np.minimum(np.minimum(edge/3.4e-6,1.),(26.7e-6-edge)/11.3e-6)
            self.im2avg[bc]=(fi[:-1]**2+fi[:-1]*fi[1:]+fi[1:]**2)/3
        self.aclp=[np.where((self.pcell==c)&(self.pk=='ACL'))[0] for c in range(n_cells)]
        self.cclp=[np.where((self.pcell==c)&(self.pk=='CCL'))[0] for c in range(n_cells)]
        self.memb=[np.where((self.bcell==c)&(self.bk=='PEM'))[0] for c in range(n_cells)]
        start=0;self.sl={}
        for name,size in [('T',self.nt),('mb',self.nb),('mv',self.np),('ml',self.np),('mi',self.np),('ng',self.np),
                          ('aux',n_cells),('rxn',n_cells),('elec',n_cells),('charge',1),('conv',1),('wout',1)]:
            self.sl[name]=slice(start,start+size);start+=size
        self.size=start;self.scale=np.ones(start)
        self.scale[self.sl['T']]=TM
        for name in ('mb','mv','ml','mi'):self.scale[self.sl[name]]=100
        self.scale[self.sl['ng']]=PREF/(R*TM)
        for name in ('aux','rxn','elec','conv'):self.scale[self.sl[name]]=1000
        self.scale[self.sl['wout']]=.001
        self.scale[self.sl['charge']]=20
        self.atol=np.full(start,self.cfg.atol)
        for name in ('mv','ml','mi'):self.atol[self.sl[name]]=1e-15
        self.heat_c=np.column_stack([m.source_from_cell_power(np.eye(n_cells)[c],where='CCL') for c in range(n_cells)])
        self.heat_b=np.column_stack([m.source_from_cell_power(np.eye(n_cells)[c],where='BP') for c in range(n_cells)])
        self.jac_sparsity=self._sparsity()
        self.backend = os.environ.get('FC_BACKEND','numba') if backend=='auto' else backend
        if self.backend=='numba':
            attach_fast_rhs(self)
        elif self.backend!='numpy':
            raise ValueError('backend must be numpy or numba')

    def _sparsity(self):
        s=lil_matrix((self.size,self.size),dtype=int)

        for c in range(self.nc):
            ids=list(np.where(self.mesh.cell_ids==c)[0])
            for name in ('mb','mv','ml','mi','ng'):
                cells=self.bcell if name=='mb' else self.pcell
                ids+=list(self.sl[name].start+np.where(cells==c)[0])
            s[np.ix_(ids,ids)]=1
        for i in range(self.nt):s[i,max(0,i-1):min(self.nt,i+2)]=1
        s[self.sl['aux'].start:,:self.sl['aux'].start]=1
        return s.tocsr()

    def initial(self,temperature_C):
        T=np.broadcast_to(np.asarray(temperature_C)+TM,(self.nt,)).copy()
        y=np.zeros(self.size);y[self.sl['T']]=T
        y[self.sl['mb']]=(self.phi[:,0]*CLAMBDA*self.cfg.lambda0)
        y[self.sl['ng']]=self.eps0[:,0]*self.gas_y[:,0]*PREF/(R*T[self.pidx])
        return y/self.scale

    def unpack(self,y):
        Y=np.asarray(y);one=Y.ndim==1
        if one:Y=Y[:,None]
        physical=Y*self.scale[:,None]
        return {k:physical[v] for k,v in self.sl.items()},one

    def quantities(self,y,j_c):
        d,one=self.unpack(y);T=d['T'];Tp=T[self.pidx];Tb=T[self.bidx]
        ml=np.maximum(d['ml'],0);mi=np.maximum(d['mi'],0);mv=np.maximum(d['mv'],0)
        eg=np.maximum(self.eps0-ml/990-mi/920,1e-12)
        lam=np.maximum(d['mb']/(CLAMBDA*self.phi),1e-8)
        c=np.maximum(d['ng']/eg,1e-15)
        kap=(.5139*lam-.326)*np.exp(1268*(1/303.15-1/Tb))
        j=max(float(j_c),0)*1e4
        V=[];act=[];ohm=[];con=[];rev=[]
        for cell,(a,o,b) in enumerate(zip(self.aclp,self.cclp,self.memb)):
            wa=self.dx_p[a]/3.4e-6;wo=self.dx_p[o]/11.3e-6
            Te=np.sum(Tp[o]*wo,axis=0)
            pH=np.exp(np.sum(wa*np.log(c[a]*R*Tp[a]),axis=0))
            pO=np.exp(np.sum(wo*np.log(c[o]*R*Tp[o]),axis=0))
            E=1.229-.00085*(Te-298.15)+R*Te/(4*F)*np.log(YO)
            co=R*Te/(2*F)*np.log(PREF/pH)+R*Te/(4*F)*np.log(YO*PREF/pO)
            si=mi[o]/(920*self.eps0[o])
            cb_c=self.bcl[np.isin(self.pcl,o)]
            hydration=(np.minimum(lam[cb_c],14.003)/3.)**self.cfg.hydration_exponent
            j0=np.sum(wo*self.cfg.gamma0*.01*np.exp(-self.cfg.activation_energy/R*(1/Tp[o]-1/298.15))
                         *np.maximum(1-si,1e-12)**3.5*hydration,axis=0)
            ac=R*Te/(self.cfg.alpha*F)*np.arcsinh(j/(2*j0))
            oh=j*(self.cfg.gamma_R*1e-6+np.sum(self.dx_b[b]/np.maximum(kap[b],1e-9),axis=0))
            if self.cfg.cl_resistance_factor:
                cb=np.where((self.bcell==cell)&(self.bk!='PEM'))[0]
                oh+=j*self.cfg.cl_resistance_factor*np.sum(self.dx_b[cb]*self.im2avg[cb,None]
                        /(self.phi[cb]**1.5*np.maximum(kap[cb],1e-9)),axis=0)
            V.append(E-co-ac-oh);act.append(ac);ohm.append(oh);con.append(co);rev.append(E-co)
        return d,dict(Tp=Tp,Tb=Tb,eg=eg,lam=lam,c=c,kap=kap,V=np.array(V),act=np.array(act),
                       ohm=np.array(ohm),con=np.array(con),rev=np.array(rev)),one

    def rhs(self,t,y,j_c,q,ambient_C):
        cfg=self.cfg;d,z,one=self.quantities(y,j_c);T=d['T'];Tp=z['Tp'];eg=z['eg'];lam=z['lam'];j=float(j_c)*1e4
        Dg=self.gas_d*(Tp/298.15)**1.75*eg**1.5
        Dv=self.vap_d*(Tp/298.15)**1.75*eg**1.5
        Ng=self._diffusion(d['ng']/eg,Dg,self.dx_p,self.pl,self.pr)
        Nv=self._diffusion(d['mv']/eg,Dv,self.dx_p,self.pl,self.pr)
        bp=self.boundp;cin=self.gas_y[bp]*PREF/(R*Tp[bp])
        gout=2*Dg[bp]/self.dx_p[bp]*(d['ng'][bp]/eg[bp]-cin)
        vout=2*Dv[bp]/self.dx_p[bp]*(d['mv'][bp]/eg[bp])
        Ng[bp]-=gout/self.dx_p[bp];Nv[bp]-=vout/self.dx_p[bp]
        Ng[self.pk=='ACL']-=j/(2*F*3.4e-6)
        Ng[self.pk=='CCL']-=j/(4*F*11.3e-6)
        Db=cfg.gamma_Db*1e-10*np.exp(2416*(1/303.15-1/z['Tb']))*(2.563-.33*lam+.0264*lam**2-.000671*lam**3)
        Db=np.maximum(Db,1e-20)
        Dmass=CLAMBDA*self.phi**1.5*Db
        if getattr(cfg,'exponential_water_flux',False):


            g=1/(self.dx_b[self.bl]/(2*Dmass[self.bl])+self.dx_b[self.br]/(2*Dmass[self.br]))
            v=cfg.gamma_drag*MW/F*(2.5/22)*j*self.ifa[:,None]
            pe=v/g
            flux=g*(bernoulli(-pe)*lam[self.bl]-bernoulli(pe)*lam[self.br])
            B=np.zeros_like(lam)
            np.add.at(B,self.bl,-flux/self.dx_b[self.bl]);np.add.at(B,self.br,flux/self.dx_b[self.br])
        else:
            B=self._diffusion(lam,Dmass,self.dx_b,self.bl,self.br)
            drag=cfg.gamma_drag*MW/F*(2.5/22)*lam[self.bl]*j*self.ifa[:,None]
            np.add.at(B,self.bl,-drag/self.dx_b[self.bl]);np.add.at(B,self.br,drag/self.dx_b[self.br])
        lp=np.zeros_like(eg);lp[self.pcl]=lam[self.bcl]
        mv=np.maximum(d['mv'],0);ml=np.maximum(d['ml'],0);mi=np.maximum(d['mi'],0)
        ps,pi=saturation(Tp);pv=mv*R*Tp/(MW*eg)
        a=np.minimum(pv/ps,1.);le=.043+17.81*a-39.85*a*a+36*a**3
        fm=lambda x:x/(x+cfg.mstar)
        vb=self.pphi*CLAMBDA*(np.maximum(le-lp,0)/cfg.tau_ads*fm(mv)-np.maximum(lp-le,0)/cfg.tau_des)*eg/self.eps0
        lb=self.pphi*CLAMBDA*(np.maximum(14.003-lp,0)/cfg.tau_lb*fm(ml)-np.maximum(lp-14.003,0)/cfg.tau_lb_des)
        vl=np.maximum(mv-eg*MW*ps/(R*Tp),0)/cfg.tau_c-ml/cfg.tau_e*np.maximum(1-pv/ps,0)
        gf=np.maximum((TM-Tp)/cfg.delta_phase_K,0);gm=np.maximum((Tp-TM)/cfg.delta_phase_K,0)
        li=cfg.gamma_f*(gf*ml/cfg.tau_f-gm*mi/cfg.tau_m)
        vi=gf/(1+gf)*(fm(mi)*np.maximum(mv-eg*MW*pi/(R*Tp),0)/cfg.tau_di-mi/cfg.tau_su*np.maximum(1-pv/pi,0))
        B[self.bcl]+=vb[self.pcl]+lb[self.pcl]
        dy=np.zeros((self.size,T.shape[1]))
        dy[self.sl['mb']]=B;dy[self.sl['ng']]=Ng
        dy[self.sl['mv']]=Nv-vb-vl-vi
        dl=vl-li-lb;dl[self.pk=='CCL']+=MW*j/(2*F*11.3e-6)
        dy[self.sl['ml']]=dl;dy[self.sl['mi']]=li+vi
        heatphase=LV*(vb+vl)+(LV+LF)*vi+LF*li
        Paux=np.broadcast_to(np.asarray(q,dtype=float),(self.nc,))*cfg.area_cm2
        Prxn=self.mesh.area_m2*j*(1.48-z['V'])
        heat=self.heat_b@Paux[:,None]+self.heat_c@Prxn
        heat[self.pidx]+=heatphase*self.dx_p*self.mesh.area_m2
        dy[self.sl['T']]=self.mesh.A_s_inv@(T-(ambient_C+TM))+heat/self.mesh.capacity_J_K[:,None]
        dy[self.sl['aux']]=Paux[:,None];dy[self.sl['rxn']]=Prxn
        dy[self.sl['elec']]=self.mesh.area_m2*j*z['V']
        dy[self.sl['charge']]=j_c
        bc=self.mesh.boundary_conductance_W_K
        dy[self.sl['conv']]=bc[0]*(T[0]-(ambient_C+TM))+bc[1]*(T[-1]-(ambient_C+TM))
        dy[self.sl['wout']]=self.mesh.area_m2*np.sum(vout,axis=0)
        result=dy/self.scale[:,None]
        return result[:,0] if one else result

    @staticmethod
    def _diffusion(c,D,dx,left,right):
        g=1/(dx[left]/(2*D[left])+dx[right]/(2*D[right]))
        f=g*(c[left]-c[right]);out=np.zeros_like(c)
        np.add.at(out,left,-f/dx[left]);np.add.at(out,right,f/dx[right])
        return out


PARAMETERS = ("area_cm2", "gamma0", "gamma_R", "gamma_f", "gamma_Db",
              "gamma_drag", "cl_resistance_factor", "hydration_exponent",
              "activation_energy", "alpha", "tau_ads", "tau_des", "tau_lb",
              "tau_lb_des", "tau_c", "tau_e", "tau_f", "tau_m", "tau_di",
              "tau_su", "mstar", "delta_phase_K")


@njit(cache=True, nogil=True, fastmath=False)
def _bernoulli(x):
    if abs(x) < 1e-4:
        x2 = x*x
        return 1-x/2+x2/12-x2*x2/720+x2*x2*x2/30240
    if x > 50.:
        ex = np.exp(-x)
        return x*ex/(1-ex)
    if x < -50.:
        return -x/(1-np.exp(x))
    return x/np.expm1(x)


@njit(cache=True, nogil=True, fastmath=False)
def _scalar(Y, column, j_c, q, ambient_C, ids, arrays):
    pidx, bidx, pk, bk, pcell, bcell, pl, pr, bl, br, boundp, p_to_b, offsets, aindices, aindptr = ids
    eps0, phi, pphi, gas_y, gas_d, vap_d, dxp, dxb, ifa, im2avg, heat_b, heat_c, adata, capacity, bc, scale, cfg = arrays
    nt = offsets[1]
    nb = bidx.size
    npore = pidx.size
    nc = offsets[7] - offsets[6]
    physical = np.empty(Y.shape[0])
    for i in range(physical.size):
        physical[i] = Y[i, column] * scale[i]
    T = physical[:nt]
    mb = physical[offsets[1]:offsets[2]]
    mvraw = physical[offsets[2]:offsets[3]]
    mlraw = physical[offsets[3]:offsets[4]]
    miraw = physical[offsets[4]:offsets[5]]
    ng = physical[offsets[5]:offsets[6]]
    eg = np.empty(npore)
    Tp = np.empty(npore)
    c = np.empty(npore)
    Dg = np.empty(npore)
    Dv = np.empty(npore)
    lam = np.empty(nb)
    kap = np.empty(nb)
    Dmass = np.empty(nb)
    for i in range(npore):
        Tp[i] = T[pidx[i]]
        eg[i] = max(eps0[i] - max(mlraw[i], 0.) / 990. - max(miraw[i], 0.) / 920., 1e-12)
        c[i] = max(ng[i] / eg[i], 1e-15)
        temperature_factor = (Tp[i] / 298.15)**1.75
        Dg[i] = gas_d[i] * temperature_factor * eg[i]**1.5
        Dv[i] = vap_d[i] * temperature_factor * eg[i]**1.5
    for i in range(nb):
        lam[i] = max(mb[i] / (CLAMBDA * phi[i]), 1e-8)
        temperature = T[bidx[i]]
        kap[i] = (.5139 * lam[i] - .326) * np.exp(1268. * (1 / 303.15 - 1 / temperature))
        Db = cfg[4] * 1e-10 * np.exp(2416. * (1 / 303.15 - 1 / temperature)) * (
            2.563 - .33 * lam[i] + .0264 * lam[i]**2 - .000671 * lam[i]**3)
        Dmass[i] = CLAMBDA * phi[i]**1.5 * max(Db, 1e-20)
    j = j_c * 1e4
    j_voltage = max(j_c, 0.) * 1e4
    voltage = np.empty(nc)
    for cell in range(nc):
        Te, logpH, logpO, j0 = 0., 0., 0., 0.
        for i in range(npore):
            if pcell[i] != cell:
                continue
            if pk[i] == 1:
                logpH += dxp[i] / 3.4e-6 * np.log(c[i] * R * Tp[i])
            elif pk[i] == 2:
                weight = dxp[i] / 11.3e-6
                Te += Tp[i] * weight
                logpO += weight * np.log(c[i] * R * Tp[i])
                si = max(miraw[i], 0.) / (920. * eps0[i])
                hydration = (min(lam[p_to_b[i]], 14.003) / 3.)**cfg[7]
                j0 += weight * cfg[1] * .01 * np.exp(-cfg[8] / R * (1 / Tp[i] - 1 / 298.15)) * max(1 - si, 1e-12)**3.5 * hydration
        pH, pO = np.exp(logpH), np.exp(logpO)
        E = 1.229 - .00085 * (Te - 298.15) + R * Te / (4 * F) * np.log(YO)
        co = R * Te / (2 * F) * np.log(PREF / pH) + R * Te / (4 * F) * np.log(YO * PREF / pO)
        ac = R * Te / (cfg[9] * F) * np.arcsinh(j_voltage / (2 * j0))
        ohmic = cfg[2] * 1e-6
        cl_ohmic = 0.
        for i in range(nb):
            if bcell[i] == cell:
                if bk[i] == 1:
                    ohmic += dxb[i] / max(kap[i], 1e-9)
                elif cfg[6] != 0.:
                    cl_ohmic += dxb[i] * im2avg[i] / (phi[i]**1.5 * max(kap[i], 1e-9))
        voltage[cell] = E - co - ac - j_voltage * (ohmic + cfg[6] * cl_ohmic)
    Ng = np.zeros(npore)
    Nv = np.zeros(npore)
    B = np.zeros(nb)
    for face in range(pl.size):
        left, right = pl[face], pr[face]
        g = 1. / (dxp[left] / (2 * Dg[left]) + dxp[right] / (2 * Dg[right]))
        f = g * (ng[left] / eg[left] - ng[right] / eg[right])
        Ng[left] -= f / dxp[left]
        Ng[right] += f / dxp[right]
        gv = 1. / (dxp[left] / (2 * Dv[left]) + dxp[right] / (2 * Dv[right]))
        fv = gv * (mvraw[left] / eg[left] - mvraw[right] / eg[right])
        Nv[left] -= fv / dxp[left]
        Nv[right] += fv / dxp[right]
    total_vout = 0.
    for i in boundp:
        cin = gas_y[i] * PREF / (R * Tp[i])
        gout = 2 * Dg[i] / dxp[i] * (ng[i] / eg[i] - cin)
        vout = 2 * Dv[i] / dxp[i] * (mvraw[i] / eg[i])
        Ng[i] -= gout / dxp[i]
        Nv[i] -= vout / dxp[i]
        total_vout += vout
    for face in range(bl.size):
        left, right = bl[face], br[face]
        g = 1. / (dxb[left] / (2 * Dmass[left]) + dxb[right] / (2 * Dmass[right]))
        if cfg[23] != 0.:
            velocity = cfg[5] * MW / F * (2.5 / 22) * j * ifa[face]
            pe = velocity / g
            f = g * (_bernoulli(-pe) * lam[left] - _bernoulli(pe) * lam[right])
            B[left] -= f / dxb[left]
            B[right] += f / dxb[right]
        else:
            f = g * (lam[left] - lam[right])
            B[left] -= f / dxb[left]
            B[right] += f / dxb[right]
            drag = cfg[5] * MW / F * (2.5 / 22) * lam[left] * j * ifa[face]
            B[left] -= drag / dxb[left]
            B[right] += drag / dxb[right]
    dy = np.zeros(Y.shape[0])
    heatphase = np.empty(npore)
    for i in range(npore):
        temperature = Tp[i]
        lp = lam[p_to_b[i]] if p_to_b[i] >= 0 else 0.
        mv, ml, mi = max(mvraw[i], 0.), max(mlraw[i], 0.), max(miraw[i], 0.)
        ps = np.exp(54.842763 - 6763.22 / temperature - 4.210 * np.log(temperature) + .000367 * temperature
            + np.tanh(.0415 * (temperature - 218.8)) * (53.878 - 1331.22 / temperature - 9.44523 * np.log(temperature) + .014025 * temperature))
        pi = np.exp(9.550426 - 5723.265 / temperature + 3.53068 * np.log(temperature) - .00728332 * temperature)
        pv = mv * R * temperature / (MW * eg[i])
        activity = min(pv / ps, 1.)
        le = .043 + 17.81 * activity - 39.85 * activity * activity + 36 * activity**3
        vb = pphi[i] * CLAMBDA * (max(le - lp, 0.) / cfg[10] * mv / (mv + cfg[20])
             - max(lp - le, 0.) / cfg[11]) * eg[i] / eps0[i]
        lb = pphi[i] * CLAMBDA * (max(14.003 - lp, 0.) / cfg[12] * ml / (ml + cfg[20])
             - max(lp - 14.003, 0.) / cfg[13])
        vl = max(mv - eg[i] * MW * ps / (R * temperature), 0.) / cfg[14] - ml / cfg[15] * max(1 - pv / ps, 0.)
        gf = max((TM - temperature) / cfg[21], 0.)
        gm = max((temperature - TM) / cfg[21], 0.)
        li = cfg[3] * (gf * ml / cfg[16] - gm * mi / cfg[17])
        vi = gf / (1 + gf) * (mi / (mi + cfg[20]) * max(mv - eg[i] * MW * pi / (R * temperature), 0.) / cfg[18]
                               - mi / cfg[19] * max(1 - pv / pi, 0.))
        if p_to_b[i] >= 0:
            B[p_to_b[i]] += vb + lb
        if pk[i] == 1:
            Ng[i] -= j / (2 * F * 3.4e-6)
        elif pk[i] == 2:
            Ng[i] -= j / (4 * F * 11.3e-6)
        dy[offsets[2] + i] = Nv[i] - vb - vl - vi
        dy[offsets[3] + i] = vl - li - lb
        if pk[i] == 2:
            dy[offsets[3] + i] += MW * j / (2 * F * 11.3e-6)
        dy[offsets[4] + i] = li + vi
        dy[offsets[5] + i] = Ng[i]
        heatphase[i] = LV * (vb + vl) + (LV + LF) * vi + LF * li
    for i in range(nb):
        dy[offsets[1] + i] = B[i]


    area = cfg[22]
    Prxn = area * j * (1.48 - voltage)
    Paux = q * cfg[0]
    heat = np.zeros(nt)
    for i in range(nt):
        for cell in range(nc):
            heat[i] += heat_b[i * nc + cell] * Paux[cell] + heat_c[i * nc + cell] * Prxn[cell]
    for i in range(npore):
        heat[pidx[i]] += heatphase[i] * dxp[i] * area
    for i in range(nt):
        thermal = 0.
        for index in range(aindptr[i], aindptr[i + 1]):
            thermal += adata[index] * (T[aindices[index]] - (ambient_C + TM))
        dy[i] = thermal + heat[i] / capacity[i]
    for cell in range(nc):
        dy[offsets[6] + cell] = Paux[cell]
        dy[offsets[7] + cell] = Prxn[cell]
        dy[offsets[8] + cell] = area * j * voltage[cell]
    dy[offsets[9]] = j_c
    dy[offsets[10]] = bc[0] * (T[0] - (ambient_C + TM)) + bc[1] * (T[-1] - (ambient_C + TM))
    dy[offsets[11]] = area * total_vout
    for i in range(dy.size):
        dy[i] /= scale[i]
    return dy


@njit(cache=True, nogil=True, fastmath=False)
def _batch(Y, j_c, q, ambient_C, ids, arrays):
    result = np.empty(Y.shape)
    for column in range(Y.shape[1]):
        result[:, column] = _scalar(Y, column, j_c, q, ambient_C, ids, arrays)
    return result


# 计算后端加速
def attach_fast_rhs(model):
    """计算后端加速"""
    if getattr(model, "rhs_backend", None) == "numba-float64":
        return model
    integer = lambda value: np.ascontiguousarray(value, dtype=np.int64)
    real = lambda value: np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    p_to_b = np.full(model.np, -1, dtype=np.int64)
    p_to_b[model.pcl] = model.bcl
    offsets = [model.sl[name].start for name in
               ("T", "mb", "mv", "ml", "mi", "ng", "aux", "rxn", "elec", "charge", "conv", "wout")]
    matrix = model.mesh.A_s_inv.tocsr()
    ids = tuple(map(integer, (model.pidx, model.bidx,
        [{"AGDL": 0, "ACL": 1, "CCL": 2, "CGDL": 3}[s] for s in model.pk],
        [{"ACL": 0, "PEM": 1, "CCL": 2}[s] for s in model.bk],
        model.pcell, model.bcell, model.pl, model.pr, model.bl, model.br,
        model.boundp, p_to_b, offsets, matrix.indices, matrix.indptr)))
    cfg_values = real([getattr(model.cfg, name) for name in PARAMETERS] +
                     [model.mesh.area_m2, float(getattr(model.cfg, 'exponential_water_flux', False))])
    arrays = tuple(map(real, (model.eps0, model.phi, model.pphi, model.gas_y,
        model.gas_d, model.vap_d, model.dx_p, model.dx_b, model.ifa, model.im2avg,
        model.heat_b, model.heat_c, matrix.data, model.mesh.capacity_J_K,
        model.mesh.boundary_conductance_W_K, model.scale))) + (cfg_values,)
    numpy_rhs = model.rhs
    def accelerated(t, y, j_c, q, ambient_C):
        values = np.asarray(y, dtype=np.float64)
        one = values.ndim == 1
        Y = np.ascontiguousarray(values[:, None] if one else values)
        power = np.ascontiguousarray(np.broadcast_to(np.asarray(q, dtype=np.float64), (model.nc,)))

        for i, name in enumerate(PARAMETERS):
            cfg_values[i] = getattr(model.cfg, name)
        cfg_values[23] = float(getattr(model.cfg, 'exponential_water_flux', False))
        result = _batch(Y, float(j_c), power, float(ambient_C), ids, arrays)
        return result[:, 0] if one else result
    model.numpy_rhs = numpy_rhs
    model.rhs = accelerated
    model.rhs_backend = "numba-float64"
    model.fast_rhs_metadata = {"method": "njit scalar equations plus compiled batch column loop",
        "fastmath": False, "dtype": "float64", "scope": "RHS only; BDF and state acceptance unchanged"}
    return model
