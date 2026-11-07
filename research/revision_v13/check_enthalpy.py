"""Recompute the two whole-stack enthalpy balances from simulated nodal fields.

Run from the repository root: python research/revision_v13/check_enthalpy.py
Each file contains simulated temperatures, vapour/ice densities, node capacities
and porous volumes, and integrated heat, mass and electrical counters. The
time axis starts when the cold-start command is applied.
"""
from pathlib import Path
import json
import numpy as np


BASE = Path(__file__).resolve().parents[2] / "results/review_checks/enthalpy"


def check(item):
    pattern=f"{item['schedule']}_r{item['r']}.npz"
    with np.load(BASE/pattern,allow_pickle=False) as d:
        t=d["time_s"]
        assert np.array_equal(t,np.arange(361)*0.5)
        C=d["capacity_J_K"]; dT=d["temperature_K"]-d["temperature_K"][0]
        ep=d["endplate_mask"]
        U=dT @ C
        L=(2.5e6*d["vapour_kg_m3"]-333600*d["ice_kg_m3"]) @ d["porous_volume_m3"]
        change=U+L-L[0]
        out=d["convection_out_J"]-d["convection_out_J"][0]
        export=2.5e6*(d["vapour_export_kg"]-d["vapour_export_kg"][0])
        heater=(d["heater_J"]-d["heater_J"][0]).sum(axis=1)
        rxn=(d["reaction_J"]-d["reaction_J"][0]).sum(axis=1)
        elec=(d["electrical_J"]-d["electrical_J"][0]).sum(axis=1)
        predicted_input=5*float(d["area_cm2"])*np.minimum(t,float(d["cutoff_s"]))
        thermoneutral=5*float(d["area_cm2"])*1.48*(d["charge_C_cm2"]-d["charge_C_cm2"][0])
        np.testing.assert_allclose(heater,predicted_input,rtol=0,atol=1e-7)
        np.testing.assert_allclose(rxn,thermoneutral-elec,rtol=0,atol=1e-7)
        res=change+out+export-rxn-heater
        norm=1+abs(change)+abs(out)+abs(export)+abs(rxn)+abs(heater)
        # The endpoint residuals are around 10 nJ, so floating-point reduction
        # order matters. Compare individual energy terms in joules as well.
        for k,v in (("enthalpy_change_J",change[-1]),("sensible_change_J",U[-1]),
                    ("latent_change_J",L[-1]-L[0]),("convection_out_J",out[-1]),
                    ("vapor_export_J",export[-1]),("aux_J",heater[-1]),("rxn_J",rxn[-1]),
                    ("endplate_storage_38_55p5_J",(dT[t==55.5][:,ep]-dT[t==38][:,ep])[0]@C[ep])):
            np.testing.assert_allclose(v,item[k],rtol=0,atol=2e-8)
        assert max(abs(res)) < 2e-8 and max(abs(res)/norm)<1e-12
        print(f"{pattern}: heater {heater[-1]:.3f} J; maximum |enthalpy residual| {max(abs(res))*1e9:.2f} nJ")


if __name__=="__main__":
    for row in json.loads((BASE/"reference.json").read_text("utf8"))["cases"]:check(row)
