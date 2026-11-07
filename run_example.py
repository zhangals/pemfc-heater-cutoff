"""Run a uniform five-cell heater schedule and report the complete-window minimum."""
import argparse
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'research/study_v2'))
import engine as eng

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=BASE/'case.json')
    args=parser.parse_args()
    case=json.loads(args.case.read_text(encoding='utf-8'))
    q=case['q_W_cm2']
    cutoff=case['cutoff_s']
    interval=case['interval_s']
    horizon=case['horizon_s']
    if not(0<=q<=1 and 0<cutoff<=interval[0]<interval[1]<=horizon):
        raise ValueError("Invalid heat command or assessment window")
    model,ti,ta,_=eng.make_model({'refine':case['refine'],'overrides':{}})
    command,_,breaks,_=eng.waveform([q]*5,cutoff,'constant')
    result,_,segments,_=eng.integrate(model,horizon,ti,ta,eng.current_ramp,command,
        breakpoints=breaks,dense_interval=interval)
    diag=eng.analyze_dense(segments,interval,result['time_s'],threshold=case['required_C'])
    out={'solver_status':result['status'],'final_time_s':result['time_s'],
         'command_energy_J':25*5*q*cutoff,'interval_s':interval,
         'continuous_minimum_C':diag['minimum_mea_mean_C'],'time_of_minimum_s':diag['minimum_time_s'],
         'limiting_cell':diag['minimum_cell_one_based'],
         'passes':result['status']=='COMPLETE' and diag['all_means_above_threshold_for_full_interval']}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()


