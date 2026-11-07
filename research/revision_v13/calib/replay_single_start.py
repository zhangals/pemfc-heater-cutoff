"""Re-fit one of the documented single-temperature starts from a locally held workbook."""
import argparse
import csv
import json
from pathlib import Path
import sys


def refit(repo, workbook, trial_file, temperature, start):
    sys.path.insert(0,str(repo/"research/revision_v9_current/numerics"))
    import run_recorded as recorded
    import numpy as np
    from scipy.optimize import least_squares

    model=recorded.original
    recorded.BOOK=workbook
    data=recorded.records()[temperature]
    protocol=repo/"research/study_v3_diagnostics/cross_condition/protocol.json"
    p0=json.loads(protocol.read_text(encoding="utf-8"))["starts"][start-1]
    x0=model.encode(np.asarray(p0))
    trials=[]

    class BudgetReached(Exception):
        pass

    def residual(x):
        if len(trials)>=48:
            raise BudgetReached
        params=model.decode(x)
        values,_,meta,_,_,_=model.simulate(params,data,refine=2,convention="recorded",fields=False)
        values=(np.full(368,np.sqrt((10000+100*(36.6-meta["time_s"]))/368))
                if values is None else values)
        trials.append({"evaluation":len(trials)+1,"parameters":params.tolist(),
                       "status":meta["status"],"loss":float(values@values)})
        return values

    try:
        least_squares(residual,x0,bounds=(np.zeros(7),np.ones(7)),method="trf",diff_step=1e-3,max_nfev=48)
    except BudgetReached:
        pass
    with trial_file.open(encoding="utf-8-sig") as f:
        previous=[r for r in csv.DictReader(f) if int(r["temperature_C"])==temperature and int(r["start"])==start]
    assert len(previous)==len(trials)==48
    errors=[abs(float(p["objective"])-r["loss"]) for p,r in zip(previous,trials)]
    winner=min((r for r in trials if r["status"]=="COMPLETE"),key=lambda r:r["loss"])
    archive_winner=min((r for r in previous if r["status"]=="COMPLETE"),key=lambda r:float(r["objective"]))
    same_winner=winner["evaluation"]==int(archive_winner["evaluation"])
    return {"status":"PASS" if same_winner and max(errors)<1e-7 else "CHECK",
            "temperature_C":temperature,"start":start,"attempted":len(trials),
            "winning_evaluation":winner["evaluation"],"winner_loss":winner["loss"],
            "archived_winner_loss":float(archive_winner["objective"]),
            "max_abs_objective_difference":max(errors),"trials":trials}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--workbook",type=Path,required=True)
    parser.add_argument("--trials",type=Path,required=True)
    parser.add_argument("--temperature",type=int,choices=(-20,-25),default=-25)
    parser.add_argument("--start",type=int,choices=(1,2),default=1)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    result=refit(a.repo.resolve(),a.workbook.resolve(),a.trials,a.temperature,a.start)
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in result.items() if k!="trials"})
