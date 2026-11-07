"""Refit both feasibility models and replay all archived Bayesian decisions."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys


def replay(repo):
    sys.path.insert(0, str(repo / "research/study_v2"))
    import search
    import numpy as np
    from scipy.stats import qmc

    chains = defaultdict(list)
    with (repo / "results/stack/search_evaluations.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            chains[row["chain_id"]].append(row)
    checks = {}
    for chain, rows in sorted(chains.items()):
        rows.sort(key=lambda row: int(row["budget_index"]))
        seed, family = int(rows[0]["seed"]), rows[0]["family"]
        bounds = np.asarray(search.P["design_families"][family]["bounds"])
        dim = len(bounds)
        sampler = qmc.Sobol(dim, scramble=True, seed=seed+1000)
        initial = []
        for power, cutoff in search.P["search"]["anchors_uniform_q_and_cutoff"]:
            initial.append(([power] if family == "uniform" else [power]*3) + [cutoff])
        sobol = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(2)
        initial.extend(qmc.scale(sobol, bounds[:, 0], bounds[:, 1]).tolist())
        hist, deviations = [], []
        for index, row in enumerate(rows):
            x = json.loads(row["design_x"])
            if index < len(initial):
                proposal = initial[index]
            else:
                pool = sampler.random(search.P["search"]["candidate_pool_per_iteration"])
                proposal, info = search.propose(family, hist, pool, seed)
            deviations.append(float(np.max(np.abs(np.asarray(proposal)-x))))
            hist.append(dict(
                design_x=x, complete_180s=row["complete_180s"]=="True",
                g_window_K=float(row["g_window_K"]) if row["g_window_K"] else None,
                numerically_resolved=row["numerically_resolved"]=="True",
                audit={"passed":row["physical_audit_passed"]=="True"},
                feasible=row["feasible"]=="True", planned_energy_J=float(row["planned_energy_J"])))
        winner = min((r for r in hist if r["feasible"]), key=lambda r:r["planned_energy_J"])
        checks[chain] = {"evaluations":len(rows), "adaptive":len(rows)-len(initial),
                         "maximum_command_coordinate_difference":max(deviations),
                         "winner_energy_J":winner["planned_energy_J"]}
        print(chain, checks[chain], flush=True)
    expected = len(chains)==12 and sum(c["adaptive"] for c in checks.values())==288
    passed = expected and all(c["maximum_command_coordinate_difference"]<1e-10 for c in checks.values())
    return {"status":"PASS" if passed else "FAIL", "chains":checks}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    result=replay(args.repo.resolve())
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if result["status"]!="PASS":
        raise SystemExit("At least one replayed proposal differed from its archived command")
