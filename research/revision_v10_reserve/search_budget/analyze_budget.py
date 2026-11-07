"""Reconstruct running incumbents and score recorded, sequential acquisition predictions."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "research/study_v2"
BUDGETS = (8, 16, 24, 32)
RESERVES = (.01, 1., 3.)
SEEDS = (20261106, 20261107)
FAMILIES = ("uniform", "spatial")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def beta_rate(outcomes):
    return (1+sum(outcomes))/(2+len(outcomes))


def lowest(history):
    feasible = [h for h in history if h["feasible"]]
    return min(feasible, key=lambda h: (h["energy_J"], h["evaluation"])) if feasible else None


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired(en, zh):
    return {"en":en,"zh":zh}


def score_summary(forecasts):
    result={}
    for task in ("admissible_completion", "margin_given_completion", "feasibility"):
        subset = [r for r in forecasts if r["task"]==task]
        result[task]={"n":len(subset), "positive":sum(r["observed"] for r in subset),
            "mean_forecast":mean(r["p"] for r in subset),
            "frequency":mean(r["observed"] for r in subset),
            "brier":mean((r["p"]-r["observed"])**2 for r in subset),
            "running_frequency_brier":mean((r["reference_probability"]-r["observed"])**2 for r in subset)}
    return result


def reconstruct(chain, inputs, protocol_hash):
    history=[]
    forecasts=[]
    prefix_rows=[]
    for idx, job_id in enumerate(chain["job_ids"],1):
        path=STUDY/"runs/search"/(job_id+".json")
        result=load(path)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        job=result["job"]
        assert result["frozen_protocol_sha256"]==protocol_hash
        assert int(job["budget_index"])==idx and job["job_id"]==job_id
        assert (job["family"],job["reserve_C"],job["seed"])==(chain["family"],chain["reserve_C"],chain["seed"])
        assert job["refine"]==1 and job["dense"] is False and job["allowance_K"]==0.1
        complete = result["complete_180s"]
        resolved = result["numerically_resolved"]
        audited = result["audit"]["passed"]
        gain = result["g_window_K"]
        feasible = bool(complete and audited and gain is not None and gain>=0)
        assert feasible == result["feasible"]
        assert complete==(result["status"]=="COMPLETE" and abs(result["time_s"]-180.)<=1e-8)
        assert resolved==(result["status"] not in ("NUMERICAL_LIMIT","NUMERICAL_FAILURE"))
        assert (gain is not None)==bool(result["sampled_window_covered"])
        if gain is not None:
            assert abs(gain-(result["min_window_mea_C"]-(chain["reserve_C"]+.1)))<1e-10
        energy=25*sum(job["powers"])*job["cutoff_s"]
        assert math.isclose(energy,result["planned_energy_J"],rel_tol=1e-12)
        a=job["acquisition"]
        assert (a is None)==(idx<=8)
        if idx>8:
            archived=STUDY/"searches"/chain["chain_id"]/f"acquisition_{idx:02d}.json"
            assert load(archived)==a
            called=[i for i,r in enumerate(history) if r["complete"] and r["margin_K"] is not None]
            resolved_ids=[i for i,r in enumerate(history) if r["resolved"]]
            assert a["margin_training_indices"]==called and a["completion_training_indices"]==resolved_ids
            previous=lowest(history)
            assert math.isclose(previous["energy_J"],a["incumbent_J"],rel_tol=1e-12) if previous else a["incumbent_J"] is None
            pc,pg=a["completion_probability"],a["margin_probability"]
            assert 0<=pc<=1 and 0<=pg<=1
            def save_score(task, prob, y, ref):
                forecasts.append({"chain":chain["chain_id"],"evaluation":idx,"task":task,
                                  "p":prob,"observed":int(y),"reference_probability":beta_rate(ref)})
            if resolved:
                save_score("admissible_completion",pc,complete and audited,
                           [history[i]["admissible_completion"] for i in resolved_ids])
                save_score("feasibility",pc*pg,feasible,[history[i]["feasible"] for i in resolved_ids])
            if gain is not None:
                save_score("margin_given_completion",pg,gain>=0,
                           [history[i]["margin_K"]>=0 for i in called])
        history.append({"chain":chain["chain_id"], "evaluation":idx, "job_id":job_id,
                        "family":chain["family"],"reserve_C":chain["reserve_C"],"seed":chain["seed"],
                        "status":result["status"],"resolved":resolved,"complete":complete,
                        "admissible_completion":complete and audited,"margin_K":gain,"feasible":feasible,
                        "energy_J":energy,"q_edge":job["powers"][0],"q_inner":job["powers"][1],
                        "q_center":job["powers"][2],"cutoff_s":job["cutoff_s"]})
        if idx in BUDGETS:
            incumbent=lowest(history)
            prefix_rows.append({"chain":chain["chain_id"],"family":chain["family"],
                    "reserve_C":chain["reserve_C"],"seed":chain["seed"],"budget":idx,
                    "feasible_count":sum(r["feasible"] for r in history),
                    "energy_J":incumbent["energy_J"] if incumbent else None,
                    "winning_evaluation":incumbent["evaluation"] if incumbent else None})
    best=lowest(history)
    assert best["job_id"]==chain["winner"]["job"]["job_id"]
    assert math.isclose(best["energy_J"],chain["winner"]["planned_energy_J"],rel_tol=1e-12)
    return history,forecasts,prefix_rows


def main():
    summary_path=STUDY/"search_summary.json"
    protocol_path=STUDY/"protocol.json"
    rep_path=STUDY/"representatives.json"
    protocol=load(protocol_path)
    protocol_hash=digest(protocol_path)
    summary=load(summary_path)
    assert summary["protocol_sha256"]==protocol_hash and len(summary["chains"])==12
    inputs={p.relative_to(ROOT).as_posix():digest(p) for p in [summary_path,protocol_path,
            rep_path,STUDY/"search.py",STUDY/"engine.py"]}
    histories={};scores=[];prefixes=[]
    for chain in summary["chains"]:
        key=(chain["reserve_C"],chain["family"],chain["seed"])
        assert key not in histories
        history,forecasts,prefix=reconstruct(chain,inputs,protocol_hash)
        histories[key]=history;scores.extend(forecasts);prefixes.extend(prefix)
    assert len(prefixes)==48
    for r in RESERVES:
        for seed in SEEDS:
            uniform=histories[r,"uniform",seed][:4]
            spatial=histories[r,"spatial",seed][:4]
            for u,s,(q,tau) in zip(uniform,spatial,protocol["search"]["anchors_uniform_q_and_cutoff"]):
                assert all(x["cutoff_s"]==tau and all(x[k]==q for k in ("q_edge","q_inner","q_center")) for x in (u,s))
                assert u["feasible"]==s["feasible"] and abs(u["energy_J"]-s["energy_J"])<1e-8
    seed_ranges=[];rankings=[]
    for r in RESERVES:
        for budget in BUDGETS:
            best_by_family={}
            for family in FAMILIES:
                runs=[row for row in prefixes if (row["reserve_C"],row["budget"],row["family"])==(r,budget,family)]
                low=min(row["energy_J"] for row in runs);high=max(row["energy_J"] for row in runs)
                best_by_family[family]=low
                seed_ranges.append({"reserve_C":r,"budget":budget,"family":family,"min_J":low,"max_J":high,
                                    "spread_J":high-low})
            u,s=best_by_family["uniform"],best_by_family["spatial"]
            rank="tie" if math.isclose(u,s,abs_tol=1e-8) else ("uniform" if u<s else "spatial")
            rankings.append({"reserve_C":r,"budget":budget,"uniform_min_J":u,"spatial_min_J":s,
                             "spatial_minus_uniform_J":s-u,"lower_energy_family":rank})
    pool=[r for h in histories.values() for r in h]
    write_csv(HERE/"budget_prefixes.csv",prefixes)
    write_csv(HERE/"seed_ranges.csv",seed_ranges)
    write_csv(HERE/"rankings.csv",rankings)
    write_csv(HERE/"evaluation_labels.csv",pool)
    write_csv(HERE/"sequential_forecasts.csv",scores)
    by_family={fam:score_summary([row for row in scores if row["chain"].startswith(fam+"_")]) for fam in FAMILIES}
    overall=score_summary(scores)
    reps=load(rep_path)["representatives"]
    report={"protocol_sha256":protocol_hash,"n_evaluations":len(pool),
            "n_complete":sum(r['complete'] for r in pool),"n_feasible":sum(r['feasible'] for r in pool),
            "status_counts":dict(Counter(r["status"] for r in pool)),
            "n_adaptive":sum(r['evaluation']>8 for r in pool),
            "n_adaptive_unresolved":sum(r['evaluation']>8 and not r['resolved'] for r in pool),
            "n_fixed_initializations":sum(r['evaluation']<=8 for r in pool),
            "ordered_forecast_training_indices_and_incumbents_verified":True,
            "unresolved_excluded_from_outcomes":True,"prequential_scores":overall,"scores_by_family":by_family,
            "groups":rankings,"final_representatives":reps,
            "physical_nesting":"Each uniform schedule is admissible as a spatial-family command when all three power densities coincide. Four uniform anchors are shared within every pair of searches. The two Sobol sequences and the adaptive proposals are family-specific.",
            "scoring_scope":"Stored predictions on adaptive proposals only. Each reference forecast uses the previous resolved outcomes from the same chain and a Beta(1,1) prior. Margin scores condition on a complete trajectory with a defined margin. The joint score uses the product of the stored forecasts for admissible completion and the temperature margin. The sampled design ensemble is adaptive.",
            "new_simulator_integrations":0}
    dump(HERE/"summary.json",report)
    cap=paired("Lowest feasible heater energy on the search mesh at four cumulative evaluation budgets. The two initialized runs for each family and reserve are listed separately. Each entry includes all preceding evaluations in the same chain.",
        "四个累积评价预算下搜索网格上的最低可行辅助供热能量。每种供热方式和温度要求的两个搜索种子分别列示，各项均含该链此前的全部评价。")
    rows=[]
    for r in RESERVES:
        for fam in FAMILIES:
            for seed in SEEDS:
                values=[p for p in prefixes if (p["reserve_C"],p["family"],p["seed"])==(r,fam,seed)]
                vals=sorted(values,key=lambda x:x["budget"])
                rows.append([f"{r:g}",paired("Uniform","均匀") if fam=="uniform" else paired("Spatial","空间"),str(seed),*[f"{p['energy_J']/1000:.3f}" for p in vals]])
    budget_block={"type":"table","id":"Table S37","caption":cap,"headers":paired(
       ["Reserve (°C)","Heating","Seed","8 (kJ)","16 (kJ)","24 (kJ)","32 (kJ)"],
       ["温度要求（℃）","供热","种子","8次（kJ）","16次（kJ）","24次（kJ）","32次（kJ）"]),
       "widths":[1.2,.9,1.25,1.,1.,1.,1.],"rows":rows}
    score_rows=[]
    for task,en,zh in [("admissible_completion","Admissible completion","完整且满足状态约束"),
                       ("margin_given_completion","Thermal margin on completed runs","完整轨迹温度达标"),
                       ("feasibility","Overall feasibility","综合可行性")]:
        q=overall[task]
        score_rows.append([paired(en,zh),str(q['n']),str(q['positive']),f"{q['brier']:.3f}",f"{q['running_frequency_brier']:.3f}"])
    score_block={"type":"table","id":"Table S38","caption":paired(
       "Brier scores from probability forecasts stored before each adaptive evaluation. The reference probability is the running event frequency in the same search chain, with one success and one failure as prior counts. Lower scores indicate smaller squared probability error.",
       "自适应评价开始前保存的概率预测及其 Brier 评分。参照概率取同一条搜索链此前的事件发生频率，成功与失败各加入一个先验计数。评分越低，概率预测的均方误差越小。"),
       "headers":paired(["Predicted event","Scored","Positive","Model","Frequency reference"],["预测事件","计分数","阳性数","模型","频率参照"]),
       "widths":[2.65,.85,.85,.85,1.45],"rows":score_rows}
    dump(HERE/"si_blocks.json",{"budget":budget_block,"brier":score_block})
    dump(HERE/"verification.json",{"status":"PASS","checked_result_files":len(pool),
         "verified_recorded_adaptive_predictions":report["n_adaptive"],
         "script_sha256":digest(Path(__file__)),"inputs":inputs,
         "outputs":{p.name:digest(p) for p in [HERE/n for n in ["summary.json","si_blocks.json","budget_prefixes.csv","seed_ranges.csv","rankings.csv","evaluation_labels.csv","sequential_forecasts.csv"]]}})
    print(json.dumps({k:report[k] for k in ["n_evaluations","status_counts","n_feasible","n_adaptive_unresolved","prequential_scores","groups"]},indent=2))


if __name__ == "__main__":
    main()
