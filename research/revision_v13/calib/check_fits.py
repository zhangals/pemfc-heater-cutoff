"""Verify the calibration objectives and reported trial selections from public summaries."""
from collections import defaultdict
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv(name):
    with (HERE/name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def loss(voltage_rmse, temperature_rmse):
    return (float(voltage_rmse)/0.02)**2 + (float(temperature_rmse)/0.5)**2


def main():
    protocol=json.loads((HERE/"fit_protocol.json").read_text(encoding="utf-8"))
    fits=read_csv("joint_fit_trials.csv")
    assert len(fits)==40
    for row in fits:
        value=sum(loss(row[prefix+"_voltage_rmse_V"],row[prefix+"_temperature_rmse_K"])
                  for prefix in ("m20","m25"))
        assert abs(value-float(row["objective"]))<1e-12
    index=protocol["historical_joint"]["selection_snapshot_evaluations"]
    joint=min(fits[:index],key=lambda row:float(row["objective"]))
    assert int(joint["evaluation"])==protocol["historical_joint"]["selected_evaluation"]
    assert all(float(joint[key])==val for key,val in protocol["reference"].items())
    singles=read_csv("single_fit_trials.csv")
    groups=defaultdict(list)
    for row in singles:
        assert row["status"]=="COMPLETE"
        assert abs(loss(row["voltage_rmse_V"],row["temperature_rmse_K"])-float(row["objective"]))<1e-12
        groups[int(row["temperature_C"]),int(row["start"])].append(row)
    assert len(groups)==4 and all(len(rows)==48 for rows in groups.values())
    winners={key:min(rows,key=lambda row:float(row["objective"])) for key,rows in groups.items()}
    print("Joint candidate",joint["evaluation"],"of",index,"in the selection snapshot")
    for temp in (-20,-25):
        win=min((winners[temp,s] for s in (1,2)),key=lambda row:float(row["objective"]))
        print(temp, "deg C, start",win["start"],"evaluation",win["evaluation"],"loss",win["objective"])


if __name__=="__main__":
    main()
