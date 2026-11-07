# Reproducing the calculations

[Version 1.0.0 energy, boundary and calibration checks](REPRODUCE_V13.md).

Tests used CPython 3.13.3. Run commands from the repository root in a Python environment with the packages in `requirements.txt` installed. The first solve compiles the numerical kernels. To vary the example, copy `case.json`, edit the copy and run `python run_example.py --case your_case.json`. The expected baseline is in `example_output.json`.

| Paper material | Code | Numerical output |
|---|---|---|
| Fig. 1 geometry and imposed load | `python research/plots/figure1.py` | Geometry and protocol in the physics module and `research/study_v2/protocol.json` |
| Fig. 2 recooling and regional heat balances | `python research/plots/figure2.py` | `results/figure2_post_cutoff.csv`, `results/figure2_heat_balance.json` |
| Fig. 3 matched heater cutoffs | `python research/plots/figure3.py` | `results/matched_cutoff.csv`, fine-mesh dense temperature coefficients in the two NPZ files under `results/` |
| Fig. 4 endplate capacity | `python research/plots/figure4.py` | `results/figure4_capacity_scan.csv`, `results/figure4_capacity_traces.csv` |
| Heating schedules, mesh and waveform comparisons | `research/study_v2/search.py`, `run_followups.py` | `results/stack/` |
| Window start, reserve and mesh sensitivity | `research/revision_v10_reserve/window_grid/recompute_windows.py` | `results/window_grid/` |
| Search-budget and forecast scoring | `research/revision_v10_reserve/search_budget/analyze_budget.py` | `results/search_budget/` |
| Matched spatial heating | `research/revision_v10_reserve/spatial/run_matched.py`, then `confirm_boundary.py` | `results/spatial/all_cases.csv` |
| Single-cell fit, cross-temperature predictions and nominal replays | `research/revision_v9_current/numerics/run_recorded.py` | `results/single_cell_fit_metrics.csv` (192 training integrations, eight selected-parameter assessments, four nominal runs) |

The four plotting commands generate images in `regenerated_figures/`. Published stack results use simulated five-cell trajectories. In the matched full-power comparison, switching off at 37.85 s gives a late-window minimum of −0.248 °C. Extending heating to 41.25 s gives 1.567 °C. Table 2's 1 °C uniform case in `run_example.py` uses a lower power and a longer pulse.

## Running larger calculations

For a new set of runs use a fresh working copy. The model code is in `代码/`; `research/study_v2/engine.py` assembles the stack and records accepted BDF temperature interpolants.

1. Run `python research/study_v2/search.py --workers 2` to produce the search chains, then `python research/study_v2/run_followups.py --phase all --workers 2` for the baseline, validation, perturbation and waveform jobs. Stopped or unresolved trajectories retain their recorded status.
2. After those runs, the budget and window scripts in the table recompute their published CSVs from the newly produced records. The spatial experiment starts directly from the selected control in `research/study_v2/representatives.json`.
3. For the endplate capacity sweep run `python research/study_v3_diagnostics/run_design_recovery.py --freeze`, then the same script with `--workers 2`. Run `python research/study_v3_diagnostics/refine_design_recovery.py` for the refined cutoffs. With the sweep and refined runs in place, `python research/revision_v10_reserve/matched_cutoff/run.py` calculates the first-crossing pair.

## Single-cell input

The published single-cell fit requires the original Problem B Attachment 2 from the 2026 China Post-Graduate Mathematical Contest in Modeling. Access requests can be directed to the organizer through the [competition portal](https://cpipc.acge.org.cn/cw/hp/4). Place the workbook at `local_inputs/attachment_2.xlsx` and run `python research/revision_v9_current/numerics/run_recorded.py` from a fresh copy. The reader uses the two original temperature sheets with time in column A, cell voltage in C, mean temperature in D and current density in E. The first two rows contain labels, followed by 184 samples per sheet. The public results give fitted parameters and scalar fit errors. The original workbooks and row-level observations must be obtained separately.

The bundled [stack time series](results/stack_time_series.zip) contain 600 numerical run stages, with 158,433 stored time points. Each record supplies time in seconds and the five cell-mean temperatures (°C) and voltages (V). The archive index identifies complete and stopped runs. [Checksums and counts](results/stack_time_series_manifest.json) are provided for verification.

The calibration check uses two fitted parameter vectors in the matched five-cell comparison. Run `python research/revision_v11/calibration_sensitivity/run.py --output reruns/calibration`. Results on both meshes and all eight simulated temperature series are in `results/cutoff_sensitivity/`.

The rolling-window eligibility calculation and limiting-cell heat budgets are in `results/diagnostics/`. Use `python results/diagnostics/analyze_diagnostics.py --help` for the table-building inputs.
