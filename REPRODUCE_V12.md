### Endplate storage and finer mesh checks

The [review data](results/review_checks/) include simulated endplate temperature profiles, boundary heat powers and refined-grid thermal traces. Recompute the plate storage, the decay of the end-to-centre temperature contrast and the ten continuous window minima with `python research/revision_v12/check_results.py`. This calculation reads the published five-cell simulations and the earlier spatial comparisons in `results/spatial/all_cases.csv`.

To integrate a case again, run `python research/revision_v12/run_refined.py --case uniform_1_back_loaded --output reruns/grid8_back`. Omit `--case` to run all ten controls. The original meshes had 232 and 464 thermal nodes; the new mesh has 928. The `mesh_minima.csv` file contains one row per control and tested mesh. The endplate profile and heat-power data are predictions over 38.0–55.5 s following the first 1 °C cutoff. After regenerating the full matched-cutoff runs, use `python research/revision_v12/extract_plate_profiles.py` to recreate the small sensor-design inputs.

