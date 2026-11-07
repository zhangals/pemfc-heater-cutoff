"""One exploratory paired confirmation, after the initial matched-energy and shortened runs."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import run_matched as experiment

OUT=Path(__file__).resolve().parent

def main():
    baseline=experiment.read(OUT/'metrics.json')
    winner=baseline['selected_for_shortening']
    assert winner
    inputs={'base_metrics_sha256':experiment.eng.digest(OUT/'metrics.json'),
            'base_protocol_sha256':experiment.eng.digest(OUT/'protocol.json'),
            'confirmation_script_sha256':experiment.eng.digest(Path(__file__))}
    protocol={'frozen_utc':experiment.eng.utc_now(),'stage':'exploratory targeted boundary check',
        'purpose':'A paired same-cutoff comparison on the two meshes at 0.25 s earlier than the selected original cutoff.',
        'selection_basis':'At 0.125 s earlier both waveforms pass. At 0.5 s earlier the selected regional waveform fails. Interpolation of the observed local response places 0.25 s close to the 1 C threshold, making it informative for a paired check. This cutoff was chosen after seeing the preceding results.',
        'allocation_comparison':['uniform',winner],'meshes':[2,4],'shortening_s':.25,'additional_runs':4,
        'end_rule':'Four cases only. Record all outcomes. No optimum inferred.', 'inputs':inputs}
    experiment.new(OUT/'confirmation_protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=experiment.run_batch(pool,[(name,r,.25) for name in ['uniform',winner] for r in (2,4)])
    report={'confirmation_protocol_sha256':experiment.eng.digest(OUT/'confirmation_protocol.json'),'cases':rows}
    experiment.new(OUT/'confirmation.json',report)
    experiment.new(OUT/'all_results.json',dict(baseline,exploratory_confirmation=report))
if __name__=='__main__':
    main()

