Output directory: /home/robert/dev/glades-ml/artifacts/geode_gpu_gate_20260411-070514

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/epoch point
- acceptance_summary.tsv: optional 10-repeat summary rows when --acceptance is used
- raw/*.log: full raw benchmark outputs

Quick comparison examples:

  column -t -s $'\t' "/home/robert/dev/glades-ml/artifacts/geode_gpu_gate_20260411-070514/epoch_sweep_summary.tsv"

  rg '^token-lm-document' "/home/robert/dev/glades-ml/artifacts/geode_gpu_gate_20260411-070514/epoch_sweep_summary.tsv" | column -t -s $'\t'

  rg '^token-lm-corpus-large' "/home/robert/dev/glades-ml/artifacts/geode_gpu_gate_20260411-070514/epoch_sweep_summary.tsv" | column -t -s $'\t'
