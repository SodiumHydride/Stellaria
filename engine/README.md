# engine/ — DDSP-SVC

The singing voice conversion engine Stellaria runs: [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
6.5 by yxlllc, under its own MIT License ([`LICENSE`](LICENSE)), not Stellaria's.

Only the sources the app needs are here: the scripts and `ddsp/`, `reflow/`, `encoder/`,
`nsf_hifigan/`, `optimizer/`, `logger/`, `configs/`. Pretrained weights are downloaded by the
app (see `resources/manifest.json`). `scripts/stage-bundle.mjs` copies this folder into the
installers.

## Changes from upstream

- `optimizer/muon.py` imports the Triton Gram-NS kernel lazily, so training works without
  Triton when fused kernels are off.
- `train_reflow.py` turns fused kernels off (with a message) instead of crashing when the
  device is not CUDA or Triton is unavailable.
- `reflow/solver.py` stops gracefully, saving a checkpoint first, when a `STOP` file appears in
  the experiment directory or `train.max_steps` is reached.
- `ddsp/vocoder.py` lets `DDSP_SVC_RMVPE_PATH` override the RMVPE checkpoint path, and loads
  encoder checkpoints with `map_location='cpu'`; `encoder/rmvpe/inference.py` does the same.
- `configs/reflow.yaml` adds `train.max_steps` (null: no limit).
