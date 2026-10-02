# engine/ — DDSP-SVC

The singing voice conversion engine Stellaria runs: [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
6.5 by yxlllc, under its own MIT License ([`LICENSE`](LICENSE)), not Stellaria's.

This copy carries Stellaria's engine fixes, made on the `6.5` branch of
[SodiumHydride/DDSP-SVC](https://github.com/SodiumHydride/DDSP-SVC/tree/6.5) (a fork of the
original). Only the sources the app needs are here: the scripts and `ddsp/`, `reflow/`,
`encoder/`, `nsf_hifigan/`, `optimizer/`, `logger/`, `configs/`. Pretrained weights are
downloaded by the app (see `resources/manifest.json`).

Changes to the engine go here; `scripts/stage-bundle.mjs` copies it into the installers.
