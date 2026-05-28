# Results and Discussion

## Corrected Evaluation Summary

After fixing the coded-evaluation path and removing stale uncoded summaries, the rate-distortion results changed materially from the earlier misleading plots. The corrected evaluation uses true coded bitrate from `compress()/decompress()` and only finite metric rows.

After the latest `rerun_eval.sh` pass, the results directory was cleaned so that only the newest summary pair for each unique `(variant, quality, lambda, freeze_mode)` combination remains. The plots and tables in `experiments/plots/` are therefore now derived from a single refreshed evaluation set rather than a mix of older reruns.

The main comparison baseline is the locally fine-tuned vanilla hyperprior model `A_ft`. Under the cleaned evaluation, the SE variants do not deliver a meaningful gain over this control baseline.

## Main Rate-Distortion Results

BD-rate is reported against `A_ft`, where negative values are better.

| Model | PSNR BD-rate | MS-SSIM BD-rate |
|---|---:|---:|
| Pretrained Baseline (`A`) | -0.11% | -1.62% |
| Encoder SE (`B`) | +0.08% | -0.15% |
| Encoder+Decoder SE (`C`) | +0.17% | -0.05% |
| BMSHJ 2018 Factorized | +22.23% | +1.45% |
| MBT 2018 Mean | -8.02% | -7.53% |
| MBT 2018 | -19.42% | -15.56% |
| Cheng 2020 Anchor | -24.87% | -20.61% |

The corrected interpretation is:

- `B` and `C` are effectively tied with `A_ft`.
- The SE modifications are too small to yield a convincing rate-distortion improvement.
- The strongest gains come from stronger entropy models, not from the lightweight SE augmentation.

## Runtime and Lightweight Tradeoff

The rerun also produced complete runtime measurements for all evaluated models. For the three main project variants under `freeze_mode=none`, the average Kodak timings are:

| Model | Params (M) | Forward ms | Encode ms | Decode ms | Codec total ms |
|---|---:|---:|---:|---:|---:|
| `A_ft` | 5.076 | 11.39 | 73.34 | 92.85 | 166.18 |
| `B` | 5.078 | 11.56 | 74.95 | 93.84 | 168.79 |
| `C` | 5.080 | 11.66 | 73.05 | 91.20 | 164.25 |

The parameter overhead remains tiny:

- `B` adds only `+0.043%` parameters over `A_ft`.
- `C` adds only `+0.086%` parameters over `A_ft`.

The runtime story is similarly modest:

- `B` is slightly slower than `A_ft` in codec total time (`+1.57%`).
- `C` is slightly faster than `A_ft` in codec total time (`-1.17%`).
- Forward-pass latency stays very close across the three models, with only a `1.44%` increase for `B` and a `2.33%` increase for `C`.

So the academically safe claim is that the SE additions are genuinely lightweight in both parameter count and runtime, but they still do not buy a meaningful rate-distortion improvement.

## Project Variant Curves

The cleaned operating points for the primary project comparison are:

| Variant | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| `A_ft` coded bpp | 0.1318 | 0.2114 | 0.3279 | 0.4885 |
| `A_ft` PSNR | 27.56 | 29.22 | 31.09 | 33.00 |
| `B` coded bpp | 0.1316 | 0.2113 | 0.3276 | 0.4881 |
| `B` PSNR | 27.56 | 29.21 | 31.09 | 33.00 |
| `C` coded bpp | 0.1317 | 0.2118 | 0.3273 | 0.4884 |
| `C` PSNR | 27.56 | 29.21 | 31.09 | 33.00 |

These values show that all three project variants occupy nearly the same RD curve. The observed differences are on the order of hundredths of a dB or tiny bitrate shifts, which is not strong enough to support a claim that SE attention improved compression performance.

## Ablation Findings

The freeze-mode ablations also do not reveal a meaningful gain:

- `A_ft`: `freeze_mode=none` is marginally better than `frozen_hyperprior`.
- `B`: `none`, `frozen_hyperprior`, and `attention_only` remain close, but none clearly outperforms `A_ft`.
- `C`: the same pattern holds; `attention_only` and `frozen_hyperprior` do not produce a robust advantage.

This suggests that the proposed SE insertion is not the bottleneck that limits performance. The dominant factor remains the entropy model quality rather than the addition of lightweight channel attention.

## Comparison to Reference Models

The reference models behave as expected:

- `BMSHJ 2018 Factorized` is clearly weaker than the hyperprior baseline.
- `MBT 2018 Mean` improves over `A_ft`.
- `MBT 2018` produces a substantial gain.
- `Cheng 2020 Anchor` is the strongest reference among the tested models.

This ranking is consistent with the learned image compression literature and supports the conclusion that better priors and context modeling provide materially larger gains than the SE modification tested in this project.

## Final Interpretation

The corrected experimental outcome is technically coherent but less favorable to the original hypothesis than the earlier broken evaluation suggested.

- The implementation now evaluates models correctly using coded bitrate.
- The vanilla fine-tuned baseline `A_ft` was a necessary control and remains the fairest comparison point.
- The SE variants are functional, but they do not produce a significant RD improvement.
- The strongest lesson from the experiments is that stronger entropy models such as `MBT 2018` and `Cheng 2020 Anchor` matter much more than the lightweight SE augmentation.

Therefore, the academically safe conclusion is:

> Under the corrected evaluation protocol, SE-based encoder or encoder+decoder channel attention does not provide a meaningful rate-distortion improvement over the fine-tuned scale-hyperprior baseline on Kodak.
