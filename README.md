# Attention-Augmented Scale Hyperprior for Learned Image Compression

CS566 Deep Learning — Spring 2026

This project adds lightweight Squeeze-and-Excitation (SE) channel attention blocks to the Ballé et al. scale hyperprior model and evaluates rate-distortion performance on the Kodak benchmark. The training default is CLIC images because CLIC is built for image compression research. The evaluation pipeline also supports several pretrained CompressAI reference models for direct Kodak comparisons.

## Setup

```bash
conda activate learned-image-compression
pip install -r requirements.txt
```

## Project Structure

```
├── README.md
├── train.py                   # Training script for variants A, B, C
├── evaluate.py                # Kodak evaluation (PSNR, MS-SSIM, bpp, BD-rate)
├── plot.py                    # Rate-distortion curve plotting
├── run_all.sh                 # End-to-end experiment pipeline
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── se_hyperprior.py       # SEScaleHyperprior model + weight transfer
│   └── se_block.py            # Squeeze-and-Excitation module
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # PSNR, MS-SSIM, BD-rate
│   └── datasets.py            # CLIC and Kodak dataloaders + download helpers
├── configs/
│   └── baseline.yaml          # Lambda values, lr, batch size, paths
└── experiments/
    └── results/               # Evaluation CSV/JSON outputs and plots
```

## Variants

| Variant | Description |
|---------|-------------|
| A | Pretrained CompressAI scale hyperprior baseline |
| B | Encoder-only SE attention |
| C | Encoder+decoder SE attention |

## Pretrained Comparison Models

These models can be evaluated directly on Kodak using official CompressAI pretrained weights. They do not require local training checkpoints.

| Variant name | Description |
|--------------|-------------|
| `bmshj2018_factorized` | Ballé et al. factorized-prior model |
| `mbt2018_mean` | Minnen et al. mean-scale hyperprior model |
| `mbt2018` | Minnen et al. joint autoregressive + hierarchical prior model |
| `cheng2020_anchor` | Cheng et al. anchor model |

## Quick Start

Run the full pipeline:

```bash
bash run_all.sh
```

Or run individual steps:

```bash
# 1. Check datasets
find datasets/clic_images -type f | head
find datasets/kodak -type f | head

# 2. Train (example: variant C, lambda=0.0067)
python train.py --variant C --lmbda 0.0067 --quality 3 \
    --data-dir datasets/clic_images --epochs 100

# 3. Evaluate on Kodak
python evaluate.py --variant C --lmbda 0.0067 \
    --checkpoint "$(cat checkpoints/latest_variant_C_lmbda_0.0067_quality_3_mse.txt)" \
    --data-dir datasets/kodak

# 4. Evaluate a pretrained comparison model on Kodak
python evaluate.py --variant mbt2018 --quality 3 --lmbda 0.0067 \
    --data-dir datasets/kodak --output experiments/results \
    --run-name variant_mbt2018_lmbda_0.0067_quality_3_mse_manual

# 5. Plot RD curves (after evaluating all variants and lambdas)
python plot.py --results-dir experiments/results --output experiments/plots
```

Before a full run, execute the smoke test:

```bash
python smoke_test.py
```

It checks dataset loading, model construction, one forward pass, one optimizer step for attention variants, tiny Kodak evaluation, and plot generation.

For a tiny timed training check with TensorBoard logging:

```bash
python train.py --variant B --lmbda 0.0018 --quality 1 \
    --data-dir datasets/clic_images --epochs 1 --batch-size 1 \
    --patch-size 64 --num-workers 0 \
    --max-train-samples 2 --max-val-samples 2 \
    --save-dir /tmp/lc_log_check
```

## Training Details

- **Training Dataset**: CLIC images, 256x256 random crops
- **Evaluation**: Kodak PhotoCD (24 images, 768x512)
- **Lambda values**: 0.0018, 0.0035, 0.0067, 0.013
- **CompressAI quality levels**: 1, 2, 3, 4 paired with the lambda values for RD operating points
- **Optimizer**: Adam, lr=1e-4, ReduceLROnPlateau (factor=0.1, patience=10)
- **Batch size**: 8
- **Distortion**: MSE (primary), MS-SSIM (supplementary via `--distortion msssim`)

## Kodak Evaluation Variants

`evaluate.py` supports both trained project variants and pretrained CompressAI zoo models:

- Trained/local variants: `A`, `B`, `C`
- Pretrained comparison variants: `bmshj2018_factorized`, `mbt2018_mean`, `mbt2018`, `cheng2020_anchor`

For pretrained comparison variants, do not pass `--checkpoint`.

## Training Logs

Training writes TensorBoard logs under each checkpoint folder, including loss, bpp, PSNR, distortion, auxiliary entropy loss, gradient norm, learning rates, epoch time, seconds per step, GPU memory, and GPU utilization when `nvidia-smi` is available.

Run directories are timestamped by default:

```text
checkpoints/variant_B_lmbda_0.0018_quality_1_mse_20260430_153000/
```

Each run also writes a stable latest-checkpoint pointer:

```text
checkpoints/latest_variant_B_lmbda_0.0018_quality_1_mse.txt
```

TensorBoard can read all runs with:

```bash
tensorboard --logdir checkpoints
```

## Resuming

Training writes `checkpoint_last.pth.tar` every epoch. For a single run:

```bash
python train.py --variant B --lmbda 0.0018 --quality 1 \
    --data-dir datasets/clic_images --epochs 100 \
    --resume checkpoints/variant_B_lmbda_0.0018_quality_1_mse_20260430_153000/checkpoint_last.pth.tar
```

For the full pipeline, restart with the same timestamp:

```bash
RUN_TIMESTAMP=20260430_153000 bash run_all.sh
```

The script will resume B/C runs that already have `checkpoint_last.pth.tar`.

## Architecture

Variant B inserts one SE block after the second convolution in the analysis transform `g_a`:

```
Baseline g_a:
  Conv(3→128) → GDN → Conv(128→128) → GDN → Conv(128→128) → GDN → Conv(128→192)

Encoder SE g_a:
  Conv(3→128) → GDN → Conv(128→128) → SE(128) → GDN → Conv(128→128) → GDN → Conv(128→192)
```

Variant C uses the encoder SE block above and also inserts one SE block after the second transposed convolution in the synthesis transform `g_s`:

```
Baseline g_s:
  DeconvT(192→128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→3)

Encoder+decoder SE g_s:
  DeconvT(192→128) → IGDN → DeconvT(128→128) → SE(128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→3)
```

Each SE block adds 2,184 parameters at quality 3.

## References

1. Ballé et al. (2018). Variational image compression with a scale hyperprior. ICLR 2018.
2. Hu et al. (2018). Squeeze-and-excitation networks. CVPR 2018.
3. Minnen et al. (2018). Joint autoregressive and hierarchical priors for learned image compression. NeurIPS 2018.
4. Cheng et al. (2020). Learned image compression with discretized Gaussian mixture likelihoods and attention modules. CVPR 2020.
5. CompressAI: https://github.com/InterDigitalInc/CompressAI
