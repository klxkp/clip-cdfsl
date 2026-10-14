# CLIP-CDFSL

A reproducible **frozen CLIP image-feature baseline** for episodic cross-domain few-shot classification on four target-domain datasets: **CropDisease, EuroSAT, ISIC, and ChestX**.

This repository is based on [Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-Shot Difficulty](https://github.com/sungnyun/understanding-cdfsl) (NeurIPS 2022). The current CLIP evaluation entry point is `1eval_clip.py`.

## Overview

The implemented baseline uses a pretrained OpenAI CLIP model as a **frozen visual feature extractor** and evaluates a cosine-prototype classifier within each few-shot episode.

- **Backbone:** pretrained OpenAI CLIP (ViT-B/32 in the reported run)
- **Target datasets:** CropDisease, EuroSAT, ISIC, ChestX
- **Evaluation:** N-way K-shot episodes; 5-way 1-shot and 5-way 5-shot by default
- **Classifier:** normalized support-set class prototypes with cosine-similarity prediction
- **Adaptation:** no gradient updates, fine-tuning, or task-specific training
- **Text encoder:** not used by the current baseline
- **Data protocol:** original `split_seed_1` metadata; episodes sampled from `labeled_80`
- **Execution:** local CLIP source and checkpoint; no model download during evaluation

**Scope:** This is a CLIP-based baseline evaluated under a cross-domain few-shot task setting, **not** a trained CLIP vision-language adaptation method. The current script does not implement zero-shot text-prompt classification, CoOp, Tip-Adapter, LoRA, or a new cross-domain adaptation algorithm. It does not retrain CLIP on ImageNet or miniImageNet.

## Evaluation Method

For a target-domain episode, `1eval_clip.py`:

1. Samples `N` classes with `K` labeled support images and `Q` query images per class.
2. Extracts L2-normalized image features with the frozen CLIP image encoder.
3. Averages the support features of each class and L2-normalizes the resulting class prototype.
4. Classifies each query by its cosine similarity to the class prototypes.
5. Repeats across episodes and reports mean accuracy with a 95% confidence interval.

For class `c` and query image `x`, the decision rule is:

```text
z(x) = normalize(CLIP_image_encoder(x))
p(c) = normalize(mean(z(s) for s in support_images_of_class_c))
prediction(x) = argmax_c dot(z(x), p(c))
```

The support labels define the episode classes; **no class-name text prompts are passed to CLIP**.

## Repository Components

```text
clip-cdfsl/
├── 1eval_clip.py           # Frozen CLIP episodic evaluation
├── configs.py              # Dataset root paths
├── datasets/
│   ├── datasets.py         # Dataset loaders
│   ├── sampler.py          # Support/query episode sampling
│   ├── split.py            # Original labeled/unlabeled split handling
│   └── split_seed_1/       # Expected split CSV metadata
├── requirements.txt
├── LICENSE
└── README.md
```

Additional files inherited from the upstream CDFSL repository may be present. The CLIP results below are produced by `1eval_clip.py`, not the upstream ResNet training pipeline.

## Requirements

- Python and a compatible PyTorch / torchvision environment
- Dependencies required by this repository and the local OpenAI CLIP package
- An available CUDA GPU (recommended) or CPU
- Local [OpenAI CLIP](https://github.com/openai/CLIP) source code and a compatible pretrained checkpoint
- Prepared target datasets and matching split CSV files

The repository's `requirements.txt` comes from the upstream project. Use a compatible, validated environment rather than assuming its pinned versions match every CUDA/Python installation.

The evaluation script expects:

```text
/y-stu01/clip_test/
├── clip/
│   └── __init__.py
└── models/
    └── ViT-B-32.pt
```

These are example paths from the reported server run. The script accepts the actual paths through `--clip-repo` and `--checkpoint`.

## Dataset Preparation

Configure the dataset roots in `configs.py`. The current CLIP evaluator supports the following four target domains:

| Dataset | Content | Expected data root (default configuration) |
| --- | --- | --- |
| CropDisease | Plant disease photographs | `/data/cdfsl/CropDiseases` (images under `dataset/train/`) |
| EuroSAT | Satellite land-cover images | `/data/cdfsl/EuroSAT` |
| ISIC | Dermoscopic skin-lesion images | `/data/cdfsl/ISIC` |
| ChestX | Chest X-ray images | `/data/cdfsl/chestX` (images under `images/`) |

ISIC also requires `ISIC2018_Task3_Training_GroundTruth.csv` in its data root. ChestX requires `Data_Entry_2017.csv` in its data root. See the upstream [CDFSL repository](https://github.com/sungnyun/understanding-cdfsl) and [BSCD-FSL benchmark](https://github.com/IBM/cdfsl-benchmark) for the original data sources and preparation conventions.

The evaluator uses the existing split metadata in:

```text
datasets/split_seed_1/
├── CropDisease_labeled_80.csv
├── CropDisease_unlabeled_20.csv
├── EuroSAT_labeled_80.csv
├── EuroSAT_unlabeled_20.csv
├── ISIC_labeled_80.csv
├── ISIC_unlabeled_20.csv
├── ChestX_labeled_80.csv
└── ChestX_unlabeled_20.csv
```

**Only `labeled_80` images are eligible for support/query sampling.** The `unlabeled_20` files are checked for split consistency, but those images are not used to train or adapt the model. The evaluator checks that the stored split metadata matches the prepared dataset files.

## Run Evaluation

From the repository root, run:

```bash
python 1eval_clip.py \
  --clip-repo /y-stu01/clip_test \
  --checkpoint /y-stu01/clip_test/models/ViT-B-32.pt \
  --datasets CropDisease EuroSAT ISIC ChestX \
  --shots 1 5 \
  --way 5 \
  --query 15 \
  --episodes 600 \
  --batch-size 128 \
  --workers 4
```

The command evaluates each of the four target domains in both 5-way 1-shot and 5-way 5-shot settings. Each setting uses **600 episodes** and **15 query images per class**. By default, `--seed=0`, `--episode-seed=0`, `--device=cuda`, and `--precision=fp32`.

### Repeat With Different Episode Seeds

To run the same evaluation three times while keeping the original dataset split unchanged:

```bash
for s in 0 1 2; do
  python 1eval_clip.py \
    --clip-repo /y-stu01/clip_test \
    --checkpoint /y-stu01/clip_test/models/ViT-B-32.pt \
    --datasets CropDisease EuroSAT ISIC ChestX \
    --shots 1 5 \
    --way 5 \
    --query 15 \
    --episodes 600 \
    --batch-size 128 \
    --workers 4 \
    --seed "$s" \
    --episode-seed "$s"
done
```

Here, `--episode-seed` controls episodic class/image sampling, whereas `--seed` controls Python, NumPy, PyTorch, and data-loader randomness. The dataset partition remains the original `split_seed_1` in all three runs. Each run creates a separate output directory.

For the full list of options:

```bash
python 1eval_clip.py --help
```

## Baseline Results

Results from **one completed server run**, using **OpenAI CLIP ViT-B/32**, the original `split_seed_1` / `labeled_80` partition, and the default evaluation seeds (`--seed=0`, `--episode-seed=0`):

| Target dataset | 5-way 1-shot | 5-way 5-shot |
| --- | ---: | ---: |
| CropDisease | 72.22 ± 0.91 | 90.66 ± 0.52 |
| EuroSAT | 60.83 ± 0.90 | 78.44 ± 0.57 |
| ISIC | 28.74 ± 0.50 | 38.82 ± 0.57 |
| ChestX | 21.15 ± 0.35 | 22.75 ± 0.41 |

All values are **accuracy (%) ± 95% confidence-interval half-width** over 600 episodes per setting, not the standard deviation across three runs. The confidence interval is computed as:

```text
95% CI half-width = 1.96 × sample_std(episode_accuracy) / sqrt(num_episodes)
```

For a 5-way task, chance accuracy is 20%. These results describe this specific frozen-feature baseline and evaluation protocol; direct comparisons with published methods require matching their evaluation settings.

## Outputs and Reproducibility

By default, each run writes to a new timestamped directory:

```text
logs/clip/ViT-B-32_<UTC_timestamp>/
├── run.json
├── summary.csv
├── CropDisease_samples.csv
├── CropDisease_1shot_indices.npz
├── CropDisease_1shot_episodes.csv
├── CropDisease_5shot_indices.npz
├── CropDisease_5shot_episodes.csv
└── ...                         # Equivalent outputs for the other datasets
```

- `summary.csv`: per-dataset, per-shot mean accuracy, episode standard deviation, 95% CI, and overlap count.
- `*_episodes.csv`: per-episode correctness and accuracy.
- `*_indices.npz`: sampled support/query indices and selected class IDs.
- `*_samples.csv`: sample-index to image-path/label mapping.
- `run.json`: arguments, environment information, checkpoint and source-file hashes, dataset metadata, and completion status.

The script checks for overlapping support/query indices within each episode. The default output directory is never overwritten; a custom fresh directory can be specified with `--output`.

## Project Scope

This branch currently provides a **visual-only frozen CLIP baseline** for few-shot classification across target domains. It does **not** use CLIP's text encoder or optimize prompts, adapters, or LoRA parameters. Support images are used to construct class prototypes, not to update the model.

The code can serve as a common evaluation starting point for future CLIP-based few-shot and cross-domain few-shot experiments. Any future methods and their results should be documented separately from this baseline.

## Acknowledgments and Citation

This repository is derived from the official NeurIPS 2022 implementation of:

> Jaehoon Oh, Sungnyun Kim, Namgyu Ho, Jin-Hwa Kim, Hwanjun Song, and Se-Young Yun. **Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-Shot Difficulty.** NeurIPS, 2022.

Original repository: https://github.com/sungnyun/understanding-cdfsl

The evaluation uses the pretrained CLIP model introduced in:

> Alec Radford et al. **Learning Transferable Visual Models From Natural Language Supervision.** ICML, 2021.

CLIP implementation: https://github.com/openai/CLIP

The four target domains follow the data ecosystem introduced by:

> Yunhui Guo et al. **A Broader Study of Cross-Domain Few-Shot Learning.** ECCV, 2020.

BSCD-FSL benchmark: https://github.com/IBM/cdfsl-benchmark

Dataset ownership, licenses, and usage conditions remain with their respective providers. Refer to the upstream repository for dataset attributions.

## License

This repository retains the original MIT license. See [LICENSE](LICENSE) for details.