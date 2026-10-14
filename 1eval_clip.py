"""Offline frozen-CLIP cosine-prototype evaluation on the original CDFSL splits."""

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import configs
from datasets.datasets import dataset_class_map
from datasets.sampler import EpisodeSampler
from datasets.split import split_dataset


TARGETS = ("CropDisease", "EuroSAT", "ISIC", "ChestX")
PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local OpenAI CLIP checkpoint; model names/URLs are not accepted.")
    parser.add_argument("--clip-repo", type=Path, default=Path("/y-stu01/clip_test"),
                        help="Directory containing the OpenAI CLIP package clip/__init__.py.")
    parser.add_argument("--datasets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--query", type=int, default=15, help="Query images per class.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Python, NumPy, PyTorch and loader seed.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path,
                        help="New output directory; existing directories are never overwritten.")
    args = parser.parse_args()
    if min(args.shots + [args.way, args.query, args.batch_size]) < 1:
        parser.error("Shots, way, query and batch size must be positive.")
    if args.episodes < 2 or args.workers < 0:
        parser.error("At least two episodes and nonnegative workers are required.")
    if not 0 <= args.seed < 2 ** 32 or not 0 <= args.episode_seed < 2 ** 32:
        parser.error("Seeds must be in [0, 2**32).")
    if len(set(args.shots)) != len(args.shots) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset and shot lists must not contain duplicates.")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_clip(args, device):
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    repo = args.clip_repo.expanduser().resolve(strict=True)
    if not checkpoint.is_file() or not (repo / "clip" / "__init__.py").is_file():
        raise FileNotFoundError("Expected a local checkpoint and <clip-repo>/clip/__init__.py.")
    sys.path.insert(0, str(repo))
    try:
        clip = importlib.import_module("clip")
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the local OpenAI CLIP package. Check its files and dependencies "
            "(ftfy, regex, packaging, tqdm, Pillow). No packages are installed by this script."
        ) from exc
    if not Path(clip.__file__).resolve().is_relative_to(repo / "clip"):
        raise RuntimeError(f"Imported an unexpected clip package: {clip.__file__}")
    if not callable(getattr(clip, "load", None)):
        raise RuntimeError("The local clip package must expose the OpenAI CLIP load() API.")
    print(f"CLIP source: {clip.__file__}\nCheckpoint: {checkpoint}", flush=True)
    # An absolute file path selects the offline branch of OpenAI CLIP's loader.
    model, preprocess = clip.load(str(checkpoint), device=str(device), jit=False)
    model = model.float() if args.precision == "fp32" else model.half()
    model.eval()
    model.requires_grad_(False)
    return model, preprocess, checkpoint, repo


def split_key(path, name):
    return path.removesuffix(".jpg") if name == "ISIC" else path


def prepare_dataset(name, preprocess, args, output):
    dataset = dataset_class_map[name](transform=preprocess)
    split_dir = PROJECT_DIR / "datasets" / "split_seed_1"
    split_paths = [split_dir / f"{name}_{suffix}.csv"
                   for suffix in ("unlabeled_20", "labeled_80")]
    split_sets = []
    for path in split_paths:
        with path.open(newline="", encoding="utf-8") as stream:
            entries = [split_key(row["img_path"], name) for row in csv.DictReader(stream)]
        if not entries or len(entries) != len(set(entries)):
            raise ValueError(f"Empty or duplicate entries in split file: {path}")
        split_sets.append(set(entries))
    if split_sets[0] & split_sets[1]:
        raise ValueError(f"Overlapping labeled/unlabeled split metadata for {name}.")

    _, labeled = split_dataset(dataset, ratio=20, seed=1)
    relative_paths = [os.path.relpath(path, labeled.root) for path, _ in labeled.samples]
    actual = {split_key(path, name) for path in relative_paths}
    expected = split_sets[1]
    if actual != expected or len(labeled) != len(expected):
        missing = sorted(expected - actual)
        raise ValueError(
            f"{name}: labeled split does not match dataset files/labels. "
            f"Expected {len(expected)}, matched {len(labeled)}. Missing examples: {missing[:5]}"
        )
    missing_files = [path for path, _ in labeled.samples if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{name}: missing images, e.g. {missing_files[:5]}")
    counts = Counter(int(label) for _, label in labeled.samples)
    if set(counts) != set(range(len(labeled.classes))) or len(counts) < args.way:
        raise ValueError(f"{name}: missing classes or fewer classes than --way: {dict(counts)}")
    # ISIC's numeric labels follow CSV columns, while its classes attribute is sorted.
    class_names = (labeled.metadata.columns[1:].tolist() if name == "ISIC"
                   else list(labeled.classes))
    manifest = output / f"{name}_samples.csv"
    write_csv(manifest, ["sample_index", "relative_path", "label", "class_name"], (
        {"sample_index": index, "relative_path": path, "label": int(label),
         "class_name": class_names[int(label)]}
        for index, (path, (_, label)) in enumerate(zip(relative_paths, labeled.samples))
    ))
    metadata = {
        "dataset_root": str(Path(labeled.root).resolve()),
        "full_eligible_samples": len(dataset), "labeled_samples": len(labeled),
        "class_counts": dict(sorted(counts.items())), "class_names": class_names,
        "split_sha256": {path.name: sha256(path) for path in split_paths},
        "sample_manifest_sha256": sha256(manifest),
    }
    return labeled, metadata


def sample_episodes(dataset, args):
    episodes = {}
    for shot in args.shots:
        sampler = EpisodeSampler(dataset, args.way, shot, args.query,
                                 args.episodes, seed=args.episode_seed)
        support, query = zip(*(sampler[index] for index in range(args.episodes)))
        episodes[shot] = (np.stack(support), np.stack(query))
    used_indices = np.unique(np.concatenate([
        indices.reshape(-1) for pair in episodes.values() for indices in pair
    ]))
    return episodes, used_indices


@torch.inference_mode()
def extract_features(model, dataset, used_indices, args, device):
    loader = DataLoader(
        Subset(dataset, used_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )
    features = []
    for images, _ in tqdm(loader, desc=f"{dataset.name}: encode", unit="batch"):
        images = images.to(device, non_blocking=device.type == "cuda")
        encoded = model.encode_image(images).float()
        if not torch.isfinite(encoded).all() or (encoded.norm(dim=-1) <= 1e-12).any():
            raise ValueError(f"{dataset.name}: non-finite or zero image features.")
        features.append(F.normalize(encoded, dim=-1).cpu().numpy())
    return np.concatenate(features, axis=0)


def evaluate(dataset, features, used_indices, episodes, args, output):
    feature_rows = np.full(len(dataset), -1, dtype=np.int64)
    feature_rows[used_indices] = np.arange(len(used_indices))
    labels = np.asarray(dataset.targets, dtype=np.int64)
    summaries = []
    for shot, (supports, queries) in episodes.items():
        class_ids = labels[supports[:, :, 0]]
        np.savez_compressed(output / f"{dataset.name}_{shot}shot_indices.npz",
                            support=supports, query=queries, class_ids=class_ids)
        records = []
        for episode, (support, query) in enumerate(zip(supports, queries)):
            if (feature_rows[support] < 0).any() or (feature_rows[query] < 0).any():
                raise ValueError("An episode references an unencoded image.")
            if not (labels[support] == class_ids[episode, :, None]).all() or not (
                labels[query] == class_ids[episode, :, None]
            ).all():
                raise ValueError("Sampler order does not agree with dataset labels.")
            support_features = features[feature_rows[support]]
            prototypes = support_features.mean(axis=1)
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            if (norms <= 1e-12).any():
                raise ValueError("An episode has a zero-norm class prototype.")
            prototypes = prototypes / norms
            query_features = features[feature_rows[query]].reshape(args.way * args.query, -1)
            predictions = (query_features @ prototypes.T).argmax(axis=1)
            targets = np.repeat(np.arange(args.way), args.query)
            correct = int((predictions == targets).sum())
            overlap = len(np.intersect1d(support, query))
            records.append({"episode": episode + 1, "correct": correct,
                            "query_count": len(targets), "accuracy_pct": 100 * correct / len(targets),
                            "support_query_overlap": overlap})
        write_csv(output / f"{dataset.name}_{shot}shot_episodes.csv", list(records[0]), records)
        accuracies = np.asarray([row["accuracy_pct"] for row in records], dtype=np.float64)
        std = float(accuracies.std(ddof=1))
        summary = {"dataset": dataset.name, "way": args.way, "shot": shot,
                   "query_per_class": args.query, "episodes": args.episodes,
                   "accuracy_pct": float(accuracies.mean()), "std_pct": std,
                   "ci95_half_width_pct": 1.96 * std / np.sqrt(args.episodes),
                   "overlap_episodes": sum(row["support_query_overlap"] > 0 for row in records)}
        summaries.append(summary)
        print(f"{dataset.name} {args.way}-way {shot}-shot: "
              f"{summary['accuracy_pct']:.2f}% +/- {summary['ci95_half_width_pct']:.2f}% "
              f"(95% CI, {args.episodes} episodes; overlap={summary['overlap_episodes']})", flush=True)
    return summaries


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly if intended.")
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("CPU evaluation requires --precision fp32.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    model, preprocess, checkpoint, repo = load_clip(args, device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or PROJECT_DIR / "logs" / "clip" / f"{checkpoint.stem}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    source_files = [Path(__file__).resolve(), Path(configs.__file__).resolve()]
    source_files += [PROJECT_DIR / "datasets" / name for name in ("datasets.py", "sampler.py", "split.py")]
    run = {
        "status": "running", "started_utc": stamp,
        "method": "frozen_clip_cosine_prototype", "partition": "labeled_80",
        "split_seed": 1, "ci": "mean +/- 1.96 * sample_std(ddof=1) / sqrt(episodes)",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "clip_repo": str(repo), "preprocess": repr(preprocess),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "clip_source_sha256": {str(path.relative_to(repo)): sha256(path)
                               for path in sorted((repo / "clip").rglob("*.py"))},
        "datasets": {},
    }
    write_json(output / "run.json", run)
    summaries = []
    try:
        for name in args.datasets:
            print(f"Preparing {name}: original labeled_80 split (seed 1)", flush=True)
            dataset, metadata = prepare_dataset(name, preprocess, args, output)
            episodes, used_indices = sample_episodes(dataset, args)
            metadata["encoded_samples"] = len(used_indices)
            run["datasets"][name] = metadata
            write_json(output / "run.json", run)
            print(f"{name}: {len(dataset)} labeled images; encoding {len(used_indices)} sampled images", flush=True)
            features = extract_features(model, dataset, used_indices, args, device)
            summaries.extend(evaluate(dataset, features, used_indices, episodes, args, output))
            write_csv(output / "summary.csv", list(summaries[0]), summaries)
            del dataset, features, episodes
        run["status"] = "complete"
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", run)


if __name__ == "__main__":
    main()
