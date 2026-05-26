"""Convert TFDS CLIC TFRecord shards into regular image files.

Expected input layout:
  datasets/clic/1.0.0/clic-train.tfrecord-*
  datasets/clic/1.0.0/clic-validation.tfrecord-*
  datasets/clic/1.0.0/clic-test.tfrecord-*

Output layout:
  datasets/clic_images/train/*.png
  datasets/clic_images/validation/*.png
  datasets/clic_images/test/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_split(tf, input_dir: Path, output_dir: Path, split: str) -> None:
    """Convert one split worth of TFRecord shards into image files."""
    pattern = f"clic-{split}.tfrecord-*"
    shards = sorted(input_dir.glob(pattern))
    if not shards:
        print(f"No shards found for split={split} in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total_shards = len(shards)
    total_examples = 0

    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    print(f"Converting split={split} from {input_dir}")
    print(f"Found {total_shards} shard(s). Output: {output_dir}")

    for shard_index, shard_path in enumerate(shards, start=1):
        print(f"[{shard_index}/{total_shards}] {shard_path.name}")
        dataset = tf.data.TFRecordDataset(str(shard_path))
        shard_examples = 0

        for record_index, raw_record in enumerate(dataset):
            example = tf.io.parse_single_example(raw_record, feature_spec)
            image_bytes = example["image"].numpy()
            image_name = f"{split}_{shard_index:03d}_{record_index:05d}.png"
            (output_dir / image_name).write_bytes(image_bytes)
            shard_examples += 1
            total_examples += 1

            if record_index % 25 == 0:
                print(
                    f"\r  images written: {total_examples}",
                    end="",
                    flush=True,
                )

        print(f"\r  images written: {total_examples}")
        print(f"  shard complete: {shard_examples} image(s)")

    print(f"Finished split={split}: {total_examples} image(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TFDS CLIC TFRecord shards into normal PNG image files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "clic" / "1.0.0"),
        help="Directory containing clic-*.tfrecord-* shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "clic_images"),
        help="Directory to write extracted PNG images.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which split to convert.",
    )
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for TFRecord conversion. "
            "Install it with `pip install tensorflow-cpu` or `pip install tensorflow`."
        ) from exc

    input_dir = Path(args.input_dir).resolve()
    output_base = Path(args.output_dir).resolve()
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)

    for split in splits:
        convert_split(tf, input_dir, output_base / split, split)


if __name__ == "__main__":
    main()
