from lhotse.recipes.utils import read_manifests_if_cached

dataset_parts = ["train", "dev", "test"]

output_dir = "data"
src_dir = ""
prefix = "libritts"
suffix = "jsonl.gz"

manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
        types=["recordings", "supervisions", "cuts"],
    )

print(len(manifests))