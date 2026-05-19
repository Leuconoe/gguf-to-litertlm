import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from typing import Any, Tuple
from gguf import GGUFReader, dequantize

def load_gguf_and_extract_metadata(gguf_path: str) -> Tuple[GGUFReader, list]:
    """Load GGUF file and extract metadata and tensors."""
    reader = GGUFReader(gguf_path)
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {
            'name': tensor.name,
            'shape': tuple(tensor.shape.tolist()),
            'n_elements': tensor.n_elements,
            'n_bytes': tensor.n_bytes,
            'data_offset': tensor.data_offset,
            'type': tensor.tensor_type,
        }
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata


def _field_value(field: Any) -> Any:
    if field is None:
        return None
    if hasattr(field, "contents"):
        try:
            return field.contents()
        except Exception:
            pass
    try:
        if len(field.data) == 1:
            data = field.parts[field.data[0]]
            if getattr(data, "shape", ()) == ():
                return data.item()
            if getattr(data, "dtype", None) == np.uint8:
                return bytes(data.tolist()).decode("utf-8")
            return data.tolist()
    except Exception:
        return str(field)
    return str(field)


def _save_shard(
    tensors: dict[str, np.ndarray],
    shard_index: int,
    output_dir: Path,
    metadata: dict[str, str],
) -> tuple[Path, dict[str, int]]:
    shard_path = output_dir / f"tmp-shard-{shard_index:05d}.safetensors"
    sizes = {name: tensor.nbytes for name, tensor in tensors.items()}
    save_file(tensors, shard_path, metadata=metadata)
    print(f"Saved shard {shard_path.name} ({sum(sizes.values()) / (1024 ** 3):.2f} GiB)")
    return shard_path, sizes


def _finalize_shards(
    output_dir: Path,
    shard_paths: list[Path],
    tensor_to_tmp_shard: dict[str, Path],
    tensor_sizes: dict[str, int],
) -> None:
    total_shards = len(shard_paths)
    tmp_to_final: dict[Path, str] = {}
    for i, tmp_path in enumerate(shard_paths, start=1):
        final_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        final_path = output_dir / final_name
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        tmp_to_final[tmp_path] = final_name

    index = {
        "metadata": {"total_size": sum(tensor_sizes.values())},
        "weight_map": {
            name: tmp_to_final[tmp_path]
            for name, tmp_path in sorted(tensor_to_tmp_shard.items())
        },
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote index: {output_dir / 'model.safetensors.index.json'}")


def convert_gguf_to_safetensors(
    gguf_path: str,
    output_path: str,
    dtype: str,
    shard_size_gb: float,
) -> None:
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f"Extracted {len(tensors_metadata)} tensors from GGUF file")

    output = Path(output_path)
    sharded = output.suffix != ".safetensors"
    if sharded:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float32 if dtype == "fp32" else np.float16
    metadata = {
        "general.architecture": str(_field_value(reader.fields.get("general.architecture"))),
        "general.name": str(_field_value(reader.fields.get("general.name"))),
        "general.file_type": str(_field_value(reader.fields.get("general.file_type"))),
        "description": "Model tensors converted from GGUF.",
    }

    tensors_dict: dict[str, np.ndarray] = {}
    current_shard_size = 0
    max_shard_size = int(shard_size_gb * 1024**3)
    shard_paths: list[Path] = []
    tensor_to_tmp_shard: dict[str, Path] = {}
    tensor_sizes: dict[str, int] = {}

    def flush_shard() -> None:
        nonlocal tensors_dict, current_shard_size
        if not tensors_dict:
            return
        shard_path, sizes = _save_shard(tensors_dict, len(shard_paths) + 1, output, metadata)
        shard_paths.append(shard_path)
        for name in tensors_dict:
            tensor_to_tmp_shard[name] = shard_path
        tensor_sizes.update(sizes)
        tensors_dict = {}
        current_shard_size = 0

    for i, tensor_info in enumerate(tensors_metadata):
        tensor_name = tensor_info['name']

        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type)
        weights = np.ascontiguousarray(weights.astype(np_dtype, copy=False))

        print(
            f"[{i + 1}/{len(tensors_metadata)}] {tensor_name} | "
            f"shape={weights.shape} dtype={weights.dtype} "
            f"size={weights.nbytes / (1024 ** 2):.1f} MiB"
        )

        if sharded:
            if tensors_dict and current_shard_size + weights.nbytes > max_shard_size:
                flush_shard()
            tensors_dict[tensor_name] = weights
            current_shard_size += weights.nbytes
        else:
            tensors_dict[tensor_name] = weights

    if sharded:
        flush_shard()
        _finalize_shards(output, shard_paths, tensor_to_tmp_shard, tensor_sizes)
    else:
        save_file(tensors_dict, output, metadata=metadata)
    print("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(description="Convert GGUF files to safetensors format.")
    parser.add_argument("--input", required=True, help="Path to the input GGUF file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .safetensors file, or an output directory for sharded safetensors.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Output tensor dtype. Defaults to fp16.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=4.0,
        help="Approximate maximum shard size when --output is a directory.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Deprecated compatibility flag. Use --dtype fp16 or --dtype fp32.",
    )

    args = parser.parse_args()

    if args.bf16:
        raise SystemExit("BF16 output is not supported by the streaming NumPy writer. Use --dtype fp16 or --dtype fp32.")

    convert_gguf_to_safetensors(args.input, args.output, args.dtype, args.shard_size_gb)

if __name__ == "__main__":
    main()
