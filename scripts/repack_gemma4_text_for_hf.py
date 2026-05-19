import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _load_tensor(source_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    shard = source_dir / weight_map[name]
    with safe_open(shard, framework="numpy") as f:
        return np.ascontiguousarray(f.get_tensor(name))


def _put(
    output: dict[str, np.ndarray],
    source_dir: Path,
    weight_map: dict[str, str],
    source_name: str,
    target_name: str,
) -> None:
    if source_name in weight_map:
        output[target_name] = _load_tensor(source_dir, weight_map, source_name)


def _flush(
    tensors: dict[str, np.ndarray],
    output_dir: Path,
    shard_paths: list[Path],
    tensor_to_shard: dict[str, str],
) -> int:
    if not tensors:
        return 0
    shard_name = f"model-{len(shard_paths) + 1:05d}-of-PENDING.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, shard_path, metadata={"format": "pt"})
    shard_paths.append(shard_path)
    for name in tensors:
        tensor_to_shard[name] = shard_name
    size = sum(t.nbytes for t in tensors.values())
    tensors.clear()
    return size


def _finalize(output_dir: Path, shard_paths: list[Path], tensor_to_shard: dict[str, str]) -> None:
    total = len(shard_paths)
    rename_map: dict[str, str] = {}
    for idx, old_path in enumerate(shard_paths, start=1):
        new_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
        new_path = output_dir / new_name
        if new_path.exists():
            new_path.unlink()
        old_path.rename(new_path)
        rename_map[old_path.name] = new_name

    index = {
        "metadata": {"total_size": sum((output_dir / name).stat().st_size for name in rename_map.values())},
        "weight_map": {name: rename_map[shard] for name, shard in sorted(tensor_to_shard.items())},
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def repack(source_dir: Path, metadata_dir: Path, output_dir: Path, shard_size_gb: float) -> None:
    source_index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    source_weight_map = source_index["weight_map"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in metadata_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)

    pending: dict[str, np.ndarray] = {}
    shard_paths: list[Path] = []
    tensor_to_shard: dict[str, str] = {}
    current_size = 0
    max_shard_size = int(shard_size_gb * 1024**3)

    def add(source_name: str, target_name: str) -> None:
        nonlocal current_size
        before = len(pending)
        _put(pending, source_dir, source_weight_map, source_name, target_name)
        if len(pending) == before:
            return
        current_size += pending[target_name].nbytes
        if current_size >= max_shard_size:
            _flush(pending, output_dir, shard_paths, tensor_to_shard)
            current_size = 0

    add("token_embd.weight", "model.language_model.embed_tokens.weight")
    add("output_norm.weight", "model.language_model.norm.weight")
    add("per_layer_token_embd.weight", "model.language_model.embed_tokens_per_layer.weight")
    add("per_layer_model_proj.weight", "model.language_model.per_layer_model_projection.weight")
    add("per_layer_proj_norm.weight", "model.language_model.per_layer_projection_norm.weight")

    for layer in range(35):
        prefix = f"blk.{layer}"
        target = f"model.language_model.layers.{layer}"
        add(f"{prefix}.layer_output_scale.weight", f"{target}.layer_scalar")
        add(f"{prefix}.attn_q.weight", f"{target}.self_attn.q_proj.weight")
        add(f"{prefix}.attn_q_norm.weight", f"{target}.self_attn.q_norm.weight")
        if layer < 15:
            add(f"{prefix}.attn_k.weight", f"{target}.self_attn.k_proj.weight")
            add(f"{prefix}.attn_v.weight", f"{target}.self_attn.v_proj.weight")
            add(f"{prefix}.attn_k_norm.weight", f"{target}.self_attn.k_norm.weight")
        add(f"{prefix}.attn_output.weight", f"{target}.self_attn.o_proj.weight")
        add(f"{prefix}.ffn_gate.weight", f"{target}.mlp.gate_proj.weight")
        add(f"{prefix}.ffn_up.weight", f"{target}.mlp.up_proj.weight")
        add(f"{prefix}.ffn_down.weight", f"{target}.mlp.down_proj.weight")
        add(f"{prefix}.attn_norm.weight", f"{target}.input_layernorm.weight")
        add(f"{prefix}.post_attention_norm.weight", f"{target}.post_attention_layernorm.weight")
        add(f"{prefix}.ffn_norm.weight", f"{target}.pre_feedforward_layernorm.weight")
        add(f"{prefix}.post_ffw_norm.weight", f"{target}.post_feedforward_layernorm.weight")
        add(f"{prefix}.inp_gate.weight", f"{target}.per_layer_input_gate.weight")
        add(f"{prefix}.proj.weight", f"{target}.per_layer_projection.weight")
        add(f"{prefix}.post_norm.weight", f"{target}.post_per_layer_input_norm.weight")

    _flush(pending, output_dir, shard_paths, tensor_to_shard)
    _finalize(output_dir, shard_paths, tensor_to_shard)
    print(f"Repacked {len(tensor_to_shard)} tensors into {len(shard_paths)} shards at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack Gemma4 GGUF-derived safetensors with Hugging Face tensor names.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size-gb", type=float, default=6.0)
    args = parser.parse_args()
    repack(args.source, args.metadata, args.output, args.shard_size_gb)


if __name__ == "__main__":
    main()
