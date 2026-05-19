import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _load_tensor(source_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    with safe_open(source_dir / weight_map[name], framework="numpy") as f:
        return np.ascontiguousarray(f.get_tensor(name))


def _copy_or_link(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def repack(text_hf_dir: Path, mmproj_dir: Path, output_dir: Path) -> None:
    text_index = json.loads((text_hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mm_index = json.loads((mmproj_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    text_weight_map: dict[str, str] = text_index["weight_map"]
    mm_weight_map: dict[str, str] = mm_index["weight_map"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in text_hf_dir.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, output_dir / path.name)

    final_weight_map: dict[str, str] = {}
    for shard_name in sorted(set(text_weight_map.values())):
        _copy_or_link(text_hf_dir / shard_name, output_dir / shard_name)
    final_weight_map.update(text_weight_map)

    vision_tensors: dict[str, np.ndarray] = {}

    def add(source_name: str, target_name: str, reshape: tuple[int, ...] | None = None) -> None:
        tensor = _load_tensor(mmproj_dir, mm_weight_map, source_name)
        if reshape is not None:
            tensor = np.ascontiguousarray(tensor.reshape(reshape))
        vision_tensors[target_name] = tensor

    add("v.position_embd.weight", "model.vision_tower.patch_embedder.position_embedding_table")

    # GGUF stores the patch projection as Conv2D weights in OIHW order. Gemma4's
    # HF image processor flattens patches in HW_C order, so a plain reshape
    # scrambles color/spatial features and makes images look noisy/gray.
    patch_weight = _load_tensor(mmproj_dir, mm_weight_map, "v.patch_embd.weight")
    if patch_weight.shape != (768, 3, 16, 16):
        raise ValueError(f"Unexpected v.patch_embd.weight shape: {patch_weight.shape}")
    vision_tensors["model.vision_tower.patch_embedder.input_proj.weight"] = np.ascontiguousarray(
        patch_weight.transpose(0, 2, 3, 1).reshape(768, 768)
    )

    add("mm.input_projection.weight", "model.embed_vision.embedding_projection.weight")

    for layer in range(16):
        source = f"v.blk.{layer}"
        target = f"model.vision_tower.encoder.layers.{layer}"
        add(f"{source}.attn_q.weight", f"{target}.self_attn.q_proj.linear.weight")
        add(f"{source}.attn_k.weight", f"{target}.self_attn.k_proj.linear.weight")
        add(f"{source}.attn_v.weight", f"{target}.self_attn.v_proj.linear.weight")
        add(f"{source}.attn_out.weight", f"{target}.self_attn.o_proj.linear.weight")
        add(f"{source}.attn_q_norm.weight", f"{target}.self_attn.q_norm.weight")
        add(f"{source}.attn_k_norm.weight", f"{target}.self_attn.k_norm.weight")
        add(f"{source}.ffn_gate.weight", f"{target}.mlp.gate_proj.linear.weight")
        add(f"{source}.ffn_up.weight", f"{target}.mlp.up_proj.linear.weight")
        add(f"{source}.ffn_down.weight", f"{target}.mlp.down_proj.linear.weight")
        add(f"{source}.ln1.weight", f"{target}.input_layernorm.weight")
        add(f"{source}.attn_post_norm.weight", f"{target}.post_attention_layernorm.weight")
        add(f"{source}.ln2.weight", f"{target}.pre_feedforward_layernorm.weight")
        add(f"{source}.ffn_post_norm.weight", f"{target}.post_feedforward_layernorm.weight")

    vision_shard = "model-vision-00001-of-00001.safetensors"
    save_file(vision_tensors, output_dir / vision_shard, metadata={"format": "pt"})
    for name in vision_tensors:
        final_weight_map[name] = vision_shard

    total_size = sum((output_dir / shard).stat().st_size for shard in set(final_weight_map.values()))
    final_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(final_weight_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(final_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(vision_tensors)} vision tensors")
    print(f"Combined {len(final_weight_map)} total tensors at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Gemma4 mmproj vision tensors to a text HF safetensors directory.")
    parser.add_argument("--text-hf", required=True, type=Path)
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repack(args.text_hf, args.mmproj, args.output)


if __name__ == "__main__":
    main()
