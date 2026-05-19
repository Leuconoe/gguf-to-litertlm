import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader, dequantize
from safetensors import safe_open
from safetensors.numpy import save_file


GEMMA4_VISION_METADATA_V2 = (
    "CgUKAwoBAhIFCgMKAQESBQoDCgFqEgUKAwoBMiiAIDJhQl8KChIIPHxpbWFnZT4SChIIPGltYWdlfD4qDDx8dG9vbF9j"
    "YWxsPjIMPHRvb2xfY2FsbHw+UgU8fCJ8PloFPHwifD5iEDx8dG9vbF9yZXNwb25zZT5oAXgQgAEQiAHsCTq9A3t7"
    "IGJvc190b2tlbiB9fXslIGZvciBtZXNzYWdlIGluIG1lc3NhZ2VzICV9PHx0dXJuPnslIGlmIG1lc3NhZ2VbJ3Jv"
    "bGUnXSA9PSAnYXNzaXN0YW50JyAlfW1vZGVseyUgZWxzZSAlfXt7IG1lc3NhZ2VbJ3JvbGUnXSB9fXslIGVuZGlm"
    "ICV9CnslIGlmIG1lc3NhZ2VbJ2NvbnRlbnQnXSBpcyBzdHJpbmcgJX17eyBtZXNzYWdlWydjb250ZW50J10gfX17"
    "JSBlbHNlICV9eyUgZm9yIGl0ZW0gaW4gbWVzc2FnZVsnY29udGVudCddICV9eyUgaWYgaXRlbVsndHlwZSddID09"
    "ICd0ZXh0JyAlfXt7IGl0ZW1bJ3RleHQnXSB9fXslIGVsaWYgaXRlbVsndHlwZSddID09ICdpbWFnZScgJX08fGlt"
    "YWdlfD57JSBlbmRpZiAlfXslIGVuZGZvciAlfXslIGVuZGlmICV9PHR1cm58Pgp7JSBlbmRmb3IgJX17JSBpZiBh"
    "ZGRfZ2VuZXJhdGlvbl9wcm9tcHQgJX08fHR1cm4+bW9kZWwKeyUgZW5kaWYgJX1CKQoHdGhvdWdodBISPHxjaGFu"
    "bmVsPnRob3VnaHQKGgo8Y2hhbm5lbHw+"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def field_value(field: Any) -> Any:
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


def write_sharded_safetensors(
    output_dir: Path,
    tensors_iter,
    metadata: dict[str, str],
    shard_size_gb: float,
) -> None:
    max_shard_size = int(shard_size_gb * 1024**3)
    shard_paths: list[Path] = []
    tensor_to_tmp_shard: dict[str, Path] = {}
    tensor_sizes: dict[str, int] = {}
    pending: dict[str, np.ndarray] = {}
    current_size = 0

    def flush() -> None:
        nonlocal pending, current_size
        if not pending:
            return
        shard_path = output_dir / f"tmp-shard-{len(shard_paths) + 1:05d}.safetensors"
        sizes = {name: tensor.nbytes for name, tensor in pending.items()}
        save_file(pending, shard_path, metadata=metadata)
        print(f"Saved {shard_path.name} ({sum(sizes.values()) / (1024 ** 3):.2f} GiB)")
        shard_paths.append(shard_path)
        for name in pending:
            tensor_to_tmp_shard[name] = shard_path
        tensor_sizes.update(sizes)
        pending = {}
        current_size = 0

    for name, tensor in tensors_iter:
        if pending and current_size + tensor.nbytes > max_shard_size:
            flush()
        pending[name] = tensor
        current_size += tensor.nbytes
    flush()

    tmp_to_final: dict[Path, str] = {}
    for i, tmp_path in enumerate(shard_paths, start=1):
        final_name = f"model-{i:05d}-of-{len(shard_paths):05d}.safetensors"
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
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def convert_gguf(gguf_path: Path, output_dir: Path, dtype: str, shard_size_gb: float, force: bool) -> None:
    clean_dir(output_dir, force)
    reader = GGUFReader(str(gguf_path))
    np_dtype = np.float32 if dtype == "fp32" else np.float16
    metadata = {
        "general.architecture": str(field_value(reader.fields.get("general.architecture"))),
        "general.name": str(field_value(reader.fields.get("general.name"))),
        "general.file_type": str(field_value(reader.fields.get("general.file_type"))),
        "description": "Model tensors converted from GGUF.",
    }

    def tensors():
        for i, tensor in enumerate(reader.tensors):
            weights = dequantize(tensor.data, tensor.tensor_type)
            weights = np.ascontiguousarray(weights.astype(np_dtype, copy=False))
            print(
                f"[{i + 1}/{len(reader.tensors)}] {tensor.name} "
                f"shape={weights.shape} dtype={weights.dtype} size={weights.nbytes / (1024 ** 2):.1f} MiB"
            )
            yield tensor.name, weights

    print(f"Converting GGUF: {gguf_path}")
    write_sharded_safetensors(output_dir, tensors(), metadata, shard_size_gb)


def load_tensor(source_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    with safe_open(source_dir / weight_map[name], framework="numpy") as f:
        return np.ascontiguousarray(f.get_tensor(name))


def finalize_repack(output_dir: Path, shard_paths: list[Path], tensor_to_shard: dict[str, str]) -> None:
    rename_map: dict[str, str] = {}
    for idx, old_path in enumerate(shard_paths, start=1):
        new_name = f"model-{idx:05d}-of-{len(shard_paths):05d}.safetensors"
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


def repack_text(source_dir: Path, metadata_dir: Path, output_dir: Path, shard_size_gb: float, force: bool) -> None:
    clean_dir(output_dir, force)
    source_index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    source_weight_map: dict[str, str] = source_index["weight_map"]
    for path in metadata_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)

    pending: dict[str, np.ndarray] = {}
    shard_paths: list[Path] = []
    tensor_to_shard: dict[str, str] = {}
    current_size = 0
    max_shard_size = int(shard_size_gb * 1024**3)

    def flush() -> None:
        nonlocal pending, current_size
        if not pending:
            return
        shard_name = f"model-{len(shard_paths) + 1:05d}-of-PENDING.safetensors"
        save_file(pending, output_dir / shard_name, metadata={"format": "pt"})
        shard_paths.append(output_dir / shard_name)
        for name in pending:
            tensor_to_shard[name] = shard_name
        pending = {}
        current_size = 0

    def add(source_name: str, target_name: str) -> None:
        nonlocal current_size
        if source_name not in source_weight_map:
            return
        tensor = load_tensor(source_dir, source_weight_map, source_name)
        pending[target_name] = tensor
        current_size += tensor.nbytes
        if current_size >= max_shard_size:
            flush()

    add("token_embd.weight", "model.language_model.embed_tokens.weight")
    add("output_norm.weight", "model.language_model.norm.weight")
    add("per_layer_token_embd.weight", "model.language_model.embed_tokens_per_layer.weight")
    add("per_layer_model_proj.weight", "model.language_model.per_layer_model_projection.weight")
    add("per_layer_proj_norm.weight", "model.language_model.per_layer_projection_norm.weight")
    for layer in range(35):
        src = f"blk.{layer}"
        dst = f"model.language_model.layers.{layer}"
        add(f"{src}.layer_output_scale.weight", f"{dst}.layer_scalar")
        add(f"{src}.attn_q.weight", f"{dst}.self_attn.q_proj.weight")
        add(f"{src}.attn_q_norm.weight", f"{dst}.self_attn.q_norm.weight")
        if layer < 15:
            add(f"{src}.attn_k.weight", f"{dst}.self_attn.k_proj.weight")
            add(f"{src}.attn_v.weight", f"{dst}.self_attn.v_proj.weight")
            add(f"{src}.attn_k_norm.weight", f"{dst}.self_attn.k_norm.weight")
        add(f"{src}.attn_output.weight", f"{dst}.self_attn.o_proj.weight")
        add(f"{src}.ffn_gate.weight", f"{dst}.mlp.gate_proj.weight")
        add(f"{src}.ffn_up.weight", f"{dst}.mlp.up_proj.weight")
        add(f"{src}.ffn_down.weight", f"{dst}.mlp.down_proj.weight")
        add(f"{src}.attn_norm.weight", f"{dst}.input_layernorm.weight")
        add(f"{src}.post_attention_norm.weight", f"{dst}.post_attention_layernorm.weight")
        add(f"{src}.ffn_norm.weight", f"{dst}.pre_feedforward_layernorm.weight")
        add(f"{src}.post_ffw_norm.weight", f"{dst}.post_feedforward_layernorm.weight")
        add(f"{src}.inp_gate.weight", f"{dst}.per_layer_input_gate.weight")
        add(f"{src}.proj.weight", f"{dst}.per_layer_projection.weight")
        add(f"{src}.post_norm.weight", f"{dst}.post_per_layer_input_norm.weight")

    flush()
    finalize_repack(output_dir, shard_paths, tensor_to_shard)
    print(f"Repacked {len(tensor_to_shard)} text tensors")


def copy_or_link(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def repack_vision(text_hf_dir: Path, mmproj_dir: Path, output_dir: Path, force: bool) -> None:
    clean_dir(output_dir, force)
    text_index = json.loads((text_hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mm_index = json.loads((mmproj_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    text_weight_map: dict[str, str] = text_index["weight_map"]
    mm_weight_map: dict[str, str] = mm_index["weight_map"]

    for path in text_hf_dir.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, output_dir / path.name)

    final_weight_map: dict[str, str] = {}
    for shard_name in sorted(set(text_weight_map.values())):
        copy_or_link(text_hf_dir / shard_name, output_dir / shard_name)
    final_weight_map.update(text_weight_map)

    vision_tensors: dict[str, np.ndarray] = {}

    def add(source_name: str, target_name: str) -> None:
        vision_tensors[target_name] = load_tensor(mmproj_dir, mm_weight_map, source_name)

    add("v.position_embd.weight", "model.vision_tower.patch_embedder.position_embedding_table")
    patch_weight = load_tensor(mmproj_dir, mm_weight_map, "v.patch_embd.weight")
    if patch_weight.shape != (768, 3, 16, 16):
        raise ValueError(f"Unexpected v.patch_embd.weight shape: {patch_weight.shape}")
    vision_tensors["model.vision_tower.patch_embedder.input_proj.weight"] = np.ascontiguousarray(
        patch_weight.transpose(0, 2, 3, 1).reshape(768, 768)
    )
    add("mm.input_projection.weight", "model.embed_vision.embedding_projection.weight")

    for layer in range(16):
        src = f"v.blk.{layer}"
        dst = f"model.vision_tower.encoder.layers.{layer}"
        add(f"{src}.attn_q.weight", f"{dst}.self_attn.q_proj.linear.weight")
        add(f"{src}.attn_k.weight", f"{dst}.self_attn.k_proj.linear.weight")
        add(f"{src}.attn_v.weight", f"{dst}.self_attn.v_proj.linear.weight")
        add(f"{src}.attn_out.weight", f"{dst}.self_attn.o_proj.linear.weight")
        add(f"{src}.attn_q_norm.weight", f"{dst}.self_attn.q_norm.weight")
        add(f"{src}.attn_k_norm.weight", f"{dst}.self_attn.k_norm.weight")
        add(f"{src}.ffn_gate.weight", f"{dst}.mlp.gate_proj.linear.weight")
        add(f"{src}.ffn_up.weight", f"{dst}.mlp.up_proj.linear.weight")
        add(f"{src}.ffn_down.weight", f"{dst}.mlp.down_proj.linear.weight")
        add(f"{src}.ln1.weight", f"{dst}.input_layernorm.weight")
        add(f"{src}.attn_post_norm.weight", f"{dst}.post_attention_layernorm.weight")
        add(f"{src}.ln2.weight", f"{dst}.pre_feedforward_layernorm.weight")
        add(f"{src}.ffn_post_norm.weight", f"{dst}.post_feedforward_layernorm.weight")

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
    print(f"Added {len(vision_tensors)} vision tensors")


def write_vision_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(GEMMA4_VISION_METADATA_V2))


def docker_path(path: Path) -> str:
    return "/work/" + path.as_posix()


def export_tflite(hf_dir: Path, export_dir: Path, with_vision: bool, export_mode: str, force: bool) -> None:
    if export_mode == "skip":
        print("Skipping LiteRT export")
        return
    clean_dir(export_dir, force)
    task = "image_text_to_text" if with_vision else "text_generation"
    flags = [
        f"litert-torch export_hf {shlex.quote(docker_path(hf_dir))} {shlex.quote(docker_path(export_dir))}",
        f"--task={task}",
        "--externalize_embedder=True",
        "--bundle_litert_lm=False",
        "--keep_temporary_files=True",
    ]
    if with_vision:
        flags.append("--export_vision_encoder=True")

    if export_mode == "local":
        cmd = ["litert-torch", "export_hf", str(hf_dir), str(export_dir), f"--task={task}",
               "--externalize_embedder=True", "--bundle_litert_lm=False", "--keep_temporary_files=True"]
        if with_vision:
            cmd.append("--export_vision_encoder=True")
        run(cmd)
        return

    run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{Path.cwd().resolve()}:/work",
            "-w",
            "/work",
            "python:3.12-slim",
            "sh",
            "-lc",
            "python -m pip install --upgrade pip && "
            "pip install --pre litert-torch pillow torchvision && "
            + " ".join(flags),
        ]
    )


def bundle_vision_litertlm(export_dir: Path, tokenizer_json: Path, metadata_pb: Path, output: Path, builder_cmd: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(builder_cmd) + [
        "system_metadata", "--str", "Authors", "ODML",
        "--str", "Source", "GGUF text plus mmproj vision repacked for Gemma4 / Google AI Edge Gallery",
        "--str", "Template", "simple-gemma4-vision-turns-v2",
        "--str", "Note", "max_num_patches=1260; v.patch_embd OIHW to OHWI flatten",
        "llm_metadata", "--path", str(metadata_pb),
        "hf_tokenizer", "--path", str(tokenizer_json),
        "tflite_model", "--path", str(export_dir / "model_quantized.tflite"), "--model_type", "prefill_decode",
        "tflite_model", "--path", str(export_dir / "embedder_quantized.tflite"), "--model_type", "embedder",
        "tflite_model", "--path", str(export_dir / "per_layer_embedder_quantized.tflite"), "--model_type", "per_layer_embedder",
        "tflite_model", "--path", str(export_dir / "vision_encoder_quantized.tflite"), "--model_type", "vision_encoder",
        "tflite_model", "--path", str(export_dir / "vision_adapter_quantized.tflite"), "--model_type", "vision_adapter",
        "output", "--path", str(output),
    ]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Gemma4 GGUF to a LiteRT-LM bundle.")
    parser.add_argument("--input", required=True, type=Path, help="Text model GGUF file.")
    parser.add_argument("--output", required=True, type=Path, help="Output .litertlm path.")
    parser.add_argument("--vision-encoder", type=Path, help="Optional Gemma4 mmproj GGUF file.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("hf_base_metadata/gemma-4-e2b-it"))
    parser.add_argument("--work-dir", type=Path, default=Path("converted/gguf-to-litertlm-work"))
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--shard-size-gb", type=float, default=6.0)
    parser.add_argument("--export-mode", choices=("docker", "local", "skip"), default="docker")
    parser.add_argument("--builder-cmd", default="uv tool run litert-lm-builder")
    parser.add_argument("--skip-bundle", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.metadata_dir.exists():
        raise SystemExit(f"Metadata directory not found: {args.metadata_dir}")

    text_safe = args.work_dir / "01-text-safetensors"
    hf_text = args.work_dir / "02-hf-text"
    mmproj_safe = args.work_dir / "03-mmproj-safetensors"
    hf_vision = args.work_dir / "04-hf-vision"
    export_dir = args.work_dir / "05-litert-export"
    metadata_pb = args.work_dir / "gemma4_vision_v2.pb"

    convert_gguf(args.input, text_safe, args.dtype, args.shard_size_gb, args.force)
    repack_text(text_safe, args.metadata_dir, hf_text, args.shard_size_gb, args.force)

    with_vision = args.vision_encoder is not None
    hf_for_export = hf_text
    if with_vision:
        write_vision_metadata(metadata_pb)
        convert_gguf(args.vision_encoder, mmproj_safe, args.dtype, args.shard_size_gb, args.force)
        repack_vision(hf_text, mmproj_safe, hf_vision, args.force)
        hf_for_export = hf_vision

    export_tflite(hf_for_export, export_dir, with_vision, args.export_mode, args.force)

    if not args.skip_bundle:
        if not with_vision:
            raise SystemExit("Text-only bundling is not implemented in this single-file path yet. Use --skip-bundle.")
        bundle_vision_litertlm(export_dir, hf_for_export / "tokenizer.json", metadata_pb, args.output, args.builder_cmd)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
