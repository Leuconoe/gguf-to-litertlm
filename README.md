# gguf-to-litertlm

Utilities for converting Gemma4 GGUF checkpoints into a Hugging Face-style
safetensors directory and then exporting a Google AI Edge Gallery-compatible
`.litertlm` bundle.

This repository does not include model weights. Keep `.gguf`, `.safetensors`,
`.tflite`, and `.litertlm` files outside git.

## What This Handles

- Converts quantized GGUF tensors to sharded FP16/FP32 safetensors.
- Repackages HauhauCS Gemma4 text tensors into Hugging Face Gemma4 names.
- Adds Gemma4 `mmproj` vision tensors.
- Fixes the vision patch projection layout:
  GGUF stores `v.patch_embd.weight` as `OIHW`, while Gemma4 image preprocessing
  flattens patches as `HW_C`. The repacker converts `OIHW -> OHWI -> flatten`.
- Provides a small Gemma4 vision LiteRT-LM metadata file with
  `max_num_patches=1260`.

## Requirements

Local conversion scripts:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

LiteRT export currently works best from Linux. On Windows, use Docker:

```powershell
docker run --rm -v "${PWD}:/work" -w /work python:3.12-slim sh -lc "python -m pip install --upgrade pip && pip install --pre litert-torch pillow torchvision && litert-torch export_hf /work/converted/gemma4-hf-vision /work/converted/gemma4-export --task=image_text_to_text --export_vision_encoder=True --externalize_embedder=True --bundle_litert_lm=False --keep_temporary_files=True"
```

## Workflow

1. Convert the text GGUF to sharded safetensors:

```powershell
.\.venv\Scripts\python scripts\gguf_to_safetensors.py `
  --input Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf `
  --output converted\gemma4-q3-fp16 `
  --dtype fp16 `
  --shard-size-gb 6
```

2. Prepare a metadata directory from the matching base Hugging Face Gemma4 model.
   It should include files such as `config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `processor_config.json`, and `chat_template.jinja`.

3. Repack text tensors into Hugging Face names:

```powershell
.\.venv\Scripts\python scripts\repack_gemma4_text_for_hf.py `
  --source converted\gemma4-q3-fp16 `
  --metadata hf_base_metadata\gemma-4-e2b-it `
  --output converted\gemma4-hf-text `
  --shard-size-gb 6
```

4. Convert the mmproj GGUF:

```powershell
.\.venv\Scripts\python scripts\gguf_to_safetensors.py `
  --input mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf `
  --output converted\gemma4-mmproj-fp16 `
  --dtype fp16 `
  --shard-size-gb 6
```

5. Add vision tensors:

```powershell
.\.venv\Scripts\python scripts\repack_gemma4_mmproj_for_hf.py `
  --text-hf converted\gemma4-hf-text `
  --mmproj converted\gemma4-mmproj-fp16 `
  --output converted\gemma4-hf-vision
```

6. Export TFLite files with `litert-torch export_hf` using the Docker command
   above.

7. Bundle the exported TFLite files into `.litertlm`:

```powershell
.\scripts\build_litertlm.ps1 `
  -ExportDir converted\gemma4-export `
  -TokenizerJson hf_base_metadata\gemma-4-e2b-it\tokenizer.json `
  -Output converted\gemma4-gallery-vision.litertlm
```

## Notes

- The tested Google AI Edge Gallery path used INT8 dynamic-range TFLite files.
- For GPU in Edge Gallery, forcing GPU in the app can work when the device
  delegate accepts the generated LiteRT model. FP16/mixed-precision export may
  be useful for GPU-first experiments, but it produces much larger files.
- Q4/FP8 GGUF quantization formats are not directly preserved in `.litertlm` by
  these scripts. LiteRT export uses its own TFLite quantization recipes.
