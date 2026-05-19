# gguf-to-litertlm

Single-command workflow for converting a Gemma4 GGUF model into a Google AI
Edge Gallery-compatible `.litertlm` bundle.

The main entrypoint is one file:

```powershell
python gguf_to_litertlm.py --input model.gguf --output model.litertlm
```

Model weights are not included. Keep `.gguf`, `.safetensors`, `.tflite`, and
`.litertlm` files outside git.

## Features

- Converts the text GGUF to sharded safetensors.
- Repackages text tensors into Hugging Face Gemma4 tensor names.
- Optionally adds a Gemma4 `mmproj` vision encoder.
- Fixes the Gemma4 vision patch projection layout:
  GGUF stores `v.patch_embd.weight` as `OIHW`, but Gemma4 image patches are
  flattened as `HW_C`.
- Runs `litert-torch export_hf` automatically, using Docker by default.
- Bundles the exported TFLite files into `.litertlm`.
- Embeds the Gemma4 vision LiteRT metadata in the Python file, so there is no
  separate metadata protobuf to manage.

## Requirements

Install the local preprocessing dependencies:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

For the LiteRT export step on Windows, Docker is the default and recommended
mode. The script runs the long Docker command internally.

The final bundling step uses:

```powershell
uv tool run litert-lm-builder
```

## Metadata Directory

The script needs a small Hugging Face metadata/tokenizer directory from the
matching Gemma4 base model. By default it looks for:

```text
hf_base_metadata/gemma-4-e2b-it
```

That directory should include files such as:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `chat_template.jinja`

You can override it with `--metadata-dir`.

## Vision Bundle

```powershell
.\.venv\Scripts\python gguf_to_litertlm.py `
  --input Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf `
  --vision-encoder mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf `
  --output converted\gemma4-gallery-vision.litertlm `
  --metadata-dir hf_base_metadata\gemma-4-e2b-it `
  --force
```

This is the recommended path for Google AI Edge Gallery image-text use.

## Common Options

- `--input`: text model GGUF.
- `--output`: final `.litertlm` file.
- `--vision-encoder`: optional Gemma4 `mmproj` GGUF. Required for image input.
- `--metadata-dir`: Hugging Face metadata/tokenizer directory.
- `--work-dir`: intermediate output directory. Default:
  `converted/gguf-to-litertlm-work`.
- `--dtype fp16|fp32`: safetensors dtype before LiteRT export. Default: `fp16`.
- `--export-mode docker|local|skip`: Docker is default.
- `--skip-bundle`: stop after TFLite export.
- `--force`: rebuild generated intermediate directories.

## Notes

- The tested Google AI Edge Gallery path used INT8 dynamic-range TFLite files.
- GPU execution can work in Edge Gallery when GPU is forced in the app and the
  device delegate accepts the generated LiteRT model.
- Q4/FP8 GGUF quantization formats are not directly preserved in `.litertlm` by
  this workflow. LiteRT export applies its own TFLite quantization recipes.
- Text-only export can produce TFLite files, but this repository currently
  bundles the vision path manually because that is the tested Gallery workflow.
