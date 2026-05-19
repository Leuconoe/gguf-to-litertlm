param(
    [Parameter(Mandatory = $true)]
    [string]$ExportDir,

    [Parameter(Mandatory = $true)]
    [string]$TokenizerJson,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$Metadata = "metadata/gemma4_vision_v2.pb"
)

$ErrorActionPreference = "Stop"

uv tool run litert-lm-builder `
    system_metadata `
        --str Authors ODML `
        --str Source "GGUF text plus mmproj vision repacked for Gemma4 / Google AI Edge Gallery" `
        --str Template "simple-gemma4-vision-turns-v2" `
        --str Note "max_num_patches=1260; v.patch_embd OIHW to OHWI flatten" `
    llm_metadata --path $Metadata `
    hf_tokenizer --path $TokenizerJson `
    tflite_model --path (Join-Path $ExportDir "model_quantized.tflite") --model_type prefill_decode `
    tflite_model --path (Join-Path $ExportDir "embedder_quantized.tflite") --model_type embedder `
    tflite_model --path (Join-Path $ExportDir "per_layer_embedder_quantized.tflite") --model_type per_layer_embedder `
    tflite_model --path (Join-Path $ExportDir "vision_encoder_quantized.tflite") --model_type vision_encoder `
    tflite_model --path (Join-Path $ExportDir "vision_adapter_quantized.tflite") --model_type vision_adapter `
    output --path $Output
