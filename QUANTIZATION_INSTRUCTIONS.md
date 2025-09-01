# Quantization Instructions (dynamic quantization)

This repository includes `quantize.py` to perform simple dynamic quantization on a saved Hugging Face model.
Dynamic quantization works best for transformer models (BERT) and can improve CPU inference speed while reducing model size.

## Example: Quantize a BERT checkpoint
1. Ensure you have a saved model directory (e.g., `models/exp1`) containing pytorch_model.bin and config files.
2. Run:
   ```bash
   python quantize.py --model_dir models/exp1 --out_dir models/exp1_quantized --model_type bert
   ```
3. Load the quantized model in your code as usual:
   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   model = AutoModelForSequenceClassification.from_pretrained('models/exp1_quantized')
   tokenizer = AutoTokenizer.from_pretrained('models/exp1_quantized')
   ```

## Example: Quantize a ViT checkpoint (vision transformer)
Dynamic quantization may help but ONNX quantization is often better for vision models. To try dynamic quantization:
```bash
python quantize.py --model_dir models/exp1_vit --out_dir models/exp1_vit_quant --model_type vit
```

## Notes & caveats
- Quantized models are optimized for **CPU** inference. For GPU inference, quantization may not help.
- For production-grade quantization for vision models, consider exporting to ONNX and using ONNX Runtime quantization tools.
- Always validate model outputs after quantization to ensure performance drop is acceptable.
