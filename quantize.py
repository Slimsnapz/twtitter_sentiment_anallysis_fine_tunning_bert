\
    """quantize.py
    Simple post-training dynamic quantization script for Hugging Face models.
    - Works for BERT-style models (AutoModelForSequenceClassification) on CPU (dynamic quantization).
    - Works for ViT/vision transformers in PyTorch by quantizing linear layers (may not always reduce size dramatically).

    Usage examples:
      python quantize.py --model_dir models/exp1 --out_dir models/exp1_quant --model_type bert
      python quantize.py --model_dir models/exp1 --out_dir models/exp1_quant --model_type vit

    Notes:
    - Dynamic quantization reduces model size and improves CPU inference latency in many cases.
    - This script requires torch and transformers. Quantized model is saved with `save_pretrained` and can be loaded with `from_pretrained`.
    """
    import argparse
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForImageClassification

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", required=True, help="Path to the original model directory")
        parser.add_argument("--out_dir", required=True, help="Path to save quantized model")
        parser.add_argument("--model_type", choices=["bert","vit"], default="bert", help="Model topology")
        return parser.parse_args()

    def quantize_bert(model_dir, out_dir):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        # dynamic quantization for linear layers to qint8
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized.save_pretrained(out_dir)
        print("Saved quantized BERT model to", out_dir)

    def quantize_vit(model_dir, out_dir):
        # For ViT, dynamic quantization on Linear layers may help for CPU inference.
        # Note: For better results with vision models, consider ONNX export + ONNXRuntime quantization.
        model = AutoModelForImageClassification.from_pretrained(model_dir)
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized.save_pretrained(out_dir)
        print("Saved quantized ViT model to", out_dir)

    def main():
        args = parse_args()
        if args.model_type == "bert":
            quantize_bert(args.model_dir, args.out_dir)
        else:
            quantize_vit(args.model_dir, args.out_dir)

    if __name__ == "__main__":
        main()
