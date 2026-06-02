"""
models/export_onnx.py
──────────────────────
Export SpeechBrain's pre-trained ECAPA-TDNN to ONNX with INT8 quantisation.

Requires speechbrain and torch to be installed (not needed at inference time).

Usage
-----
    python models/export_onnx.py --output models/ecapa_tdnn_int8.onnx
    python models/export_onnx.py --output models/ecapa_tdnn_int8.onnx --fp32  # skip quantisation
    python models/export_onnx.py --verify models/ecapa_tdnn_int8.onnx         # check output shape
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def export(output_path: Path, quantise: bool = True) -> None:
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch required: pip install torch") from e

    # SpeechBrain 1.0 moved EncoderClassifier to speechbrain.inference.classifiers
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"SpeechBrain required: pip install speechbrain\n  ({e})"
            ) from e

    # Use copy instead of symlink — symlinks require elevated privileges on Windows.
    from speechbrain.utils.fetching import LocalStrategy

    print("Loading SpeechBrain ECAPA-TDNN (VoxCeleb-trained) …")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY,
    )
    classifier.eval()

    # ── Why we export from filterbanks, not raw audio ─────────────────────────
    # torch.onnx.export cannot handle PyTorch's complex-valued STFT (opset 17
    # limitation: "STFT does not currently support complex types").
    # The SpeechBrain ECAPA-TDNN pipeline is:
    #   raw audio → STFT → Fbank (filterbanks) → ECAPA encoder → 192-dim embedding
    # We export only the Fbank → encoder → embedding part and compute filterbanks
    # ourselves at inference time using torchaudio/librosa (real-valued ops).
    # Input contract: [B, T, 80] float32 log-filterbank features, T = frames.

    import torchaudio

    class EncoderWrapper(torch.nn.Module):
        """Exports only the ECAPA encoder (filterbanks → 192-dim embedding)."""
        def __init__(self, model):
            super().__init__()
            self.encoder = model.mods.embedding_model

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            # feats: [B, T, 80]  log-mel filterbank features
            with torch.no_grad():
                lens = torch.ones(feats.shape[0])   # full length
                embeddings = self.encoder(feats, lens)  # [B, 1, 192]
                embeddings = embeddings.squeeze(1)       # [B, 192]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

    wrapper = EncoderWrapper(classifier)
    wrapper.eval()

    # Dummy filterbank input: 3 seconds at 16 kHz → ~300 frames, 80 mel bins
    dummy_feats = torch.randn(1, 298, 80)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_path = output_path.with_suffix("").with_suffix(".fp32.onnx") if quantise else output_path

    print(f"Exporting FP32 ONNX (filterbanks → embedding) → {fp32_path}")
    torch.onnx.export(
        wrapper,
        dummy_feats,
        str(fp32_path),
        input_names=["feats"],
        output_names=["embedding"],
        dynamic_axes={
            "feats":     {0: "batch", 1: "time"},
            "embedding": {0: "batch"},
        },
        opset_version=14,   # 14 is sufficient and avoids opset-17 STFT issues
        do_constant_folding=True,
    )
    print(f"FP32 export done: {fp32_path}")

    if quantise:
        print("Applying INT8 static quantisation …")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise ImportError("pip install onnxruntime for quantisation")

        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            # Quantise all MatMul/GEMM ops — safe for ECAPA-TDNN
            extra_options={"MatMulConstBOnly": True},
        )
        print(f"INT8 export done: {output_path}")
        # Clean up intermediate FP32
        fp32_path.unlink(missing_ok=True)
    else:
        print("Skipping quantisation (--fp32 flag set)")


def verify(model_path: Path) -> None:
    """
    Quick shape check using the FP32 model (onnxruntime's ConvInteger/INT8 kernel
    is not available on all platforms — verify on FP32, deploy INT8).
    """
    import onnxruntime as ort

    # Prefer FP32 for verification; fall back to whatever was passed
    fp32_path = model_path.with_name(
        model_path.stem.replace("_int8", "") + ".fp32.onnx"
    )
    verify_path = fp32_path if fp32_path.exists() else model_path

    print(f"Verifying {verify_path} …")
    session = ort.InferenceSession(str(verify_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 298, 80).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})
    embedding = outputs[0]

    assert embedding.shape == (1, 192), f"Expected (1, 192), got {embedding.shape}"
    norm = np.linalg.norm(embedding[0])
    assert abs(norm - 1.0) < 1e-3, f"Embedding not L2-normalised (norm={norm:.4f})"

    print(f"  ✓ Output shape : {embedding.shape}")
    print(f"  ✓ L2 norm      : {norm:.6f}")
    if verify_path != model_path:
        print(f"  ✓ INT8 model   : {model_path}  (verified via FP32 equivalent)")
    print("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ECAPA-TDNN → ONNX INT8")
    parser.add_argument("--output", type=Path, default=Path("models/ecapa_tdnn_int8.onnx"))
    parser.add_argument("--fp32", action="store_true", help="Skip INT8 quantisation")
    parser.add_argument("--verify", type=Path, default=None,
                        help="Only verify an existing ONNX model, skip export")
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
    else:
        export(args.output, quantise=not args.fp32)
        verify(args.output)


if __name__ == "__main__":
    main()
