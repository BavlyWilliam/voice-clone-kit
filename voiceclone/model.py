"""Loading the open-source Chatterbox models (Resemble AI, MIT licence) with a small memory footprint.

Two engines:
  english       ChatterboxTTS: best English quality, with emotion-intensity control. ~3.2 GB download.
  multilingual  ChatterboxMultilingualTTS: 23 languages (Arabic, Italian, French, Spanish, …). ~3.3 GB download.

Weights download once into the Hugging Face cache (~/.cache/huggingface). After that everything runs offline.

Why a custom loader: the stock loader memory-maps each 2 GB weights file "copy-on-write", which on Windows
reserves the whole size in the page file and fails with "The paging file is too small" when other apps are using
memory. Here weights are read tensor by tensor, so only one tensor is ever held twice.
"""
from __future__ import annotations

import gc
import json
import struct
from pathlib import Path

import torch

ENGINES = ("english", "multilingual")
_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
_cache: dict[str, object] = {}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stream(path: Path):
    """Yield (name, tensor) from a .safetensors file one tensor at a time."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        base = 8 + size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            data = bytearray(f.read(end - start))
            dtype = _DTYPES[info["dtype"]]
            yield name, (torch.frombuffer(data, dtype=dtype).reshape(info["shape"]) if data
                         else torch.empty(info["shape"], dtype=dtype))


def _fill(module: torch.nn.Module, path: Path) -> None:
    """Copy weights into a module in place. Keys only on one side are ignored (like strict=False)."""
    own = module.state_dict()
    if path.suffix == ".safetensors":
        items = _stream(path)
    else:  # small .pt files
        items = torch.load(path, map_location="cpu", weights_only=True).items()
    for name, tensor in items:
        if name in own:
            own[name].copy_(tensor)


def _files(names: list[str]) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {n: Path(hf_hub_download(repo_id="ResembleAI/chatterbox", filename=n)) for n in names}


def _free_other(keep: str) -> None:
    for key in [k for k in _cache if k != keep and k != "encoder"]:
        del _cache[key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(engine: str = "english"):
    """The text-to-speech model for an engine (cached; only one big model is kept on the GPU at a time)."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    if engine in _cache:
        return _cache[engine]
    _free_other(engine)
    dev = device()
    if engine == "english":
        from chatterbox import tts as ct

        f = _files(["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"])
        ve = ct.VoiceEncoder(); _fill(ve, f["ve.safetensors"]); ve.to(dev).eval()
        t3 = ct.T3(); _fill(t3, f["t3_cfg.safetensors"]); t3.to(dev).eval()
        s3 = ct.S3Gen(); _fill(s3, f["s3gen.safetensors"]); s3.to(dev).eval()
        tok = ct.EnTokenizer(str(f["tokenizer.json"]))
        conds = ct.Conditionals.load(f["conds.pt"], map_location=dev).to(dev)
        model = ct.ChatterboxTTS(t3, s3, ve, tok, dev, conds=conds)
    else:
        from chatterbox import mtl_tts as mt

        f = _files(["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt", "Cangjie5_TC.json"])
        ve = mt.VoiceEncoder(); _fill(ve, f["ve.pt"]); ve.to(dev).eval()
        t3 = mt.T3(mt.T3Config.multilingual()); _fill(t3, f["t3_mtl23ls_v2.safetensors"]); t3.to(dev).eval()
        s3 = mt.S3Gen(); _fill(s3, f["s3gen.pt"]); s3.to(dev).eval()
        tok = mt.MTLTokenizer(str(f["grapheme_mtl_merged_expanded_v1.json"]))
        conds = mt.Conditionals.load(f["conds.pt"], map_location=dev).to(dev)
        model = mt.ChatterboxMultilingualTTS(t3, s3, ve, tok, dev, conds=conds)
    _cache[engine] = model
    return model


def encoder():
    """Just the small speaker encoder (for choosing clips and scoring similarity): fast, ~20 MB."""
    if "encoder" not in _cache:
        from chatterbox import tts as ct

        ve = ct.VoiceEncoder()
        _fill(ve, _files(["ve.safetensors"])["ve.safetensors"])
        _cache["encoder"] = ve.to(device()).eval()
    return _cache["encoder"]


def languages() -> dict[str, str]:
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

    return dict(SUPPORTED_LANGUAGES)
