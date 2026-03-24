#!/usr/bin/env python3
"""
Activation Server — serves intermediate-layer hidden states from a transformer.

Loads a model via HuggingFace transformers, registers hooks to capture
residual stream activations at a specified layer, and serves them via HTTP.

Usage:
    python scripts/activation_server.py \
        --model Qwen/Qwen3-8B \
        --layer 16 \
        --port 8100

    # With 4-bit quantization (fits in 16GB VRAM):
    python scripts/activation_server.py \
        --model Qwen/Qwen3-8B \
        --layer 16 \
        --quantize 4bit \
        --port 8100

Dependencies:
    pip install torch transformers fastapi uvicorn accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model wrapper with hook-based activation capture
# ---------------------------------------------------------------------------

class ActivationModel:
    def __init__(self, model_name: str, target_layer: int, device: str = "auto",
                 quantize: Optional[str] = None):
        print(f"Loading {model_name} (layer {target_layer})...")

        self.target_layer = target_layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True, "device_map": device}

        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("  Using 4-bit quantization")
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("  Using 8-bit quantization")
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        print(f"  Loaded: {self.hidden_dim}d, {self.n_layers} layers")

        if target_layer >= self.n_layers:
            print(f"  WARNING: target_layer {target_layer} >= n_layers {self.n_layers}, "
                  f"using layer {self.n_layers - 1}")
            self.target_layer = self.n_layers - 1

        # Detect layer modules
        self.layers = self._detect_layers()
        print(f"  Architecture: {type(self.model).__name__}, {len(self.layers)} layers detected")

        # Hook state
        self._captured: Optional[torch.Tensor] = None
        self._lock = threading.Lock()

        # Register persistent hook on target layer
        self._hook = self.layers[self.target_layer].register_forward_hook(self._capture_hook)
        print(f"  Hook registered on layer {self.target_layer}")

    def _detect_layers(self) -> list:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)  # LLaMA/Qwen/Mistral
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)  # GPT-2
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return list(self.model.gpt_neox.layers)  # Pythia
        else:
            raise ValueError(f"Unknown architecture: {type(self.model).__name__}")

    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Last token position — carries the contextualized meaning of the full input.
        # Mean-pooling washes out value-specific signal (all descriptions look the same).
        self._captured = hidden[0][-1].detach().float().cpu()

    def get_hidden_state(self, text: str) -> tuple[List[float], int]:
        """Run text through the model and return the captured hidden state.

        Returns (hidden_state_list, n_tokens).
        """
        with self._lock:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            n_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                self.model(**inputs)

            if self._captured is None:
                raise RuntimeError("Hook did not fire")

            result = self._captured.tolist()
            self._captured = None
            return result, n_tokens


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="GoT Activation Server")
_model: Optional[ActivationModel] = None


class HiddenStateRequest(BaseModel):
    text: str
    layer: Optional[int] = None  # ignored for now — uses the configured layer


class HiddenStateResponse(BaseModel):
    hidden_state: List[float]
    layer: int
    n_tokens: int
    hidden_dim: int


@app.post("/hidden_states")
async def hidden_states(req: HiddenStateRequest) -> HiddenStateResponse:
    if _model is None:
        raise RuntimeError("Model not loaded")

    hs, n_tokens = _model.get_hidden_state(req.text)
    return HiddenStateResponse(
        hidden_state=hs,
        layer=_model.target_layer,
        n_tokens=n_tokens,
        hidden_dim=_model.hidden_dim,
    )


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[dict]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if _model is None:
        raise RuntimeError("Model not loaded")

    with _model._lock:
        text = _model.tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = _model.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(_model.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _model.model.generate(
                **inputs,
                max_new_tokens=req.max_tokens or 1024,
                do_sample=True,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = _model.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # OpenAI-compatible response format
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "model": req.model or "qwen3-8b",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hidden_dim": _model.hidden_dim if _model else 0,
        "layer": _model.target_layer if _model else 0,
        "n_layers": _model.n_layers if _model else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _model

    parser = argparse.ArgumentParser(description="GoT Activation Server")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--layer", type=int, required=True, help="Target layer for hidden state extraction")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization (4bit or 8bit, requires bitsandbytes)")
    args = parser.parse_args()

    _model = ActivationModel(
        model_name=args.model,
        target_layer=args.layer,
        device=args.device,
        quantize=args.quantize,
    )

    print(f"Activation server listening on http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  Layer: {_model.target_layer}/{_model.n_layers}")
    print(f"  Hidden dim: {_model.hidden_dim}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
