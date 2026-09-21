from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import torch

from data import TinyStoriesDataModule, tokenizer_from_state_dict
from generate import generate_token_ids, generate_token_ids_with_cache
from model import MiniTransformerLM, TransformerConfig
from train import _load_config, _resolve_path
from training.trainer import get_default_device


class GenerationApp:
    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path | None,
        device: str | None,
    ) -> None:
        self.config_path = config_path
        self.config = _load_config(config_path)
        self.project_root = config_path.resolve().parents[1]
        self.checkpoint = (
            torch.load(checkpoint_path, map_location="cpu")
            if checkpoint_path is not None
            else None
        )
        self.tokenizer = self._load_tokenizer()
        self.device = torch.device(device) if device else get_default_device()
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        greedy: bool,
        use_cache: bool,
    ) -> str:
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=self.device,
        )
        generate_fn = generate_token_ids_with_cache if use_cache else generate_token_ids
        output_ids = generate_fn(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            context_length=self.config["model"].get("context_length"),
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        if output_text.startswith(prompt):
            return output_text
        continuation = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :].tolist())
        return prompt + continuation

    def _load_tokenizer(self):
        if self.checkpoint is not None and "tokenizer" in self.checkpoint:
            return tokenizer_from_state_dict(self.checkpoint["tokenizer"])

        data_config = self.config["dataset"]
        data = TinyStoriesDataModule(
            train_path=_resolve_path(self.project_root, data_config["train_path"]),
            validation_path=_resolve_path(self.project_root, data_config["validation_path"]),
            text_column=data_config["text_column"],
            tokenizer_type=data_config["tokenizer"],
            vocab_size=data_config["vocab_size"],
            min_pair_frequency=data_config["min_pair_frequency"],
            context_length=data_config["context_length"],
            batch_size=data_config["batch_size"],
            max_train_chars=data_config.get("max_train_chars"),
            max_validation_chars=data_config.get("max_validation_chars"),
            tokenizer_cache_path=data_config.get("tokenizer_cache_path"),
        )
        data.setup()
        return data.tokenizer

    def _load_model(self) -> MiniTransformerLM:
        model_config = self.config["model"]
        checkpoint_config = (
            self.checkpoint.get("transformer_config")
            if self.checkpoint is not None
            else None
        )
        if checkpoint_config is not None:
            config = TransformerConfig(**checkpoint_config)
        else:
            config = TransformerConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=model_config["d_model"],
                n_layers=model_config["n_layers"],
                n_heads=model_config["n_heads"],
            )
        model = MiniTransformerLM(config)
        if self.checkpoint is not None:
            model.load_state_dict(self.checkpoint["model"])
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a tiny local generation UI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mini-transformer/configs/cloud_gpu.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cloud_gpu/step_020000.pt"),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    app = GenerationApp(
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            self._send_html(_render_page(checkpoint_path))

        def do_POST(self) -> None:
            if self.path != "/generate":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            prompt = _form_value(form, "prompt", "Once upon a time")
            greedy = _form_value(form, "greedy", "") == "on"
            top_k_value = _form_value(form, "top_k", "40").strip()
            try:
                output = app.generate(
                    prompt=prompt,
                    max_new_tokens=int(_form_value(form, "max_new_tokens", "120")),
                    temperature=_parse_float(_form_value(form, "temperature", "0.8")),
                    top_k=int(top_k_value) if top_k_value else None,
                    greedy=greedy,
                    use_cache=_form_value(form, "use_cache", "on") == "on",
                )
                payload = {"output": output}
            except Exception as exc:  # noqa: BLE001 - surfaced to local UI.
                payload = {"error": str(exc)}
            self._send_json(payload)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"generation UI: http://{args.host}:{args.port}")
    print(f"device: {app.device}")
    print(f"checkpoint: {checkpoint_path or 'random weights'}")
    server.serve_forever()


def _form_value(form: dict[str, list[str]], key: str, default: str) -> str:
    values = form.get(key)
    return values[0] if values else default


def _parse_float(value: str) -> float:
    return float(value.replace(",", "."))


def _resolve_checkpoint_path(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.exists():
        return checkpoint_path

    candidates = sorted(Path("checkpoints/cloud_gpu").glob("step_*.pt"))
    if candidates:
        return candidates[-1]

    candidates = sorted(Path("checkpoints").glob("step_*.pt"))
    if candidates:
        return candidates[-1]

    return None


def _render_page(checkpoint_path: Path | None) -> str:
    checkpoint_label = html.escape(str(checkpoint_path or "random weights"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TinyLM Generation</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111316;
      color: #f4f0e8;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #111316;
    }}
    main {{
      width: min(1080px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 32px 0;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0 0 20px;
      color: #9aa0a6;
      font-size: 13px;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      align-items: start;
    }}
    form, .output {{
      border: 1px solid #30343a;
      border-radius: 8px;
      background: #191c20;
    }}
    form {{
      padding: 16px;
    }}
    label {{
      display: block;
      margin: 0 0 12px;
      color: #d7d0c3;
      font-size: 13px;
      font-weight: 650;
    }}
    textarea, input {{
      box-sizing: border-box;
      width: 100%;
      margin-top: 6px;
      border: 1px solid #3b4148;
      border-radius: 6px;
      background: #101215;
      color: #f4f0e8;
      padding: 10px;
      font: inherit;
    }}
    textarea {{
      min-height: 120px;
      resize: vertical;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .check {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }}
    .check input {{
      width: 16px;
      height: 16px;
      margin: 0;
    }}
    button {{
      width: 100%;
      height: 42px;
      border: 0;
      border-radius: 6px;
      background: #f0b35a;
      color: #17120b;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }}
    button:disabled {{
      opacity: 0.6;
      cursor: wait;
    }}
    .output {{
      min-height: 520px;
      padding: 18px;
      white-space: pre-wrap;
      line-height: 1.55;
      color: #f7f2e8;
    }}
    .muted {{
      color: #9aa0a6;
    }}
    @media (max-width: 820px) {{
      main {{
        width: min(100vw - 24px, 1080px);
        padding: 18px 0;
      }}
      .workspace {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>TinyLM Generation</h1>
    <p class="meta">Checkpoint: {checkpoint_label}</p>
    <div class="workspace">
      <form id="form">
        <label>Prompt
          <textarea name="prompt">Once upon a time</textarea>
        </label>
        <div class="row">
          <label>Tokens
            <input name="max_new_tokens" type="number" min="1" max="512" value="120">
          </label>
          <label>Top-k
            <input name="top_k" type="number" min="1" value="40">
          </label>
        </div>
        <label>Temperature
          <input name="temperature" type="number" min="0.05" max="3" step="0.05" value="0.8">
        </label>
        <label class="check"><input name="use_cache" type="checkbox" checked> Use KV cache</label>
        <label class="check"><input name="greedy" type="checkbox"> Greedy decoding</label>
        <button id="button" type="submit">Generate</button>
      </form>
      <section id="output" class="output muted">Generated text will appear here.</section>
    </div>
  </main>
  <script>
    const form = document.querySelector("#form");
    const button = document.querySelector("#button");
    const output = document.querySelector("#output");
    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      button.disabled = true;
      output.classList.add("muted");
      output.textContent = "Generating...";
      const response = await fetch("/generate", {{
        method: "POST",
        body: new FormData(form),
      }});
      const data = await response.json();
      button.disabled = false;
      output.classList.toggle("muted", Boolean(data.error));
      output.textContent = data.error ? data.error : data.output;
    }});
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
