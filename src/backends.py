"""Swappable model backends behind one interface:

    backend.ask(image: PIL.Image, prompt: str, options: list[str], system: str | None) -> dict
        {answer, p_options (dict option->prob, or None), logits (dict or None), option_mass, raw_text}

hf_local  : forced-choice logit readout at the answer position (deterministic, single forward pass)
gemini_api: temperature 0, parse the option; logprobs when the API returns them
openrouter: same as gemini_api via the OpenAI-compatible endpoint
mock      : a simulated model with known effects, used to test the pipeline and the statistics
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
import time

from PIL import Image


def _parse_answer(text: str, options: list[str]) -> str | None:
    t = (text or "").strip()
    for o in options:
        if re.match(rf"^\W*{re.escape(o)}\b", t, flags=re.I):
            return o
    hits = [
        (m.start(), o) for o in options for m in [re.search(rf"\b{re.escape(o)}\b", t, flags=re.I)] if m
    ]
    return min(hits)[1] if hits else None


def _jpeg_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=92)
    return b.getvalue()



class HFLocal:
    def __init__(
        self,
        hf_id: str,
        dtype="float16",
        quant="none",
        qwen_max_pixels=None,
        device=None,
        image_processor=None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        kw = {}
        if "qwen" in hf_id.lower() and qwen_max_pixels:
            kw["max_pixels"] = qwen_max_pixels
            kw["min_pixels"] = 64 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(hf_id, **kw)
        for k, v in (image_processor or {}).items():
            setattr(self.processor.image_processor, k, v)
        mkw = dict(torch_dtype=td, low_cpu_mem_usage=True)
        if quant == "4bit" and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            mkw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=td,
                bnb_4bit_use_double_quant=True,
            )
            mkw["device_map"] = {"": 0}
        elif quant == "8bit" and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            mkw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            mkw["device_map"] = {"": 0}
        try:
            mkw["attn_implementation"] = "sdpa"
            self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **mkw)
        except (ValueError, ImportError):
            mkw.pop("attn_implementation")
            self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **mkw)
        if "device_map" not in mkw:
            self.model.to(self.device)
        self.model.eval()
        self.tok = self.processor.tokenizer
        self._opt_cache: dict[str, list[int]] = {}
        self.hf_id = hf_id

    def option_ids(self, opt: str) -> list[int]:
        if opt not in self._opt_cache:
            ids = set()
            for v in (opt, " " + opt, opt.lower(), " " + opt.lower()):
                t = self.tok.encode(v, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            self._opt_cache[opt] = sorted(ids)
        return self._opt_cache[opt]

    def build_inputs(self, image, prompt, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        try:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        except (ValueError, TypeError):
            inputs = self.processor(text=[text], images=[[image]], return_tensors="pt")
        return inputs.to(self.model.device)

    def forward_logits(self, inputs, **kw):
        with self.torch.no_grad():
            out = self.model(**inputs, use_cache=False, **kw)
        return out

    def ask(self, image, prompt, options, system=None):
        inputs = self.build_inputs(image, prompt, system)
        out = self.forward_logits(inputs)
        logits = out.logits[0, -1].float()
        logp_full = self.torch.log_softmax(logits, -1)
        scores = {}
        mass = 0.0
        for o in options:
            ids = self.option_ids(o)
            lp = self.torch.logsumexp(logp_full[ids], 0).item()
            scores[o] = lp
            mass += math.exp(lp)
        m = max(scores.values())
        z = sum(math.exp(v - m) for v in scores.values())
        p = {o: math.exp(v - m) / z for o, v in scores.items()}
        top = self.tok.decode([int(logits.argmax())])
        return {
            "answer": max(p, key=p.get),
            "p_options": p,
            "logits": scores,
            "option_mass": mass,
            "raw_text": top,
        }



class GeminiAPI:
    def __init__(self, api_model: str, rpm: float = 10, thinking: str = "none", logprobs: bool = False):
        from google import genai
        from google.genai import types

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (AI Studio) in .env or the environment.")
        self.genai = genai
        # v1beta: thinking controls are rejected on v1
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))
        self.model = api_model
        self.min_gap = 60.0 / rpm
        self._last = 0.0
        self.thinking = thinking
        self.logprobs_ok = (
            logprobs  # AI Studio keys currently return 400 "Logprobs is not enabled" -> text readout
        )

    def _generation_config(self, system, with_logprobs):
        from google.genai import types

        kw = dict(temperature=0.0, max_output_tokens=32)
        if system and not self.model.startswith("gemma"):
            kw["system_instruction"] = system
        if self.thinking == "budget0":
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if with_logprobs:
            kw["response_logprobs"] = True
            kw["logprobs"] = 5
        return types.GenerateContentConfig(**kw)

    def ask(self, image, prompt, options, system=None):
        from google.genai import errors, types

        if system and self.model.startswith("gemma"):  # gemma on AI Studio has no system role
            prompt = system + "\n" + prompt
        parts = [types.Part.from_bytes(data=_jpeg_bytes(image), mime_type="image/jpeg"), prompt]
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=self._generation_config(system, self.logprobs_ok),
                )
                break
            except errors.ClientError as e:
                msg = str(e)
                if self.logprobs_ok and "logprob" in msg.lower():
                    self.logprobs_ok = False
                    continue
                if e.code == 429:
                    if "PerDay" in msg or "per day" in msg.lower():
                        raise QuotaExhausted(msg) from e
                    time.sleep(min(60, 5 * 2**attempt))
                    continue
                raise
            except errors.ServerError:
                time.sleep(5 * 2**attempt)
        else:
            raise RuntimeError("Gemini retries exhausted")
        text = resp.text or ""
        ans = _parse_answer(text, options)
        p = None
        try:
            cand = resp.candidates[0]
            lr = cand.logprobs_result
            if lr and lr.top_candidates:
                first = lr.top_candidates[0].candidates
                sc = {}
                for o in options:
                    lps = [c.log_probability for c in first if c.token.strip().lower() == o.lower()]
                    if lps:
                        sc[o] = max(lps)
                if len(sc) == len(options):
                    m = max(sc.values())
                    z = sum(math.exp(v - m) for v in sc.values())
                    p = {o: math.exp(v - m) / z for o, v in sc.items()}
        except (AttributeError, IndexError, TypeError):
            pass
        if p is None and ans is not None:
            p = {o: float(o == ans) for o in options}
        return {"answer": ans, "p_options": p, "logits": None, "option_mass": None, "raw_text": text}


class QuotaExhausted(RuntimeError):
    pass



class OpenRouter:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_model: str, rpm: float = 15):
        import requests

        self.requests = requests
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("Set OPENROUTER_API_KEY in the environment.")
        self.model, self.min_gap, self._last = api_model, 60.0 / rpm, 0.0

    def ask(self, image, prompt, options, system=None):
        url = "data:image/jpeg;base64," + base64.b64encode(_jpeg_bytes(image)).decode()
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        body = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0,
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 5,
        }
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.requests.post(
                self.URL, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=120
            )
            if r.status_code == 429:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            r.raise_for_status()
            j = r.json()
            break
        else:
            raise QuotaExhausted("OpenRouter rate limit")
        ch = j["choices"][0]
        text = ch["message"].get("content") or ""
        ans = _parse_answer(text, options)
        p = None
        try:
            top = ch["logprobs"]["content"][0]["top_logprobs"]
            sc = {
                o: max(t["logprob"] for t in top if t["token"].strip().lower() == o.lower())
                for o in options
            }
            m = max(sc.values())
            z = sum(math.exp(v - m) for v in sc.values())
            p = {o: math.exp(v - m) / z for o, v in sc.items()}
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        if p is None and ans is not None:
            p = {o: float(o == ans) for o in options}
        return {"answer": ans, "p_options": p, "logits": None, "option_mass": None, "raw_text": text}



class Mock:
    """Simulated model with KNOWN effects, for pipeline and statistics tests.

    Ground truth built in: text susceptibility rises with confidence level (H2 true), most rebinding happens
    with intact perception (H3 true), flagging gives no protection (H4 true), small A-letter bias that order
    averaging must cancel.
    """

    LEVEL = {"contra_1": 0.6, "contra_2": 1.4, "contra_3": 1.9, "contra_4": 2.6}

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _uniform_sample(self, *key) -> float:
        h = hashlib.sha256(("|".join(map(str, key)) + str(self.seed)).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def _normal_sample(self, *key) -> float:  # ~N(0,1)
        u1, u2 = max(self._uniform_sample(*key, "a"), 1e-12), self._uniform_sample(*key, "b")
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def ask_query(self, q, item):
        skill = (
            2.5 + 1.2 * self._normal_sample(q.item_id, "skill") - (0.8 if item["dataset"] == "real" else 0)
        )
        gull = 0.9 * self._normal_sample(q.item_id, "gull")
        noise = 0.4 * self._normal_sample(q.qid)
        if q.question == "q4":
            x = skill
            if q.condition in self.LEVEL:
                x -= self.LEVEL[q.condition] * math.exp(gull * 0.6) + (0.0 if q.system is None else -0.8)
            elif q.condition == "consistent":
                x += 0.8
            elif q.condition == "neutral":
                x -= 0.15
            x += noise
            bias = 0.3 if q.options[0] == q.correct else -0.3
            p_true = 1 / (1 + math.exp(-(x + bias)))
            p = {q.correct: p_true, q.distractor: 1 - p_true}
        elif q.question in ("q2", "q2_foil"):
            want_yes = q.correct == "Yes"
            x = (
                2.2
                + 0.5 * self._normal_sample(q.item_id, "perc")
                - (0.35 * self.LEVEL.get(q.condition, 0) if want_yes else 0)
                + noise
            )
            if not want_yes:
                x = 1.0 + noise
            p_c = 1 / (1 + math.exp(-x))
            p = {q.correct: p_c, ("No" if q.correct == "Yes" else "Yes"): 1 - p_c}
        elif q.question == "q3":
            x = (
                0.4
                + 0.6 * self._normal_sample(q.item_id, "flag")
                + 0.3 * self.LEVEL.get(q.condition, 0) * (q.correct == "No")
                + noise
            )
            p_c = 1 / (1 + math.exp(-x))
            p = {q.correct: p_c, ("No" if q.correct == "Yes" else "Yes"): 1 - p_c}
        else:
            p_c = 1 / (1 + math.exp(-(2.5 + noise)))
            other = [o for o in q.options if o != q.correct][0]
            p = {q.correct: p_c, other: 1 - p_c}
        lg = {o: math.log(max(v, 1e-9)) for o, v in p.items()}
        return {
            "answer": max(p, key=p.get),
            "p_options": p,
            "logits": lg,
            "option_mass": 0.95,
            "raw_text": max(p, key=p.get),
        }


def make_backend(spec: dict, runtime: dict):
    b = spec["backend"]
    if b == "hf_local":
        return HFLocal(
            spec["hf_id"],
            spec.get("dtype", "float16"),
            spec.get("quant", "none"),
            runtime.get("qwen_max_pixels"),
            image_processor=spec.get("image_processor"),
        )
    if b == "gemini_api":
        return GeminiAPI(
            spec["api_model"],
            rpm=spec.get("rpm", 10),
            thinking=spec.get("thinking", "none"),
            logprobs=spec.get("logprobs", False),
        )
    if b == "openrouter":
        return OpenRouter(spec["api_model"], rpm=spec.get("rpm", 15))
    if b == "mock":
        return Mock()
    raise ValueError(b)
