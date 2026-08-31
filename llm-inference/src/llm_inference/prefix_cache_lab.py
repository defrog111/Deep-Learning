from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


TRACKED_METRICS = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


@dataclass(frozen=True)
class RequestMeasurement:
    name: str
    ttft_seconds: float
    e2e_seconds: float
    prompt_tokens: int
    completion_tokens: int
    output_text: str
    metric_deltas: Dict[str, float]

    @property
    def tpot_seconds(self) -> Optional[float]:
        if self.completion_tokens <= 1:
            return None
        decode_window = max(0.0, self.e2e_seconds - self.ttft_seconds)
        return decode_window / (self.completion_tokens - 1)

    @property
    def cache_hit_rate(self) -> Optional[float]:
        queries = self.metric_deltas.get("vllm:prefix_cache_queries", 0.0)
        if queries <= 0:
            return None
        hits = self.metric_deltas.get("vllm:prefix_cache_hits", 0.0)
        return hits / queries


def parse_prometheus_metrics(
    text: str, tracked_metrics: Iterable[str] = TRACKED_METRICS
) -> Dict[str, float]:
    """Sum selected Prometheus samples across their label combinations."""

    tracked = set(tracked_metrics)
    totals = {name: 0.0 for name in tracked}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        emitted_name = parts[0].split("{", 1)[0]
        canonical_name = (
            emitted_name[:-6] if emitted_name.endswith("_total") else emitted_name
        )
        if canonical_name not in tracked:
            continue
        try:
            totals[canonical_name] += float(parts[1])
        except ValueError:
            continue
    return totals


def metric_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        name: after.get(name, 0.0) - before.get(name, 0.0)
        for name in set(before) | set(after)
    }


def build_repository_prefix(target_characters: int) -> str:
    """Build stable, code-like context without requiring a local tokenizer."""

    header = (
        "You are a coding assistant. Read this repository snapshot carefully and "
        "answer questions about it.\n\n"
    )
    lines: List[str] = [header]
    index = 0
    while sum(map(len, lines)) < target_characters:
        lines.append(
            f"def handler_{index}(request):\n"
            f"    value = request.get('field_{index}', {index})\n"
            f"    return {{'handler': {index}, 'value': value, 'ok': True}}\n\n"
        )
        index += 1
    return "".join(lines)


class VLLMPrefixCacheExperiment:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def scrape_metrics(self) -> Dict[str, float]:
        request = urllib.request.Request(f"{self.base_url}/metrics")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return parse_prometheus_metrics(body)

    def run_request(
        self,
        name: str,
        shared_prefix: str,
        question: str,
        max_tokens: int,
    ) -> RequestMeasurement:
        before = self.scrape_metrics()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": shared_prefix},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Qwen3 uses this chat-template argument to produce a concise lab response.
            # Servers/models that ignore it still run the cache experiment correctly.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        first_token_at: Optional[float] = None
        output_parts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                completion_tokens = int(
                    usage.get("completion_tokens", completion_tokens)
                )

                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_parts.append(piece)

        finished = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("stream completed without a generated text token")
        after = self.scrape_metrics()
        return RequestMeasurement(
            name=name,
            ttft_seconds=first_token_at - started,
            e2e_seconds=finished - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_text="".join(output_parts),
            metric_deltas=metric_delta(before, after),
        )


def _format_optional_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 1_000:.2f} ms/token"


def print_report(measurement: RequestMeasurement) -> None:
    hit_rate = measurement.cache_hit_rate
    hit_rate_text = "n/a" if hit_rate is None else f"{hit_rate:.2%}"
    print(f"\n[{measurement.name}]")
    print(f"TTFT:              {measurement.ttft_seconds * 1_000:.2f} ms")
    print(f"Approx. TPOT:      {_format_optional_seconds(measurement.tpot_seconds)}")
    print(f"End-to-end:        {measurement.e2e_seconds * 1_000:.2f} ms")
    print(f"Prompt tokens:     {measurement.prompt_tokens}")
    print(f"Completion tokens: {measurement.completion_tokens}")
    print(f"Prefix hit tokens: {measurement.metric_deltas.get('vllm:prefix_cache_hits', 0):.0f}")
    print(
        "Prefix query tokens: "
        f"{measurement.metric_deltas.get('vllm:prefix_cache_queries', 0):.0f}"
    )
    print(f"Prefix hit rate:   {hit_rate_text}")
    print(
        "Cached prompt tokens: "
        f"{measurement.metric_deltas.get('vllm:prompt_tokens_cached', 0):.0f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure vLLM TTFT/TPOT with two requests sharing a long prefix"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prefix-characters",
        type=int,
        default=40_000,
        help="Approximate long-prefix size; actual token count comes from vLLM usage",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the short model/kernel warm-up request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = VLLMPrefixCacheExperiment(args.base_url, args.model, args.timeout)
    prefix = build_repository_prefix(args.prefix_characters)
    questions = (
        "Which handler reads field_7? Answer in one sentence.",
        "Which handler reads field_73? Answer in one sentence.",
    )

    try:
        if not args.skip_warmup:
            print("Sending a short warm-up request (excluded from the report)...")
            experiment.run_request(
                "model warm-up",
                "This is a model warm-up and is unrelated to the repository experiment.",
                "Reply with OK.",
                8,
            )
        cold = experiment.run_request(
            "request 1: cold prefix", prefix, questions[0], args.max_tokens
        )
        warm = experiment.run_request(
            "request 2: shared prefix", prefix, questions[1], args.max_tokens
        )
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach vLLM at {args.base_url}: {exc}. Start vllm serve first."
        ) from exc

    print_report(cold)
    print_report(warm)
    improvement = (cold.ttft_seconds - warm.ttft_seconds) / cold.ttft_seconds
    print(f"\nTTFT improvement: {improvement:.2%}")
    print("Interpretation: expect request 2 to reuse prefix tokens and reduce TTFT.")
    print("TPOT may remain similar because generated tokens still require decode steps.")


if __name__ == "__main__":
    main()
