from llm_inference.prefix_cache_lab import (
    RequestMeasurement,
    build_repository_prefix,
    metric_delta,
    parse_prometheus_metrics,
)


def test_parse_prometheus_metrics_sums_labels_and_counter_suffix() -> None:
    metrics = """
# HELP vllm:prefix_cache_hits Prefix hits
vllm:prefix_cache_hits_total{model_name="a"} 100
vllm:prefix_cache_hits_total{model_name="b"} 20
vllm:prefix_cache_queries{model_name="a"} 200
vllm:num_requests_running{model_name="a"} 3
unrelated_metric 999
"""

    parsed = parse_prometheus_metrics(metrics)

    assert parsed["vllm:prefix_cache_hits"] == 120
    assert parsed["vllm:prefix_cache_queries"] == 200
    assert parsed["vllm:num_requests_running"] == 3


def test_metric_delta_handles_counters() -> None:
    assert metric_delta(
        {"vllm:prefix_cache_hits": 10},
        {"vllm:prefix_cache_hits": 42},
    ) == {"vllm:prefix_cache_hits": 32}


def test_measurement_calculates_tpot_and_hit_rate() -> None:
    measurement = RequestMeasurement(
        name="warm",
        ttft_seconds=0.2,
        e2e_seconds=1.1,
        prompt_tokens=10_000,
        completion_tokens=10,
        output_text="done",
        metric_deltas={
            "vllm:prefix_cache_hits": 9_500,
            "vllm:prefix_cache_queries": 10_000,
        },
    )

    assert measurement.tpot_seconds is not None
    assert abs(measurement.tpot_seconds - 0.1) < 1e-9
    assert measurement.cache_hit_rate == 0.95


def test_repository_prefix_reaches_target_and_is_stable() -> None:
    first = build_repository_prefix(2_000)
    second = build_repository_prefix(2_000)

    assert len(first) >= 2_000
    assert first == second
    assert "handler_7" in first
