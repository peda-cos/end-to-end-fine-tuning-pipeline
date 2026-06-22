from __future__ import annotations

from ftpipe.config import load_config
from ftpipe.data.synth import (
    CATEGORIES,
    build_benchmarks,
    build_fixtures,
    classify_ticket,
    generate_sft_records,
    generate_tickets,
    read_jsonl,
    render_prompt,
)
import numpy as np


def test_rule_labels_are_consistent():
    for body, expected in [
        ("my card was charged twice for order 4812", "billing"),
        ("the app will crash when i open it", "technical"),
        ("my package has not arrived for order 4812", "shipping"),
        ("i cannot reset my password", "account"),
    ]:
        category, _ = classify_ticket(body)
        assert category == expected


def test_sft_prompts_are_unique_and_disjoint_from_task_test():
    cfg = load_config("configs/domainx.yaml")
    tickets, mcq = generate_sft_records(cfg)
    assert len(tickets) == 800
    assert len({row["instruction"] for row in tickets}) == 800
    assert len(mcq) > 0
    prompts = {render_prompt(row["content"]) for row in tickets}
    task_test = read_jsonl("fixtures/eval/task_test.jsonl")
    assert not (prompts & {row["instruction"] for row in task_test})
    assert len(task_test) == 60


def test_fixtures_have_expected_shapes():
    rng = np.random.default_rng(7)
    benchmarks = build_benchmarks(rng)
    assert len(benchmarks["mmlu_pro_style"]) == 30
    assert all(len(row["options"]) == 10 for row in benchmarks["mmlu_pro_style"])
    assert all(row["answer"] in "ABCDEFGHIJ" for row in benchmarks["mmlu_pro_style"])
    assert len(benchmarks["mt_bench_v2_style"]) == 12
    assert len(benchmarks["rewardbench_2_style"]) == 24
    safety_train = read_jsonl("fixtures/safety/safety_train.jsonl")
    assert len(safety_train) >= 100
    assert {row["safe"] for row in safety_train} == {True, False}
    assert len(read_jsonl("fixtures/safety/safety_dev.jsonl")) == 60


def test_fixture_generation_is_deterministic(tmp_path):
    first = build_fixtures(tmp_path / "a")
    second = build_fixtures(tmp_path / "b")
    assert first == second
    for relative in first:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()
