"""Build the calibrated TTFT lookup, validation report, and publication figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .summarize import percentile


Condition = tuple[int, int, int]


def _read(path: Path) -> dict[Condition, list[dict[str, Any]]]:
    groups: dict[Condition, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("type") != "calibration_result":
                raise ValueError(f"{path}: line {line_no} is not a calibration result")
            if row.get("status") != "ok":
                continue
            key = (
                int(row["prompt_tokens_target"]),
                int(row["cached_prefix_tokens_target"]),
                int(row["queue_depth_target"]),
            )
            groups[key].append(row)
    if not groups:
        raise ValueError(f"{path}: no successful calibration rows")
    return groups


def _bootstrap_median_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = [
        median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def _condition_stats(
    groups: dict[Condition, list[dict[str, Any]]], seed: int
) -> dict[Condition, dict[str, float | int]]:
    output: dict[Condition, dict[str, float | int]] = {}
    for index, key in enumerate(sorted(groups)):
        values = [float(row["ttft_ms"]) for row in groups[key]]
        low, high = _bootstrap_median_ci(values, seed + index)
        output[key] = {
            "n": len(values),
            "mean": mean(values),
            "median": median(values),
            "p95": percentile(values, 0.95),
            "median_ci_low": low,
            "median_ci_high": high,
        }
    return output


def _audit_counts(groups: dict[Condition, list[dict[str, Any]]]) -> dict[str, int]:
    rows = [row for values in groups.values() for row in values]
    return {
        "rows_ok": len(rows),
        "prompt_length_mismatches": sum(
            row.get("prompt_tokens_reported") != row.get("prompt_tokens_target") for row in rows
        ),
        "cache_hit_mismatches": sum(
            row.get("prefix_cache_hit_tokens_batch") is not None
            and row.get("prefix_cache_hit_tokens_batch")
            != row.get("cached_prefix_tokens_target")
            for row in rows
        ),
        "queue_depth_mismatches": sum(
            row.get("queue_depth_observed") is not None
            and row.get("queue_depth_observed") != row.get("queue_depth_target")
            for row in rows
        ),
    }


def _write_surface(
    path: Path,
    calibration: dict[Condition, dict[str, float | int]],
    validation: dict[Condition, dict[str, float | int]],
) -> None:
    fields = [
        "prompt_tokens",
        "cached_prefix_tokens",
        "queue_depth",
        "calibration_n",
        "expected_ttft_ms",
        "calibration_median_ttft_ms",
        "calibration_median_ci_low_ms",
        "calibration_median_ci_high_ms",
        "calibration_p95_ttft_ms",
        "validation_n",
        "validation_mean_ttft_ms",
        "validation_median_ttft_ms",
        "validation_p95_ttft_ms",
        "mean_error_ms",
        "median_error_ms",
        "p95_error_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(calibration):
            train = calibration[key]
            heldout = validation[key]
            writer.writerow(
                {
                    "prompt_tokens": key[0],
                    "cached_prefix_tokens": key[1],
                    "queue_depth": key[2],
                    "calibration_n": train["n"],
                    "expected_ttft_ms": train["mean"],
                    "calibration_median_ttft_ms": train["median"],
                    "calibration_median_ci_low_ms": train["median_ci_low"],
                    "calibration_median_ci_high_ms": train["median_ci_high"],
                    "calibration_p95_ttft_ms": train["p95"],
                    "validation_n": heldout["n"],
                    "validation_mean_ttft_ms": heldout["mean"],
                    "validation_median_ttft_ms": heldout["median"],
                    "validation_p95_ttft_ms": heldout["p95"],
                    "mean_error_ms": float(heldout["mean"]) - float(train["mean"]),
                    "median_error_ms": float(heldout["median"]) - float(train["median"]),
                    "p95_error_ms": float(heldout["p95"]) - float(train["p95"]),
                }
            )


def _save_figure(figure: Any, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=220, bbox_inches="tight")


def _plot_response(
    plt: Any,
    raw: dict[Condition, list[dict[str, Any]]],
    stats: dict[Condition, dict[str, float | int]],
    directory: Path,
) -> None:
    prompts = sorted({key[0] for key in stats})
    queues = sorted({key[2] for key in stats})
    figure, axes = plt.subplots(
        len(prompts), len(queues), figsize=(12, 10), sharey="col", constrained_layout=True
    )
    jitter = random.Random(699)
    for row_index, prompt in enumerate(prompts):
        for column_index, queue in enumerate(queues):
            axis = axes[row_index][column_index]
            keys = sorted(
                (key for key in stats if key[0] == prompt and key[2] == queue),
                key=lambda item: item[1],
            )
            x_values = [key[1] for key in keys]
            medians = [float(stats[key]["median"]) for key in keys]
            lows = [float(stats[key]["median_ci_low"]) for key in keys]
            highs = [float(stats[key]["median_ci_high"]) for key in keys]
            for key in keys:
                observations = [float(item["ttft_ms"]) for item in raw[key]]
                width = max(4.0, prompt * 0.006)
                axis.scatter(
                    [key[1] + jitter.uniform(-width, width) for _ in observations],
                    observations,
                    s=9,
                    alpha=0.22,
                    color="#4C78A8",
                    linewidths=0,
                )
            axis.plot(x_values, medians, marker="o", color="#D1495B", linewidth=1.8)
            axis.errorbar(
                x_values,
                medians,
                yerr=[
                    [middle - low for middle, low in zip(medians, lows)],
                    [high - middle for middle, high in zip(medians, highs)],
                ],
                fmt="none",
                ecolor="#D1495B",
                capsize=3,
                linewidth=1.2,
            )
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.18)
            axis.set_xticks(x_values)
            if row_index == 0:
                axis.set_title(f"Queue depth {queue}")
            if column_index == 0:
                axis.set_ylabel(f"Prompt {prompt}\nTTFT (ms, log scale)")
            if row_index == len(prompts) - 1:
                axis.set_xlabel("Cached-prefix tokens")
    figure.suptitle("TTFT calibration response", fontsize=15)
    _save_figure(figure, directory, "calibration-response")
    plt.close(figure)


def _plot_cache_benefit(
    plt: Any, stats: dict[Condition, dict[str, float | int]], directory: Path
) -> None:
    queues = sorted({key[2] for key in stats})
    prefixes = sorted({key[1] for key in stats if key[1] > 0})
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    figure, axes = plt.subplots(1, len(queues), figsize=(12, 4), constrained_layout=True)
    for axis, queue in zip(axes, queues):
        for prefix, color in zip(prefixes, colors):
            points = []
            for key, values in stats.items():
                prompt, cached, key_queue = key
                if cached != prefix or key_queue != queue:
                    continue
                baseline = stats[(prompt, 0, queue)]
                points.append((prompt, float(baseline["median"]) - float(values["median"])))
            if points:
                points.sort()
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    label=f"{prefix} cached",
                    color=color,
                )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(f"Queue depth {queue}")
        axis.set_xlabel("Prompt tokens")
        axis.grid(True, alpha=0.18)
    axes[0].set_ylabel("Median TTFT saved (ms)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Observed benefit of prefix-cache reuse", fontsize=15)
    _save_figure(figure, directory, "cache-benefit")
    plt.close(figure)


def _plot_validation(
    plt: Any,
    calibration: dict[Condition, dict[str, float | int]],
    validation: dict[Condition, dict[str, float | int]],
    directory: Path,
) -> None:
    colors = {0: "#4C78A8", 2: "#F58518", 8: "#54A24B"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("mean", "p95"), ("Expected TTFT", "p95 TTFT")
    ):
        all_values = []
        for queue in sorted(colors):
            keys = [key for key in calibration if key[2] == queue]
            x_values = [float(calibration[key][metric]) for key in keys]
            y_values = [float(validation[key][metric]) for key in keys]
            all_values.extend(x_values + y_values)
            axis.scatter(x_values, y_values, s=35, alpha=0.8, color=colors[queue], label=f"Queue {queue}")
        upper = max(all_values) * 1.05
        axis.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1)
        axis.set_xlim(0, upper)
        axis.set_ylim(0, upper)
        axis.set_title(title)
        axis.set_xlabel("Calibration estimate (ms)")
        axis.set_ylabel("Validation observation (ms)")
        axis.grid(True, alpha=0.18)
    axes[-1].legend(frameon=False)
    figure.suptitle("Held-out validation of the TTFT lookup", fontsize=15)
    _save_figure(figure, directory, "calibration-validation")
    plt.close(figure)


def analyze(calibration_path: Path, validation_path: Path, output_dir: Path) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires: python -m pip install -e '.[plots]'") from exc

    calibration_raw = _read(calibration_path)
    validation_raw = _read(validation_path)
    if set(calibration_raw) != set(validation_raw):
        missing = sorted(set(calibration_raw) - set(validation_raw))
        extra = sorted(set(validation_raw) - set(calibration_raw))
        raise ValueError(f"validation grid mismatch; missing={missing}, extra={extra}")

    calibration = _condition_stats(calibration_raw, 699)
    validation = _condition_stats(validation_raw, 1701)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "calibration-lookup.csv"
    _write_surface(surface_path, calibration, validation)

    mean_errors = [
        abs(float(validation[key]["mean"]) - float(calibration[key]["mean"]))
        for key in calibration
    ]
    median_errors = [
        abs(float(validation[key]["median"]) - float(calibration[key]["median"]))
        for key in calibration
    ]
    p95_errors = [
        abs(float(validation[key]["p95"]) - float(calibration[key]["p95"]))
        for key in calibration
    ]
    mean_relative_errors = [
        abs(float(validation[key]["mean"]) - float(calibration[key]["mean"]))
        / float(calibration[key]["mean"])
        for key in calibration
    ]
    median_relative_errors = [
        abs(float(validation[key]["median"]) - float(calibration[key]["median"]))
        / float(calibration[key]["median"])
        for key in calibration
    ]
    p95_relative_errors = [
        abs(float(validation[key]["p95"]) - float(calibration[key]["p95"]))
        / float(calibration[key]["p95"])
        for key in calibration
    ]
    report = {
        "calibration_file": str(calibration_path),
        "validation_file": str(validation_path),
        "conditions": len(calibration),
        "calibration_audit": _audit_counts(calibration_raw),
        "validation_audit": _audit_counts(validation_raw),
        "condition_level_mean_mae_ms": mean(mean_errors),
        "condition_level_median_mae_ms": mean(median_errors),
        "condition_level_p95_mae_ms": mean(p95_errors),
        "condition_level_mean_mape_percent": mean(mean_relative_errors) * 100,
        "condition_level_median_mape_percent": mean(median_relative_errors) * 100,
        "condition_level_p95_mape_percent": mean(p95_relative_errors) * 100,
        "condition_level_mean_max_abs_error_ms": max(mean_errors),
        "condition_level_p95_max_abs_error_ms": max(p95_errors),
        "routing_summary": "mean TTFT",
        "uncertainty": "bootstrap 95% confidence interval for condition median",
    }
    (output_dir / "calibration-validation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_response(plt, calibration_raw, calibration, output_dir)
    _plot_cache_benefit(plt, calibration, output_dir)
    _plot_validation(plt, calibration, validation, output_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibration-analysis"))
    args = parser.parse_args(argv)
    try:
        report = analyze(args.calibration, args.validation, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"calibration analysis failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
