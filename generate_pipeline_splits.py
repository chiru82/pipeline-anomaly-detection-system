#!/usr/bin/env python3
"""Generate balanced train/validation/test pipeline datasets.

The exported model inputs are exactly: flow rate, inlet pressure, outlet
pressure, pipe diameter, pipe length, and fluid type. Velocity is computed
inside the hydraulic simulator and is not exported as a model feature.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED = 20261111
SCENARIOS_PER_FLUID = {"train": 5_000, "validation": 1_000, "test": 500}
CLASS_NAMES = {
    0: "Normal",
    1: "Minor Leakage",
    2: "Major Leakage",
    3: "Partial Blockage",
    4: "Complete Blockage",
}
FLUIDS = {
    "Water": {
        "temperature_c": 20.0,
        "density_kg_m3": 998.206,
        "dynamic_viscosity_pa_s": 0.0010016,
        "velocity_range_m_s": (0.30, 2.50),
        "seed_offset": 101,
        "prefix": "W",
    },
    "Oil": {
        "temperature_c": 40.0,
        "density_kg_m3": 839.0555,
        "dynamic_viscosity_pa_s": 0.026849776,
        "velocity_range_m_s": (0.08, 1.20),
        "seed_offset": 202,
        "prefix": "O",
    },
}

PIPE_ROUGHNESS_M = 0.000045
DISCHARGE_COEFFICIENT = 0.62
MIN_NORMAL_DP_BAR = 0.002
MIN_OUTLET_PRESSURE_BAR = 0.20

INPUT_FIELDS = [
    "flow_rate_l_min",
    "inlet_pressure_bar",
    "outlet_pressure_bar",
    "pipe_diameter_mm",
    "pipe_length_m",
    "fluid_type",
]
FIELDNAMES = INPUT_FIELDS + [
    "class_id",
    "class_label",
    "data_origin",
    "record_id",
    "scenario_id",
    "split",
]


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def friction_factor(reynolds: float, diameter_m: float) -> float:
    if reynolds <= 0.0:
        return 0.0
    if reynolds < 2_000.0:
        return 64.0 / reynolds
    relative_roughness = PIPE_ROUGHNESS_M / diameter_m
    turbulent = 0.25 / math.log10(
        relative_roughness / 3.7 + 5.74 / reynolds**0.9
    ) ** 2
    if reynolds >= 4_000.0:
        return turbulent
    laminar = 64.0 / reynolds
    t = (reynolds - 2_000.0) / 2_000.0
    smooth_t = t * t * (3.0 - 2.0 * t)
    return laminar * (1.0 - smooth_t) + turbulent * smooth_t


def pressure_loss_bar(
    flow_m3_s: float,
    length_m: float,
    diameter_m: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
) -> float:
    if flow_m3_s <= 0.0 or length_m <= 0.0:
        return 0.0
    area = math.pi * diameter_m**2 / 4.0
    velocity = flow_m3_s / area
    reynolds = density_kg_m3 * velocity * diameter_m / viscosity_pa_s
    f = friction_factor(reynolds, diameter_m)
    dp_pa = f * (length_m / diameter_m) * density_kg_m3 * velocity**2 / 2.0
    return dp_pa / 100_000.0


def bisect(function, target: float, low: float, high: float) -> float:
    f_low = function(low) - target
    f_high = function(high) - target
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError("Bisection interval does not bracket target")
    for _ in range(90):
        mid = (low + high) / 2.0
        f_mid = function(mid) - target
        if abs(f_mid) <= max(1e-13, target * 1e-12):
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2.0


def feature_key(row: dict) -> tuple[str, ...]:
    return (
        f"{float(row['flow_rate_l_min']):.9f}",
        f"{float(row['inlet_pressure_bar']):.6f}",
        f"{float(row['outlet_pressure_bar']):.6f}",
        f"{float(row['pipe_diameter_mm']):.8f}",
        f"{float(row['pipe_length_m']):.6f}",
        row["fluid_type"],
    )


def make_row(
    fluid_name: str,
    fluid: dict,
    split: str,
    scenario_index: int,
    class_id: int,
    diameter_m: float,
    length_m: float,
    inlet_pressure_bar: float,
    outlet_pressure_bar: float,
    flow_m3_s: float,
) -> dict:
    area = math.pi * diameter_m**2 / 4.0
    velocity = flow_m3_s / area if flow_m3_s > 0.0 else 0.0
    flow_l_min = flow_m3_s * 60_000.0
    prefix = fluid["prefix"]
    split_code = {"train": "TR", "validation": "VA", "test": "TE"}[split]
    record_id = f"{prefix}-{split_code}-{scenario_index * 5 + class_id + 1:06d}"
    scenario_id = f"{prefix}-{split_code}-SC-{scenario_index + 1:05d}"
    return {
        "flow_rate_l_min": round(flow_l_min, 9),
        "inlet_pressure_bar": round(inlet_pressure_bar, 6),
        "outlet_pressure_bar": round(outlet_pressure_bar, 6),
        "pipe_diameter_mm": round(diameter_m * 1_000.0, 8),
        "pipe_length_m": round(length_m, 6),
        "fluid_type": fluid_name,
        "class_id": class_id,
        "class_label": CLASS_NAMES[class_id],
        "data_origin": "simulated",
        "record_id": record_id,
        "scenario_id": scenario_id,
        "split": split,
        # Private in-memory fields support the continuity check; they are not
        # included in FIELDNAMES and therefore never appear in the CSVs.
        "_flow_m3_s": flow_m3_s,
        "_derived_velocity_m_s": velocity,
    }


def create_scenario(
    rng: random.Random,
    fluid_name: str,
    fluid: dict,
    split: str,
    scenario_index: int,
) -> list[dict] | None:
    rho = fluid["density_kg_m3"]
    mu = fluid["dynamic_viscosity_pa_s"]
    v_low, v_high = fluid["velocity_range_m_s"]

    for _ in range(2_000):
        diameter = log_uniform(rng, 0.025, 0.150)
        length = log_uniform(rng, 5.0, 200.0)
        p1 = rng.uniform(3.0, 12.0)
        v_delivery = rng.uniform(v_low, v_high)
        area = math.pi * diameter**2 / 4.0
        normal_flow = area * v_delivery
        normal_drop = pressure_loss_bar(normal_flow, length, diameter, rho, mu)
        if normal_drop < MIN_NORMAL_DP_BAR or normal_drop >= p1 - MIN_OUTLET_PRESSURE_BAR:
            continue
        p2 = p1 - normal_drop

        leak_position = rng.uniform(0.15, 0.85)
        upstream_length = length * leak_position
        downstream_length = length - upstream_length

        def leak_inlet_flow(alpha: float) -> tuple[float, float] | None:
            def two_segment_drop(q_in: float) -> float:
                q_out = q_in * (1.0 - alpha)
                return pressure_loss_bar(q_in, upstream_length, diameter, rho, mu) + pressure_loss_bar(
                    q_out, downstream_length, diameter, rho, mu
                )

            try:
                q_in = bisect(two_segment_drop, normal_drop, normal_flow, normal_flow / (1.0 - alpha))
            except ValueError:
                return None
            q_out = q_in * (1.0 - alpha)
            upstream_drop = pressure_loss_bar(q_in, upstream_length, diameter, rho, mu)
            p_at_leak = p1 - upstream_drop
            if p_at_leak <= 0.0:
                return None
            q_leak = q_in - q_out
            opening_area = q_leak / (
                DISCHARGE_COEFFICIENT * math.sqrt(2.0 * p_at_leak * 100_000.0 / rho)
            )
            opening_diameter = math.sqrt(4.0 * opening_area / math.pi)
            if opening_diameter > 0.30 * diameter:
                return None
            return q_in, opening_diameter

        minor_fraction = rng.uniform(0.01, 0.10)
        major_fraction = rng.uniform(0.10, 0.35)
        minor_flow = leak_inlet_flow(minor_fraction)
        major_flow = leak_inlet_flow(major_fraction)
        if minor_flow is None or major_flow is None:
            continue

        blockage_reduction = rng.uniform(0.35, 0.85)
        open_area_ratio = 1.0 - blockage_reduction
        local_k = (1.0 / DISCHARGE_COEFFICIENT**2) * (1.0 / open_area_ratio**2 - 1.0)

        def partial_drop(q: float) -> float:
            pipe_drop = pressure_loss_bar(q, length, diameter, rho, mu)
            velocity = q / area if q > 0.0 else 0.0
            local_pa = local_k * rho * velocity**2 / 2.0
            return pipe_drop + local_pa / 100_000.0

        try:
            partial_flow = bisect(partial_drop, normal_drop, 0.0, normal_flow)
        except ValueError:
            continue

        rows = [
            make_row(fluid_name, fluid, split, scenario_index, 0, diameter, length, p1, p2, normal_flow),
            make_row(fluid_name, fluid, split, scenario_index, 1, diameter, length, p1, p2, minor_flow[0]),
            make_row(fluid_name, fluid, split, scenario_index, 2, diameter, length, p1, p2, major_flow[0]),
            make_row(fluid_name, fluid, split, scenario_index, 3, diameter, length, p1, p2, partial_flow),
            make_row(fluid_name, fluid, split, scenario_index, 4, diameter, length, p1, p2, 0.0),
        ]
        keys = [feature_key(row) for row in rows]
        if len(set(keys)) != len(keys):
            continue
        # Keep the opening and blockage settings out of the model CSV. They
        # determine class behavior internally and are intentionally not features.
        return rows
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict], split: str) -> dict:
    expected_scenarios = SCENARIOS_PER_FLUID[split]
    expected_fluids = set(FLUIDS)
    expected_per_class_per_fluid = expected_scenarios
    errors = []
    counts = Counter((row["fluid_type"], row["class_id"]) for row in rows)
    for fluid in expected_fluids:
        for class_id in CLASS_NAMES:
            if counts[(fluid, class_id)] != expected_per_class_per_fluid:
                errors.append(f"{split}/{fluid}/class {class_id} has {counts[(fluid, class_id)]} rows")

    keys = [feature_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append(f"Duplicate six-input combinations in {split}")
    if len({row["record_id"] for row in rows}) != len(rows):
        errors.append(f"Duplicate record IDs in {split}")
    if any(row["split"] != split for row in rows):
        errors.append(f"Wrong split marker in {split}")
    if any(row["data_origin"] != "simulated" for row in rows):
        errors.append(f"Wrong data_origin marker in {split}")

    scenario_groups: dict[str, list[dict]] = {}
    for row in rows:
        scenario_groups.setdefault(row["scenario_id"], []).append(row)
    for scenario_id, group in scenario_groups.items():
        if sorted(row["class_id"] for row in group) != list(CLASS_NAMES):
            errors.append(f"Scenario missing class rows: {scenario_id}")
            break

    max_continuity_error = 0.0
    for row in rows:
        q = float(row["flow_rate_l_min"])
        d_m = float(row["pipe_diameter_mm"]) / 1_000.0
        area = math.pi * d_m**2 / 4.0
        derived_v = row["_derived_velocity_m_s"]
        q_from_v = area * derived_v * 60_000.0
        max_continuity_error = max(max_continuity_error, abs(q - q_from_v))
        p1 = float(row["inlet_pressure_bar"])
        p2 = float(row["outlet_pressure_bar"])
        if p1 <= p2 or p2 <= 0.0:
            errors.append(f"Invalid pressure ordering in {row['record_id']}")
            break
        if row["class_id"] == 4 and q != 0.0:
            errors.append(f"Complete blockage has flow in {row['record_id']}")
            break
        if row["class_id"] != 4 and q <= 0.0:
            errors.append(f"Non-complete class has zero flow in {row['record_id']}")
            break

    if max_continuity_error > 0.000003:
        errors.append(f"Continuity check exceeded tolerance: {max_continuity_error}")
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors[:10]))

    split_counts = Counter((row["class_id"], row["fluid_type"]) for row in rows)
    ranges = {
        field: (min(float(row[field]) for row in rows), max(float(row[field]) for row in rows))
        for field in INPUT_FIELDS[:-1]
    }
    return {
        "rows": len(rows),
        "unique_inputs": len(set(keys)),
        "unique_record_ids": len({row["record_id"] for row in rows}),
        "scenario_groups": len(scenario_groups),
        "max_continuity_error": max_continuity_error,
        "counts": split_counts,
        "ranges": ranges,
    }


def main() -> None:
    rng = random.Random(SEED)
    per_split: dict[str, list[dict]] = {name: [] for name in SCENARIOS_PER_FLUID}
    all_feature_keys: set[tuple[str, ...]] = set()

    for fluid_name, fluid in FLUIDS.items():
        fluid_rng = random.Random(SEED + fluid["seed_offset"])
        for split, scenario_count in SCENARIOS_PER_FLUID.items():
            for scenario_index in range(scenario_count):
                for _ in range(100):
                    rows = create_scenario(fluid_rng, fluid_name, fluid, split, scenario_index)
                    if rows is None:
                        continue
                    keys = [feature_key(row) for row in rows]
                    if any(key in all_feature_keys for key in keys):
                        continue
                    all_feature_keys.update(keys)
                    per_split[split].extend(rows)
                    break
                else:
                    raise RuntimeError(f"Could not build unique {fluid_name}/{split} scenario {scenario_index}")

    for rows in per_split.values():
        rng.shuffle(rows)
    combined = per_split["train"] + per_split["validation"] + per_split["test"]
    rng.shuffle(combined)

    results = {
        split: validate(rows, split)
        for split, rows in per_split.items()
    }
    combined_keys = [feature_key(row) for row in combined]
    if len(combined) != 65_000 or len(combined_keys) != len(set(combined_keys)):
        raise RuntimeError("Combined file did not pass size or uniqueness check")

    write_csv(ROOT / "train_dataset_50000.csv", per_split["train"])
    write_csv(ROOT / "validation_dataset_10000.csv", per_split["validation"])
    write_csv(ROOT / "test_dataset_5000.csv", per_split["test"])
    write_csv(ROOT / "combined_dataset_65000.csv", combined)

    with (ROOT / "dataset_class_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "fluid_type", "class_id", "class_label", "records"])
        for split, count in SCENARIOS_PER_FLUID.items():
            for fluid_name in FLUIDS:
                for class_id, label in CLASS_NAMES.items():
                    writer.writerow([split, fluid_name, class_id, label, count])

    with (ROOT / "dataset_validation_summary.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Seed: {SEED}\n")
        stream.write("Exported model inputs: " + ", ".join(INPUT_FIELDS) + "\n")
        stream.write("\n")
        for split, result in results.items():
            stream.write(f"{split}: {result['rows']} rows\n")
            stream.write(f"  unique six-input combinations: {result['unique_inputs']}\n")
            stream.write(f"  unique record IDs: {result['unique_record_ids']}\n")
            stream.write(f"  scenario groups: {result['scenario_groups']}\n")
            stream.write(f"  max continuity difference: {result['max_continuity_error']:.12g} L/min\n")
            for class_id, label in CLASS_NAMES.items():
                fluid_counts = [result["counts"][(class_id, f)] for f in FLUIDS]
                stream.write(
                    f"  class {class_id} {label}: total={sum(fluid_counts)}, "
                    f"water={fluid_counts[0]}, oil={fluid_counts[1]}\n"
                )
            stream.write("  feature ranges:\n")
            for field, (low, high) in result["ranges"].items():
                stream.write(f"    {field}: {low:.9g} to {high:.9g}\n")
            stream.write("\n")
        stream.write(f"combined: {len(combined)} rows\n")
        stream.write(f"combined unique six-input combinations: {len(set(combined_keys))}\n")

    print(
        "Generated train=50,000, validation=10,000, test=5,000, combined=65,000; "
        f"unique_combined_inputs={len(set(combined_keys))}"
    )


if __name__ == "__main__":
    main()
