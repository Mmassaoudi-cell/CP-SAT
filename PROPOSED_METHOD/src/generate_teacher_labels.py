from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from BENCHMARKS.src.milp_cpsat import run_cpsat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--n_scenarios", type=int, default=200)
    parser.add_argument("--time_limit_s", type=float, default=15.0)
    parser.add_argument("--out_dir", default="PROPOSED_METHOD/results/teacher_labels")
    args = parser.parse_args()
    config = load_config(args.config)
    exp = config["experiment"]
    start = int(exp["train_seed_start"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset in range(args.n_scenarios):
        seed = start + offset
        scenario = generate_scenario(seed, f"train_{offset:04d}")
        metrics, schedule, status = run_cpsat(scenario, config, time_limit_s=args.time_limit_s)
        record = {"seed": seed, "offset": offset, "scenario_id": scenario.scenario_id, "status": status, "system_cost": metrics["system_cost"], "schedule": schedule}
        with open(out_dir / f"{scenario.scenario_id}.json", "w") as f:
            json.dump(record, f)
        manifest.append({"seed": seed, "offset": offset, "scenario_id": scenario.scenario_id, "status": status, "system_cost": metrics["system_cost"], "scheduled_fraction": metrics["scheduled_fraction"]})
        if (offset + 1) % 10 == 0:
            print(json.dumps({"progress": offset + 1, "of": args.n_scenarios, "status": status}))
    import pandas as pd
    pd.DataFrame(manifest).to_csv(out_dir / "_manifest.csv", index=False)
    print(json.dumps({"done": True, "n_scenarios": args.n_scenarios, "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
