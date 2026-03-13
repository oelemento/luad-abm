"""Quick test: margin seeding + higher recruit boost on CASE12/CASE18.

Tests whether spatial initialization at tumor margin and/or stronger
PD1 recruitment boost fixes the high-ratio/low-CD8 discrepancy.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gaglia_abm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.bayesian_inference import PARAM_NAMES, FIXED_PARAMS, TICKS_PER_WEEK
from luad.model import LUADModel, PresetConfig
from luad.metrics import MetricsTracker
from luad.agents import AgentType

TOTAL_WEEKS = 9
TOTAL_TICKS = TOTAL_WEEKS * TICKS_PER_WEEK
WK7_START = 7 * TICKS_PER_WEEK
WK8_END = 8 * TICKS_PER_WEEK

ARMS = {
    "untreated": [],
    "PD1_CTLA4": [(WK7_START, WK8_END, ["PD1", "CTLA4"])],
}

# Only test the two problem patients
TEST_PATIENTS = {
    "CASE12": {"tumor": 3917, "cd8": 1085, "cd4": 325, "treg": 205, "macrophage": 2468},
    "CASE18": {"tumor": 5500, "cd8": 694, "cd4": 102, "treg": 147, "macrophage": 1557},
}

# Conditions to test
CONDITIONS = [
    ("baseline_v8", {}),
    ("boost_5x", {"pd1_recruit_boost": 5.0}),
    ("boost_8x", {"pd1_recruit_boost": 8.0}),
    ("margin_seed", {"_margin_seed": True}),
    ("margin_seed+boost_5x", {"pd1_recruit_boost": 5.0, "_margin_seed": True}),
]


def run_sim(params_vec, init_agents, schedule, seed, overrides=None, margin_seed=False):
    p = dict(zip(PARAM_NAMES, params_vec))
    preset = PresetConfig(
        name="test",
        grid_width=100, grid_height=100,
        initial_agents=init_agents,
        cxcl9_10_mean=0.12,
        macrophage_bias=-0.1,
        macrophage_m2_fraction=0.45,
        suppression={"treg": float(p["treg_suppression"]),
                     "macrophage": float(p["macrophage_suppr_base"])},
        suppressive_background=float(p["suppressive_background"]),
    )

    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=[], seed=seed,
                      metrics_tracker=tracker)

    for key in PARAM_NAMES:
        if key not in ("treg_suppression", "treg_death_rate"):
            model.params[key] = float(p[key])
    for key, val in FIXED_PARAMS.items():
        model.params[key] = val
    model.params["treg_death_rate"] = float(p["treg_death_rate"])

    if overrides:
        for k, v in overrides.items():
            if k.startswith("_"):
                continue
            model.params[k] = v

    # Margin seeding: move CD8s to tumor-stroma interface
    if margin_seed:
        _reseat_cd8s_at_margin(model)

    start_events, end_events = {}, {}
    for start_tick, end_tick, interventions in schedule:
        start_events.setdefault(start_tick, []).extend(interventions)
        end_events.setdefault(end_tick, []).extend(interventions)

    for tick in range(TOTAL_TICKS):
        if tick in start_events:
            model.apply_interventions(start_events[tick])
        if tick in end_events:
            model.remove_interventions(end_events[tick])
        model.step()

    def count(at):
        return sum(1 for a in model.agents if getattr(a, "agent_type", None) == at)

    return {"n_tumor": count(AgentType.TUMOR), "n_cd8": count(AgentType.CD8),
            "n_treg": count(AgentType.TREG), "seed": seed}


def _reseat_cd8s_at_margin(model):
    """Move all CD8 T cells to positions adjacent to tumor cells."""
    import numpy as np

    # Find tumor positions
    tumor_positions = set()
    for agent in model.scheduler.agents:
        if getattr(agent, "agent_type", None) == AgentType.TUMOR:
            tumor_positions.add(agent.pos)

    if not tumor_positions:
        return

    # Find empty positions adjacent to tumor (the margin)
    margin_positions = []
    for tpos in tumor_positions:
        neighbors = model.grid.get_neighborhood(tpos, moore=True, include_center=False)
        for npos in neighbors:
            if model.grid.is_cell_empty(npos) and npos not in tumor_positions:
                margin_positions.append(npos)

    margin_positions = list(set(margin_positions))
    if not margin_positions:
        return

    # Move CD8s to margin
    cd8_cells = [a for a in model.scheduler.agents
                 if getattr(a, "agent_type", None) == AgentType.CD8]

    rng = model.np_rng
    rng.shuffle(margin_positions)

    for i, cd8 in enumerate(cd8_cells):
        if i >= len(margin_positions):
            break
        new_pos = margin_positions[i]
        if model.grid.is_cell_empty(new_pos):
            model.grid.move_agent(cd8, new_pos)


def _run_worker(args):
    return run_sim(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--out", type=str, default="outputs/test_margin_recruit")
    args = parser.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = np.load(args.posterior)
    params_mean = samples.mean(axis=0).astype(np.float32)
    print(f"Posterior mean ({len(params_mean)} params)")

    seeds = [1000 + s for s in range(args.seeds)]

    print(f"\nTesting {len(CONDITIONS)} conditions × {len(TEST_PATIENTS)} patients "
          f"× {args.seeds} seeds × 2 arms = {len(CONDITIONS) * len(TEST_PATIENTS) * args.seeds * 2} sims")
    print(f"Workers: {args.workers}\n")

    executor = ProcessPoolExecutor(max_workers=args.workers)
    all_results = {}

    for cond_name, overrides in CONDITIONS:
        margin_seed = overrides.pop("_margin_seed", False)
        for patient_name, init_agents in TEST_PATIENTS.items():
            for arm_name, schedule in ARMS.items():
                label = f"{cond_name}__{patient_name}__{arm_name}"
                print(f"=== {label} ({args.seeds} seeds) ===", flush=True)

                futures = {
                    executor.submit(_run_worker,
                                    (params_mean, init_agents, schedule, seed, overrides, margin_seed)): seed
                    for seed in seeds
                }
                seed_results = []
                for future in as_completed(futures):
                    r = future.result()
                    seed_results.append(r)
                    print(f"  seed {r['seed']}: tumor={r['n_tumor']}, cd8={r['n_cd8']}", flush=True)
                all_results[label] = sorted(seed_results, key=lambda r: r["seed"])

        # Restore _margin_seed for next iteration
        if margin_seed:
            overrides["_margin_seed"] = True

        # Checkpoint
        np.savez(out_dir / "test_results_checkpoint.npz",
                 **{k: v for k, v in all_results.items()})
        print(f"  [checkpoint: {len(all_results)} conditions]\n", flush=True)

    executor.shutdown(wait=False)

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Condition':<30} {'Patient':<10} {'Untreated':>10} {'PD1+CTLA4':>12} {'Delta%':>8}")
    print(f"{'='*90}")
    for cond_name, _ in CONDITIONS:
        for patient_name in TEST_PATIENTS:
            ut_key = f"{cond_name}__{patient_name}__untreated"
            tx_key = f"{cond_name}__{patient_name}__PD1_CTLA4"
            ut = np.mean([r["n_tumor"] for r in all_results[ut_key]])
            tx = np.mean([r["n_tumor"] for r in all_results[tx_key]])
            delta = (tx - ut) / ut * 100 if ut > 0 else 0
            print(f"{cond_name:<30} {patient_name:<10} {ut:>10.0f} {tx:>12.0f} {delta:>+7.1f}%")
    print()

    np.savez(out_dir / "test_results.npz",
             **{k: v for k, v in all_results.items()})
    print(f"Saved: {out_dir / 'test_results.npz'}")


if __name__ == "__main__":
    main()
