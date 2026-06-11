"""Monte Carlo replication driver for the simulation study (referee R2-12).

Re-runs the simulation experiment over many fresh DGP draws and collects the
out-of-sample metric distribution (mean +/- SD) for every model/variant, instead
of a single seed. This is the "number of replications" the referee asked for.

Design (decided 2026-06-08, see paper/reviews/JoE/r2_revision_2026-06-08.md):
  * Option (c): re-simulate AND retrain every model per replication.
  * HP are NOT re-tuned per draw. We reuse the already-selected HP, held fixed per
    variant (full_final_params.yaml), and pin them with single-value `hyperopt`
    lists. With n_trials=k, Optuna trains k inits at that fixed HP and keeps the
    lowest-val-loss one -> best-of-k-by-val (no re-tuning, no leakage), faithful to
    the published method and comparable to the deterministic AR/MIDAS fits.
  * Study A (primary): fix training.seed (init), vary simulation.seed (path).

Canonical regime -> (scenario, date) mapping (reconciled to Table 1 by
src/evaluation/build_simulation_table.py, 72/72 cells):
    linear -> lss/2025-10-17     mild -> nss/2025-10-17     high -> nss/2025-10-20

Resumable: a (regime, seed, variant) unit is "done" if its experiment folder has
both transformer_preds_*.csv and rmse_summary.csv. Re-running the same command
skips done units; a seed with any missing variant regenerates its (deterministic)
data + AR + MIDAS, then trains only the missing variants. Safe to Ctrl-C and rerun.

Progress: writes outputs/replications/<tag>_progress.txt every unit (a text bar +
ETA). Watch it live in another shell with:
    while :; do clear; cat outputs/replications/<tag>_progress.txt; sleep 5; done

Usage:
    python src/run_simulation_replications.py --regimes high --seeds 123 --variants full --epochs 2  # smoke
    python src/run_simulation_replications.py --regimes all --n 100 --k-inits 5 --workers 4 --tag repl100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments"
REPL_DIR = PROJECT_ROOT / "outputs" / "replications"

SCRIPTS = {
    "simulate": PROJECT_ROOT / "src" / "data" / "simulate_to_long.py",
    "train": PROJECT_ROOT / "src" / "train.py",
    "ar": PROJECT_ROOT / "src" / "models" / "ar.py",
    "midas": PROJECT_ROOT / "src" / "models" / "midas.R",
    "evaluate": PROJECT_ROOT / "src" / "evaluation" / "evaluate_forecasts.py",
}

REGIMES = {  # regime key -> (scenario tag, experiment date)
    "linear": ("lss", "2025-10-17"),
    "mild": ("nss", "2025-10-17"),
    "high": ("nss", "2025-10-20"),
}

VARIANTS = [  # variant key -> canonical folder prefix
    ("full", "synth_mixed_frequency_transformer"),
    ("AB1", "synth_B1_no_nonlinearity"),
    ("AB2", "synth_B2_no_attention"),
    ("AB3", "synth_B3_no_attention_no_nonlinearity"),
    ("AB4", "synth_B5_no_positional_encoding"),
    ("AB5", "synth_B6_y_only"),
]

HP_KEYS = ["d_model", "nhead", "num_layers", "dropout", "lr",
           "d_freq", "d_var", "dim_feedforward", "activation"]

# Directory holding the per-variant canonical {used_config,full_final_params}.yaml.
# Defaults to outputs/experiments; override with --bases-dir on a fresh checkout (e.g.
# the cluster) using the committed bundle src/config/sim_replication_bases.
BASES_DIR = EXPERIMENT_DIR

# Optional simulation-dimension overrides (set from CLI in main): p_x, p_y, latent_dim,
# nonlinearity_intensity. Used to test wider cross-sections / more factors than the base DGP.
SIM_OVERRIDES = {}


def canonical_folder(prefix: str, regime: str) -> Path:
    tag, date = REGIMES[regime]
    return BASES_DIR / f"{prefix}_{tag}_{date}"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_variant_config(prefix: str, regime: str, sim_seed: int, init_seed: int,
                         exp_name: str, epochs: int | None, k_inits: int) -> dict:
    """Variant's canonical used_config (correct ablation flags + sim params) with
    seeds/experiment_name overridden and the tuned HP pinned (best-of-k-by-val)."""
    folder = canonical_folder(prefix, regime)
    cfg = deepcopy(load_yaml(folder / "used_config.yaml"))
    hp = load_yaml(folder / "full_final_params.yaml")

    cfg["simulation"]["seed"] = sim_seed
    for k, v in SIM_OVERRIDES.items():     # widen cross-section / change factor count, etc.
        cfg["simulation"][k] = v
    cfg["training"]["seed"] = init_seed
    cfg["training"]["optimize"] = True
    cfg["training"]["experiment_name"] = exp_name
    if epochs is not None:
        cfg["training"]["epochs"] = epochs

    cfg.setdefault("hyperopt", {})
    cfg["hyperopt"]["n_trials"] = k_inits
    cfg["hyperopt"]["study_name"] = None
    for k in HP_KEYS:
        if k in hp and hp[k] is not None:
            cfg["hyperopt"][k] = [hp[k]]

    # Per-seed intermediate file names so concurrent seeds (e.g. a SLURM array) never
    # collide on the shared filesystem. transformer_preds stay in the unique exp folder
    # (train.py writes them there with the plain suffix), so they are left untouched.
    # ABSOLUTE paths: midas.R resolves the data path relative to its working directory,
    # which on the cluster is NOT the project root (~/.bashrc / R profile changes it).
    # Python steps build paths from project_root regardless; MIDAS needs the absolute path.
    base = str(PROJECT_ROOT)
    cfg.setdefault("paths", {})
    cfg["paths"]["data_processed_template_simulation"] = (
        f"{base}/data/processed/long_format_synth_s{sim_seed}_{{suffix}}.csv")
    cfg["paths"].setdefault("outputs", {})
    cfg["paths"]["outputs"]["ar_preds"] = f"{base}/outputs/ar_preds_s{sim_seed}_{{suffix}}.csv"
    cfg["paths"]["outputs"]["midas_preds"] = f"{base}/outputs/midas_preds_s{sim_seed}_{{suffix}}.csv"
    return cfg


def unit_done(exp_name: str) -> bool:
    folder = EXPERIMENT_DIR / exp_name
    return (next(folder.glob("transformer_preds_*.csv"), None) is not None
            and (folder / "rmse_summary.csv").exists())


def run(cmd: list[str], label: str) -> None:
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n"
                           f"STDOUT:\n{result.stdout[-1500:]}\nSTDERR:\n{result.stderr[-1500:]}")


def render_progress(tag: str, done: int, total: int, t0: float, current: str) -> None:
    frac = done / total if total else 1.0
    bar = "#" * int(40 * frac) + "-" * (40 - int(40 * frac))
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 and done else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")

    def hms(s):
        if s == float("inf"):
            return "??"
        h, r = divmod(int(s), 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    line = (f"[{bar}] {done}/{total} ({frac*100:4.1f}%)  "
            f"elapsed {hms(elapsed)}  ETA {hms(eta)}  now: {current}")
    (REPL_DIR / f"{tag}_progress.txt").write_text(line + "\n")
    print(line, flush=True)


def train_eval_variant(py: str, cfg_path: Path) -> None:
    run([py, str(SCRIPTS["train"]), "--config", str(cfg_path)], "train")
    run([py, str(SCRIPTS["evaluate"]), "--config", str(cfg_path)], "evaluate")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", default="all")
    p.add_argument("--variants", default="all")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--seeds", default=None, help="explicit comma list (overrides --n)")
    p.add_argument("--study", choices=["path", "init", "both"], default="path")
    p.add_argument("--init-seed", type=int, default=125)
    p.add_argument("--fixed-sim-seed", type=int, default=123)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--k-inits", type=int, default=5)
    p.add_argument("--workers", type=int, default=4, help="concurrent variant trainings per seed (MPS: <=6)")
    p.add_argument("--tag", default="repl")
    p.add_argument("--bases-dir", default=None,
                   help="dir with canonical {used_config,full_final_params}.yaml per variant "
                        "(default outputs/experiments; use src/config/sim_replication_bases on the cluster)")
    p.add_argument("--sim-px", type=int, default=None, help="override simulation.p_x (monthly cross-section)")
    p.add_argument("--sim-py", type=int, default=None, help="override simulation.p_y (quarterly cross-section)")
    p.add_argument("--sim-q", type=int, default=None, help="override simulation.latent_dim (# factors)")
    p.add_argument("--sim-almon-flat", action="store_true",
                   help="flatten the target's Almon lag weights so within-quarter monthly factors matter")
    p.add_argument("--sim-wq-avg", action="store_true",
                   help="target depends on the within-quarter AVERAGE of the monthly factor (flow)")
    p.add_argument("--sim-factor-rho", type=float, default=None,
                   help="latent factor VAR spectral radius (default 0.9; lower => weaker own-history)")
    p.add_argument("--sim-spec-y", type=float, default=None,
                   help="quarterly target AR spectral radius (default 0.4; lower => less self-prediction)")
    p.add_argument("--rscript", default="/opt/homebrew/bin/Rscript")
    p.add_argument("--skip-midas", action="store_true")
    p.add_argument("--no-manifest", action="store_true",
                   help="do not write the manifest (use for per-seed SLURM array tasks)")
    p.add_argument("--collect-only", action="store_true",
                   help="skip all runs; scan the full plan and write the unified manifest, then exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    global BASES_DIR
    if args.bases_dir:
        BASES_DIR = Path(args.bases_dir) if Path(args.bases_dir).is_absolute() \
            else PROJECT_ROOT / args.bases_dir
    if args.sim_px is not None:
        SIM_OVERRIDES["p_x"] = args.sim_px
    if args.sim_py is not None:
        SIM_OVERRIDES["p_y"] = args.sim_py
    if args.sim_q is not None:
        SIM_OVERRIDES["latent_dim"] = args.sim_q
    if args.sim_almon_flat:
        SIM_OVERRIDES["almon_flat"] = True
    if args.sim_wq_avg:
        SIM_OVERRIDES["within_quarter_avg"] = True
    if args.sim_factor_rho is not None:
        SIM_OVERRIDES["factor_spectral_target"] = args.sim_factor_rho
    if args.sim_spec_y is not None:
        SIM_OVERRIDES["spectral_target_y"] = args.sim_spec_y

    regimes = list(REGIMES) if args.regimes == "all" else args.regimes.split(",")
    variants = VARIANTS if args.variants == "all" else [v for v in VARIANTS if v[0] in args.variants.split(",")]
    sim_seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                 else list(range(args.start_seed, args.start_seed + args.n)))

    if args.study == "path":
        pairs = [(s, args.init_seed) for s in sim_seeds]
    elif args.study == "init":
        pairs = [(args.fixed_sim_seed, args.init_seed + i) for i in range(len(sim_seeds))]
    else:
        pairs = [(s, args.init_seed + i) for i, s in enumerate(sim_seeds)]

    REPL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = REPL_DIR / "_tmp_configs"
    tmp_dir.mkdir(exist_ok=True)

    total = len(regimes) * len(pairs) * len(variants)
    already = sum(unit_done(f"{args.tag}_{r}_ss{ss}_is{isd}_{v[0]}")
                  for r in regimes for ss, isd in pairs for v in variants)
    print(f"Plan: {total} variant-units ({len(regimes)} regimes x {len(pairs)} seeds x "
          f"{len(variants)} variants), k={args.k_inits}, workers={args.workers}. "
          f"Already done: {already} (will skip).")
    if args.dry_run:
        return

    if args.collect_only:
        rows = ["regime,sim_seed,init_seed,variant,exp_folder"]
        for regime in regimes:
            for ss, isd in pairs:
                for vk, _ in variants:
                    exp_name = f"{args.tag}_{regime}_ss{ss}_is{isd}_{vk}"
                    if unit_done(exp_name):
                        rows.append(f"{regime},{ss},{isd},{vk},{EXPERIMENT_DIR / exp_name}")
        (REPL_DIR / f"{args.tag}_manifest.csv").write_text("\n".join(rows) + "\n")
        print(f"Collected {len(rows)-1}/{total} completed units -> "
              f"{REPL_DIR / f'{args.tag}_manifest.csv'}")
        return

    py = sys.executable
    t0 = time.time()
    done = already
    failures = []

    for regime in regimes:
        for ss, isd in pairs:
            stem = f"{args.tag}_{regime}_ss{ss}_is{isd}"
            missing = [(vk, pre) for vk, pre in variants if not unit_done(f"{stem}_{vk}")]
            if not missing:
                continue  # whole seed already complete -> don't even regenerate data

            render_progress(args.tag, done, total, t0, f"{regime} ss{ss} [data+AR+MIDAS]")
            data_cfg = build_variant_config("synth_mixed_frequency_transformer", regime, ss, isd,
                                            f"{stem}_full", args.epochs, args.k_inits)
            data_cfg_path = tmp_dir / f"{stem}_data.yaml"
            data_cfg_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False))
            run([py, str(SCRIPTS["simulate"]), "--config", str(data_cfg_path)], "simulate")
            run([py, str(SCRIPTS["ar"]), "--config", str(data_cfg_path)], "ar")
            if not args.skip_midas:
                try:  # MIDAS is numerically fragile on some draws / wide panels; non-fatal
                    run([args.rscript, str(SCRIPTS["midas"]), str(data_cfg_path)], "midas")
                except Exception as e:
                    failures.append(f"{regime} ss{ss} midas: {str(e)[:200]}")
                    print(f"WARN midas failed for {regime} ss{ss} — continuing without MIDAS: {str(e)[:150]}", flush=True)

            # write per-variant configs, then train+eval the missing ones (parallel within seed)
            jobs = {}
            for vk, pre in missing:
                cfg = build_variant_config(pre, regime, ss, isd, f"{stem}_{vk}", args.epochs, args.k_inits)
                cfg_path = tmp_dir / f"{stem}_{vk}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs[vk] = cfg_path

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = {ex.submit(train_eval_variant, py, cp): vk for vk, cp in jobs.items()}
                for fut in as_completed(futs):
                    vk = futs[fut]
                    try:
                        fut.result()
                        done += 1
                        render_progress(args.tag, done, total, t0, f"{regime} ss{ss} {vk} done")
                    except Exception as e:  # do not abort the whole run; resume retries this unit
                        failures.append(f"{regime} ss{ss} {vk}: {str(e)[:300]}")
                        print(f"FAIL {regime} ss{ss} {vk}: {str(e)[:300]}", flush=True)
                        render_progress(args.tag, done, total, t0, f"FAIL {regime} ss{ss} {vk}")

    # rebuild manifest from disk (robust to resume) over the full plan
    n_on_disk = sum(unit_done(f"{args.tag}_{r}_ss{ss}_is{isd}_{vk}")
                    for r in regimes for ss, isd in pairs for vk, _ in variants)
    if not args.no_manifest:
        rows = ["regime,sim_seed,init_seed,variant,exp_folder"]
        for regime in regimes:
            for ss, isd in pairs:
                for vk, _ in variants:
                    exp_name = f"{args.tag}_{regime}_ss{ss}_is{isd}_{vk}"
                    if unit_done(exp_name):
                        rows.append(f"{regime},{ss},{isd},{vk},{EXPERIMENT_DIR / exp_name}")
        (REPL_DIR / f"{args.tag}_manifest.csv").write_text("\n".join(rows) + "\n")
    print(f"\nDone. {n_on_disk}/{total} units complete on disk.")
    if failures:
        print(f"{len(failures)} unit(s) failed this pass (re-run the same command to retry):")
        for f in failures:
            print("  -", f)


if __name__ == "__main__":
    main()
