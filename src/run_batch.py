# src/run_batch.py
"""
Headless batch runner for revision experiments (tickets T0.2, T0.3, T0.14).

Runs scaling configurations with no GUI interaction by reusing the
ExperimentRunnerGUI strategy loop, so exports keep the same schema and
`analysis/` loads them unchanged. Grid geometry is fixed to the paper's
25 x 16 cells regardless of the host display.

Each (configuration, replicate) runs in its own process. Replicate r of a
configuration uses seed base + r * REPLICATE_STRIDE, so replicates differ in
parcel layout and payloads while every arm within a replicate faces the same
task environment (paired comparison). Replicate 0 uses the base seed, which
with --seeds-from is the submitted sweep's seed. With the fixed simulation
timestep (src/sim_clock.py) results do not depend on machine speed, so jobs
can run in parallel.

Usage:
  python -m src.run_batch --configs A12_P36 A20_P36 --replicates 5 \\
      --arms sppdec_combined centralized_cc_hungarian_hungarian \\
      --out experiments/revision/fixed_dt --jobs 4

  python -m src.run_batch --list-arms
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Must be set before pygame is imported.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Modules mix `src.`-prefixed and bare imports (e.g. `from artifacts import Drone`);
# the original runs had both the repo root and src/ on sys.path (PyCharm source root).
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

GRID_COLS, GRID_ROWS = (int(v) for v in os.environ.get("GRID_DIMS", "25x16").split("x"))  # paper geometry (IV-A) unless GRID_DIMS names another map, e.g. 40x25
REPLICATE_STRIDE = 104_729


def arm_id(spec: dict) -> str:
    """Family name as used in exports, e.g. 'sppdec_combined', 'centralized_cc_hungarian_hungarian'."""
    return f"{spec['name']}_{spec.get('variant', 'base')}"


def parse_config(name: str):
    m = re.fullmatch(r"A(\d+)_P(\d+)", name)
    if not m:
        raise argparse.ArgumentTypeError(f"Config must look like A12_P36, got {name!r}")
    return int(m.group(1)), int(m.group(2))


def run_single(config: str, arms, replicate: int, out: Path, seed: int):
    import pygame
    import src.run_experiment as rx
    from src import sim_clock

    class HeadlessRunner(rx.ExperimentRunnerGUI):
        def __init__(self):
            super().__init__()
            self.screen_size = (GRID_COLS * rx.GRID_SIZE, GRID_ROWS * rx.GRID_SIZE)
            self.screen = pygame.display.set_mode(self.screen_size)
            self._splash_shown = True

        # --- GUI screens become no-ops ---
        def show_splash(self, *args, **kwargs):
            pass

        def _show_strategy_countdown(self, *args, **kwargs):
            return True

        def _show_post_run_inspection(self, *args, **kwargs):
            pass

        def _show_experiment_complete_screen(self, *args, **kwargs):
            pass

        def _show_experiment_status_screen(self, *args, **kwargs):
            pass

        def _show_error_and_restart_setup(self, message=None):
            raise RuntimeError(f"[BATCH] setup error: {message}")

        def show_error(self, title, message):
            raise RuntimeError(f"[BATCH] {title}: {message}")

    rx.USE_SWEEP = False  # run() must not enter the interactive sweep path

    n_agents, n_parcels = parse_config(config)

    # Resume: a previous job for this (config, replicate, arms) may exist under `out`.
    existing = _find_replicate_experiment(out / config, replicate, arms)
    if existing is not None:
        data = json.loads(existing.read_text())
        pending = [i for i, st in enumerate(data["strategies"]) if st.get("status") not in TERMINAL]
        if not pending:
            print(f"[BATCH] {config} r{replicate} already complete -> {existing.parent}")
            return

    runner = HeadlessRunner()
    runner.seed = seed
    runner.fast_mode = True

    vault = rx.build_strategy_vault(runner.planners[0])
    selected = [s for s in vault if arm_id(s) in arms]
    missing = set(arms) - {arm_id(s) for s in selected}
    if missing:
        raise SystemExit(f"Unknown arms: {sorted(missing)}. Use --list-arms.")

    if existing is not None:
        # Same state the GUI's resume path sets (_load_or_create_experiment). With the
        # fixed timestep an interrupted arm restarted from its beginning gives the
        # same result it would have given uninterrupted.
        runner.experiment = data
        runner.experiment["current_index"] = pending[0]
        runner.experiment["status"] = "in_progress"
        runner.experiment_json_path = existing
        runner.experiment_dir = existing.parent
        runner.seed = data["seed"]
        runner.run_id = data["run_id"]
        runner._resumed = True
        print(f"[BATCH] {config} r{replicate} resuming at strategy index {pending[0]} -> {existing.parent}")
        _run_to_completion(runner, config, replicate)
        pygame.quit()
        return

    runner._create_experiment(
        n_agents, n_parcels, runner.planners[0], selected, 1,
        experiment_dir=out / config, instance_offset=replicate,
    )
    runner.experiment["provenance"] = {
        "runner": "src.run_batch",
        "fixed_timestep": sim_clock.FIXED_TIMESTEP,
        "sim_dt": sim_clock.SIM_DT,
        "grid": [GRID_COLS, GRID_ROWS],
        "replicate": replicate,
        "git_commit": _git_commit(),
        "src_sha256": _src_hash(),
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "sppdec_prompt_style_default": os.getenv("SPPDEC_PROMPT_STYLE", "paired_steps"),
        "controller_pick_policy": os.getenv("CONTROLLER_PICK_POLICY", "log"),
        "ollama_mode": os.getenv("OLLAMA_MODE"),
        # every order is declared to the fleet with id, position and weight before the first step
        "orders": "declared",
        "parcel_weight_range": os.getenv("PARCEL_WEIGHT_RANGE"),
        "energy_model": "realistic_v1",   # hover power = flight power; radio 8.3e-7 units to send, 1.9e-7 to receive (earlier batches: no hover charge, 0.05 + 0.005/cell per message)
    }
    runner._write_experiment_json()
    runner._resumed = True
    _run_to_completion(runner, config, replicate)
    pygame.quit()


# Strategy statuses that end a run; anything else (pending, or an arm interrupted
# mid-run) is re-run on resume.
TERMINAL = {"completed", "failed", "infeasible", "timeout", "aborted"}


class _QuietStream:
    """Pass only status lines to the job log; the per-step trace is dropped."""
    KEEP = ("[BATCH]", "[EXPERIMENT]", "[FINALIZE]", "Traceback", "Error", "[SLM FAILURE]")

    def __init__(self, stream):
        import threading
        self._stream = stream
        self._buf = ""
        self._lock = threading.Lock()  # agents' decisions print from worker threads

    def write(self, text):
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, _, self._buf = self._buf.partition("\n")
                if line.startswith(self.KEEP):
                    self._stream.write(line + "\n")
        return len(text)

    def flush(self):
        self._stream.flush()


def _rss_gb(pid: int) -> float:
    """Resident memory of a process in GB (0 if it has exited)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _find_replicate_experiment(config_dir: Path, replicate: int, arms):
    """Latest batch experiment under config_dir for this replicate and exactly these arms."""
    import json
    matches = []
    for path in config_dir.glob("experiment_*/experiment.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        prov = data.get("provenance") or {}
        families = {st.get("family") for st in data.get("strategies", [])}
        if prov.get("runner") == "src.run_batch" and prov.get("replicate") == replicate and families == set(arms):
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _run_to_completion(runner, config: str, replicate: int):
    t0 = time.time()
    # run() leaves the strategy loop when an arm aborts (e.g. the command centre
    # declares global infeasibility); the GUI resumed from current_index, so do the same.
    total = len(runner.experiment["strategies"])
    while int(runner.experiment.get("current_index", 0)) < total:
        before = int(runner.experiment.get("current_index", 0))
        runner._resumed = True
        runner.run()
        if int(runner.experiment.get("current_index", 0)) <= before:
            raise RuntimeError(f"[BATCH] no progress at strategy index {before}")
    print(f"[BATCH] {config} r{replicate} finished in {time.time() - t0:.0f}s -> {runner.experiment_dir}")


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "src"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _src_hash() -> str:
    """Hash of all Python sources under src/, to tie results to exact code when uncommitted."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(_SRC_DIR).rglob("*.py")):
        h.update(str(f.relative_to(_SRC_DIR)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--replicates", type=int, default=5,
                    help="independent task environments per config (different seeds)")
    ap.add_argument("--only-replicates", type=int, nargs="+", default=None,
                    help="run just these replicate indices (0-based), e.g. to fill gaps")
    ap.add_argument("--out", type=Path, default=Path("experiments/revision"))
    ap.add_argument("--jobs", type=int, default=1, help="configurations run in parallel")
    ap.add_argument("--seed", type=int, default=None,
                    help="experiment seed (default: derived from the config name, so reruns are reproducible)")
    ap.add_argument("--seeds-from", type=Path, default=None,
                    help="reuse each config's seed (and so parcel layout) from an existing sweep, "
                         "e.g. experiments/station_scaling")
    ap.add_argument("--mem-gb", type=float, default=8.0,
                    help="kill a job whose resident memory exceeds this (protects the machine)")
    ap.add_argument("--verbose", action="store_true",
                    help="keep the full per-step console trace in job logs (can reach GBs)")
    ap.add_argument("--list-arms", action="store_true")
    ap.add_argument("--_single", help=argparse.SUPPRESS)
    ap.add_argument("--_replicate", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list_arms:
        import src.run_experiment as rx
        for spec in rx.build_strategy_vault("local"):
            print(arm_id(spec))
        return

    if args._single:
        if not args.verbose:
            sys.stdout = _QuietStream(sys.stdout)
        run_single(args._single, set(args.arms), args._replicate, args.out, args.seed)
        return

    if not args.configs or not args.arms:
        ap.error("--configs and --arms are required")
    for c in args.configs:
        parse_config(c)

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)

    os.environ["DRONE_PROCESSES"] = "1"      # the only mode: one process per drone, each on its own core; jobs are admitted only while the drones of all running jobs fit the physical cores
    process_mode = True
    from src.system.topology import physical_core_groups
    cores_total = len(physical_core_groups())

    def demand(cfg):          # a run needs one core per drone and one for the world
        return parse_config(cfg)[0] + 1
    reps = args.only_replicates if args.only_replicates is not None else range(args.replicates)
    pending = [(cfg, r) for r in reps for cfg in args.configs]
    running = {}
    failures = []
    while pending or running:
        while pending and len(running) < args.jobs:
            if process_mode and running and sum(demand(c) for c, _ in running) + demand(pending[0][0]) > cores_total:
                break             # wait for cores: the drones of all running jobs must fit the physical cores
            cfg, r = pending.pop(0)
            if args.seed is not None:
                base = args.seed
            elif args.seeds_from is not None:
                base = _seed_from(args.seeds_from, cfg)
            else:
                base = _config_seed(cfg)
            cmd = [sys.executable, "-m", "src.run_batch", "--_single", cfg, "--_replicate", str(r),
                   "--seed", str(base + r * REPLICATE_STRIDE),
                   "--out", str(args.out), "--arms", *args.arms]
            if args.verbose:
                cmd.append("--verbose")
            log = open(log_dir / f"{cfg}_r{r}.log", "a")
            running[(cfg, r)] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"[BATCH] started {cfg} r{r}")
        for job, (proc, log) in list(running.items()):
            rss_gb = _rss_gb(proc.pid)
            if proc.poll() is None and rss_gb > args.mem_gb:
                proc.kill()
                proc.wait()
                log.write(f"\n[BATCH] killed: resident memory {rss_gb:.1f} GB > --mem-gb {args.mem_gb}\n")
                print(f"[BATCH] {job[0]} r{job[1]}: killed at {rss_gb:.1f} GB (memory guard)")
            if proc.poll() is not None:
                log.close()
                del running[job]
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode != 0:
                    failures.append(job)
                print(f"[BATCH] {job[0]} r{job[1]}: {status}")
        time.sleep(1)

    if failures:
        print(f"[BATCH] failures: {failures} (see {log_dir})")
        sys.exit(1)


def _seed_from(sweep_dir, config: str):
    if sweep_dir is None:
        return None
    files = sorted((sweep_dir / config).glob("experiment_*/experiment.json"))
    if not files:
        raise SystemExit(f"No experiment.json for {config} under {sweep_dir}")
    import json
    return int(json.loads(files[0].read_text())["seed"])


def _config_seed(config: str) -> int:
    a, p = parse_config(config)
    return 1_000_003 * a + p  # stable per config, identical for every arm and instance


if __name__ == "__main__":
    main()
