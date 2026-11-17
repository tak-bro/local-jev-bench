"""Measure every engine on every set, one engine at a time, through scripts/measure.sh: the full pass behind the
README's Results, with its protocol in code (reps from bench/run.py's PROTOCOL_REPS, a 300 s call timeout, each Ollaya
engine on the daemon it needs).

    uv run python bench/matrix.py --dry-run          # what is left to measure in this pass, measuring nothing
    uv run python bench/matrix.py                    # every engine; resumes where an earlier run of this pass stopped
    uv run python bench/matrix.py kev-4b clef        # only these engines
    uv run python bench/matrix.py --new-pass         # start a new pass: every engine x set is measured again

A pass starts at the time in logs/matrix.since, written by its first run and replaced by --new-pass. An engine x set
whose log is complete, started after that time and was measured on the set as it is now is done and skipped, so a pass
that stops (a crash, a reboot) resumes instead of starting over. Progress goes to logs/matrix.status, each engine's
measure.sh output to logs/matrix-<engine>.out.

It refuses to start unless the Mac is on AC power, memory pressure is normal, no engine server answers yet and neither
Ollaya daemon has a model loaded: every engine is measured alone on an otherwise idle machine. It keeps the Mac awake
with `caffeinate` while it runs, and checks every set's report with `bench/score.py --check` at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests

import charts
import run
import score

ROOT = Path(__file__).resolve().parents[1]
# The two slowest engines go last, so one stuck run delays the fewest others: 15 h (Decider-4B) and 46 h (nimble-v3,
# 30 h of it typed-decisions) of the 2026-11-12..16 pass's 107 h, both on CPU.
LAST = ("decider-4b", "nimble-v3")
ENGINES = [e for e in run.ENGINES if e not in LAST] + list(LAST)
TIMEOUT = 300  # seconds per call; CPU engines take up to a minute on typed-decisions' long states
# Servers measure.sh starts besides the engines' own: CLM's embedding server and AnyJev's generate server.
EXTRA_URLS = ("http://127.0.0.1:8090", os.environ.get("ANYJEV_LLM_URL", "http://127.0.0.1:8092"))


def set_paths() -> list[Path]:
    """The README's ten sets, in its column order: the hand-written ones in bench/, the built ones in bench/data/."""
    paths = [ROOT / "bench" / f"{stem}.jsonl" if (ROOT / "bench" / f"{stem}.jsonl").exists()
             else ROOT / "bench" / "data" / f"{stem}.jsonl" for stem, _ in charts.SETS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing sets: {', '.join(str(p.relative_to(ROOT)) for p in missing)}; "
                         f"build them with `uv run python bench/make_sets.py <name>`")
    return paths


def done(log: Path, set_path: Path, since: datetime) -> bool:
    """Whether `log` is a complete run that started at or after `since`, on the set as it is now."""
    if not log.exists():
        return False
    lines = [line for line in log.read_text().splitlines() if line.strip()]
    try:
        header, last = json.loads(lines[0])["header"], json.loads(lines[-1])
    except (IndexError, KeyError, ValueError):
        return False
    return ("footer" in last and datetime.fromisoformat(header["started"]) >= since
            and header["set_sha256"] == score.sha256(set_path))


def plan(engines: list[str], sets: list[Path], runs: Path, since: datetime) -> list[tuple[str, list[Path]]]:
    """(engine, the sets it still needs in this pass) for every engine with any left."""
    todo = [(e, [s for s in sets if not done(runs / s.stem / f"{e}.jsonl", s, since)]) for e in engines]
    return [(e, left) for e, left in todo if left]


def answers(url: str) -> bool:
    """Whether anything accepts a connection on the URL's host and port."""
    parts = urlsplit(url)
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((parts.hostname or "127.0.0.1", parts.port or 80)) == 0


def loaded_models(url: str) -> list[str]:
    """The models an Ollaya daemon holds in memory; none when no daemon answers there."""
    if not answers(url):
        return []
    try:
        return [m["name"] for m in run.SESSION.get(f"{url}/api/ps", timeout=10).json()["models"]]
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        return [f"unreadable /api/ps ({e})"]  # a daemon that cannot say what it holds may hold something


def preflight(allow_battery: bool = False) -> list[str]:
    """What keeps this machine from measuring alone and awake; empty when it can."""
    problems = []
    if not allow_battery and not (source := run.power()).startswith("AC Power"):
        problems.append(f"not on AC power: {source}")
    if (pressure := run.memory_pressure()) != "normal":
        problems.append(f"memory pressure is {pressure}")
    daemons = set(run.OLLAYA_URLS.values())
    busy = sorted({cfg["url"] for cfg in run.ENGINES.values() if cfg["url"] not in daemons} | set(EXTRA_URLS))
    problems += [f"{url} already answers; stop that server first" for url in busy if answers(url)]
    problems += [f"Ollaya at {url} holds {', '.join(models)}; `ollaya stop` them first"
                 for url in sorted(daemons) if (models := loaded_models(url))]
    return problems


def pass_start(path: Path, new_pass: bool) -> datetime:
    if new_pass or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.now() + "\n")
    return datetime.fromisoformat(path.read_text().strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", help=f"default: all {len(ENGINES)}, the slowest last")
    ap.add_argument("--new-pass", action="store_true", help="start a new pass: measure every engine x set again")
    ap.add_argument("--dry-run", action="store_true", help="print what is left in this pass and measure nothing")
    ap.add_argument("--allow-battery", action="store_true", help="measure on battery power (latency will suffer)")
    ap.add_argument("--runs", type=Path, default=ROOT / "bench" / "runs")
    ap.add_argument("--logs", type=Path, default=ROOT / "logs")
    args = ap.parse_args()
    unknown = [e for e in args.engines if e not in run.ENGINES]
    if unknown:
        ap.error(f"unknown engines: {', '.join(unknown)}")
    engines = [e for e in ENGINES if e in args.engines] if args.engines else ENGINES

    since_file = args.logs / "matrix.since"
    if args.dry_run:
        since = datetime.fromisoformat(since_file.read_text().strip()) if since_file.exists() and not args.new_pass \
            else datetime.fromisoformat(run.now())
    else:
        if problems := preflight(args.allow_battery):
            raise SystemExit("not starting:\n" + "\n".join(f"- {p}" for p in problems))
        since = pass_start(since_file, args.new_pass)
    sets = set_paths()
    todo = plan(engines, sets, args.runs, since)
    for engine, left in todo:
        print(f"{engine}: {' '.join(s.stem for s in left)}")
    if args.dry_run or not todo:
        print(f"pass since {since.isoformat()}: {len(todo)} engines left")
        return 0

    subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])  # exits with this process
    status = args.logs / "matrix.status"
    env = {**os.environ, "MEASURE_BENCH": os.environ.get("MEASURE_BENCH",
                                                         f"uv run python bench/run.py --timeout {TIMEOUT}")}

    def note(text: str) -> None:
        with status.open("a") as f:
            f.write(f"{run.now()} {text}\n")

    note(f"pass since {since.isoformat()}: {len(todo)} engines")
    failed = []
    for engine, left in todo:
        held = [m for url in sorted(set(run.OLLAYA_URLS.values())) for m in loaded_models(url)]
        note(f"START {engine} {' '.join(s.stem for s in left)} (memory {run.memory_pressure()}, "
             f"Ollaya holds {', '.join(held) or 'nothing'}; {run.power()})")
        t0 = time.monotonic()
        with (args.logs / f"matrix-{engine}.out").open("w") as out:
            rc = subprocess.run([str(ROOT / "scripts" / "measure.sh"), engine,
                                 *(str(s.relative_to(ROOT)) for s in left)],
                                cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        note(f"END   {engine} rc={rc} {time.monotonic() - t0:.0f}s")
        if rc:  # a set failed, or some calls did (JevK5 refuses every 20-option set, so it always lands here)
            failed.append(engine)
    stale = [s.stem for s in sets
             if subprocess.run([sys.executable, str(ROOT / "bench" / "score.py"), str(args.runs / s.stem), "--check"],
                               cwd=ROOT, capture_output=True).returncode]
    left = [e for e, _ in plan(engines, sets, args.runs, since)]
    summary = (f"measure.sh exit 1: {', '.join(failed) or 'none'}; stale reports: {', '.join(stale) or 'none'}; "
               f"engines with sets left in this pass: {', '.join(left) or 'none'}")
    note(f"pass done; {summary}")
    print(summary)
    return 1 if stale or left else 0


if __name__ == "__main__":
    sys.exit(main())
