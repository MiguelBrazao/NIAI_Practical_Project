import argparse
import os
import subprocess
import sys
from pathlib import Path

# This script runs mario_ga_search_mlp.py (default) or mario_random_search_mlp.py
# for multiple seeds, optionally continuing on error.
# Usage:
# python run_experiments.py 42 123 999
# TASK=move_forward python run_experiments.py 42 123 999 2024 0
# METHOD=random TASK=hunter python run_experiments.py 42 123 999 2024 0
# TASK=move_forward python run_experiments.py 42 123 999 2024 0 --continue-on-error

def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Run mario_ga_search_mlp.py for multiple seeds. "
			"If TASK=move_forward, runs move_forward; otherwise runs hunter."
			"If METHOD=random, runs mario_random_search_mlp.py instead of the GA version. "
		)
	)
	parser.add_argument(
		"seeds",
		nargs="+",
		type=int,
		help="One or more integer seeds, e.g. 42 123 999",
	)
	parser.add_argument(
		"--continue-on-error",
		action="store_true",
		help="Continue remaining seeds even if one run fails.",
	)
	return parser.parse_args()


def resolve_task_mode():
	task = os.environ.get("TASK", "hunter").lower()
	if task == "move_forward":
		return "move_forward"
	return "hunter"


def resolve_method():
	method = os.environ.get("METHOD", "ga").lower()
	if method == "random":
		return "random"
	return "ga"


def main():
	args = parse_args()

	script_dir = Path(__file__).resolve().parent
	method = resolve_method()
	script_name = "mario_random_search_mlp.py" if method == "random" else "mario_ga_search_mlp.py"
	target_script = script_dir / script_name
	if not target_script.exists():
		raise FileNotFoundError(f"Cannot find target script: {target_script}")

	mode = resolve_task_mode()
	print(f"[run_experiments] Method: {method}")
	print(f"[run_experiments] Mode: {mode}")
	print(f"[run_experiments] Seeds: {args.seeds}")

	failures = []
	for i, seed in enumerate(args.seeds, start=1):
		print(f"\n[run_experiments] ({i}/{len(args.seeds)}) Running seed {seed}")
		env = os.environ.copy()
		env["TASK"] = mode

		result = subprocess.run(
			[sys.executable, str(target_script), str(seed)],
			cwd=str(script_dir),
			env=env,
		)

		if result.returncode != 0:
			failures.append((seed, result.returncode))
			print(
				f"[run_experiments] Seed {seed} failed with code {result.returncode}"
			)
			if not args.continue_on_error:
				break

	if failures:
		print("\n[run_experiments] Completed with failures:")
		for seed, code in failures:
			print(f"  - seed {seed}: exit code {code}")
		sys.exit(1)

	print("\n[run_experiments] All experiments finished successfully.")


if __name__ == "__main__":
	main()
