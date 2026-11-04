"""
Run every module's correctness check (gradient checks, equivalence checks, ...)
in learning order and print a summary. This is what CI runs on every push.

A module's checks are its `gradient_check.py` and every `*_check.py`.
Each check exits non-zero on failure.

Run:
    python run_checks.py                 all modules
    python run_checks.py rag agents      just these
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MODULES = ["mlp", "cnn", "rnn", "lstm", "attention", "transformer",
           "tokenizer", "llama", "distributed", "finetuning",
           "evals", "rag", "agents", "serving"]


def find_checks(module):
    folder = os.path.join(HERE, module)
    return sorted(glob.glob(os.path.join(folder, "*_check.py")),
                  key=lambda p: (os.path.basename(p) != "gradient_check.py", p))


def main():
    wanted = sys.argv[1:] or MODULES
    unknown = [m for m in wanted if m not in MODULES]
    if unknown:
        sys.exit(f"unknown module(s): {', '.join(unknown)}; choose from {', '.join(MODULES)}")
    results = []
    for module in wanted:
        checks = find_checks(module)
        if not checks:
            results.append((module, "-", "MISSING", 0.0))
            print(f"{module:12s} {'-':34s} MISSING", flush=True)
            continue
        for check in checks:
            start = time.time()
            proc = subprocess.run([sys.executable, check], cwd=HERE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            took = time.time() - start
            status = "PASSED" if proc.returncode == 0 else "FAILED"
            if proc.returncode != 0:
                print(proc.stdout)
            name = os.path.relpath(check, HERE)
            results.append((module, name, status, took))
            print(f"{module:12s} {name:34s} {status}  ({took:.1f} s)", flush=True)

    failed = [r for r in results if r[2] != "PASSED"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
