"""p50/p95 single-utterance latency. Usage: python -m router.bench_latency [--runs 50] [--device cpu]"""
import argparse
import statistics
import warnings

from .cli import DEFAULT_MODEL, HERE
from .core import Router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.model, str(HERE / "labels.json"), device=args.device)
    for _ in range(3):  # warm-up
        router.predict("check my oxygen level")
    ms = sorted(router.predict("check my oxygen level")["latency_ms"] for _ in range(args.runs))
    print(f"device={router.device} runs={args.runs} p50={statistics.median(ms):.1f} ms "
          f"p95={ms[int(0.95 * (len(ms) - 1))]:.1f} ms min={ms[0]:.1f} ms")


if __name__ == "__main__":
    main()
