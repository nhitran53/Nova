"""Placeholder for ONNX export. Export is not implemented.
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export router to ONNX INT8 (stub).")
    ap.add_argument("--model", required=True, help="fine-tuned model dir")
    ap.add_argument("--out", required=True, help="output dir for model.int8.onnx")
    ap.parse_args(argv)
    print("export_onnx_stub: not implemented")
    return 1


if __name__ == "__main__":
    sys.exit(main())
