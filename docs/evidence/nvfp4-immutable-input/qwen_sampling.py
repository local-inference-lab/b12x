"""Sampling contract for Qwen3.8 serving qualification requests."""

import argparse
import math


def temperature(value):
    result = float(value)
    if not math.isfinite(result) or result < 0.7:
        raise argparse.ArgumentTypeError("Qwen qualification requires temperature >= 0.7")
    return result


DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
