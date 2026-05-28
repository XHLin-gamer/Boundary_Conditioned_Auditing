from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np

from models import Qwen3_5
from watermarking import KGW


def is_from_watermarked(record):
    if "from_watermarked_pair" in record:
        return bool(record["from_watermarked_pair"])
    return bool(record.get("from_watermarked", False))


def load_answers(path, group):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = []
    for record in payload.get("results", []):
        answer = record.get("answer")
        if not answer:
            continue
        from_wm = is_from_watermarked(record)
        if group == "watermarked" and from_wm:
            answers.append(answer.strip())
        elif group == "clean" and not from_wm:
            answers.append(answer.strip())
    return answers


def z_score(green_count, token_count, gamma):
    if token_count <= 0:
        return 0.0
    return (green_count - gamma * token_count) / sqrt(gamma * (1 - gamma) * token_count)


def p_value(z):
    return 0.5 * erfc(z / sqrt(2.0))


def pooled_stats(watermark, path, group, budget, gamma):
    text = "\n\n".join(load_answers(path, group))
    flags = np.asarray(watermark.detect(text, return_green_flags=True), dtype=int)
    scored = flags[flags >= 0]
    used = scored[: min(budget, len(scored))]
    token_count = int(len(used))
    green_count = int((used == 1).sum())
    z = float(z_score(green_count, token_count, gamma))
    return {
        "available_scored_tokens": int(len(scored)),
        "token_count": token_count,
        "green_count": green_count,
        "green_ratio": green_count / token_count if token_count else None,
        "z_score": z,
        "p_value": float(p_value(z)),
    }


def record_label(path):
    stem = Path(path).stem
    if "ward" in stem:
        return "Ward"
    if "contrastive" in stem or "BCA" in stem:
        return "BCA"
    return stem


def build_parser():
    parser = argparse.ArgumentParser(description="Compute pooled KGW detection statistics.")
    parser.add_argument("records", nargs="+")
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", default="")
    return parser


def main():
    args = build_parser().parse_args()
    watermark = KGW(Qwen3_5("Qwen/Qwen3.5-2B", device=args.device))
    summary = {
        "budget": args.budget,
        "gamma": args.gamma,
        "records": [],
    }
    print("record\twm_z\tclean_z\tgap\twm_green\tclean_green")
    for path in args.records:
        wm = pooled_stats(watermark, path, "watermarked", args.budget, args.gamma)
        clean = pooled_stats(watermark, path, "clean", args.budget, args.gamma)
        gap = wm["z_score"] - clean["z_score"]
        summary["records"].append(
            {
                "label": record_label(path),
                "path": path,
                "watermarked": wm,
                "clean": clean,
                "gap_z": gap,
            }
        )
        print(
            f"{record_label(path)}\t{wm['z_score']:.4f}\t{clean['z_score']:.4f}\t"
            f"{gap:.4f}\t{wm['green_ratio']:.4f}\t{clean['green_ratio']:.4f}"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

