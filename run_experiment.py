from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime as dt
from pathlib import Path

from tqdm import tqdm

from bca import BCA
from models import BackboneLLM, Qwen3_5
from prompts import WARD_QUERY_PROMPT
from rag import ReActRAG
from watermarking import KGW


DEFAULT_TOKEN_BUDGET = 16384
DEFAULT_WORK_DIR = "Data/scale_4096_aligned_query_sets_rerun_a1"


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")


def detection_tokens(record):
    result = record.get("detection_result") or record.get("detected_as_watermarked") or {}
    return result.get("token_count", 0) if isinstance(result, dict) else 0


def safe_detect(detector, answer: str):
    try:
        return detector.detect(answer)
    except ValueError as exc:
        return {
            "is_watermarked": False,
            "score": None,
            "green_token_ratios": None,
            "token_count": 0,
            "detector_error": str(exc),
        }


def source_id(path, suffix):
    return Path(path).name.removesuffix(f"_{suffix}.txt")


def aligned_paths(work_dir):
    root = Path(work_dir)
    return {
        "root": root,
        "watermarked": root / "watermarked",
        "clean": root / "clean",
        "watermarked_db": root / "watermarked_chroma_db",
        "clean_db": root / "clean_chroma_db",
    }


def output_path(args, method):
    return str(Path(args.out_dir) / f"scaled_{method}_{args.token_budget}_{args.run_id}.json")


def aligned_dataset_ready(work_dir):
    paths = aligned_paths(work_dir)
    return any(paths["watermarked"].glob("*_wm.txt")) and any(paths["clean"].glob("*_clean.txt"))


def aligned_indexes_ready(work_dir):
    paths = aligned_paths(work_dir)
    return paths["watermarked_db"].exists() and paths["clean_db"].exists()


def sample_aligned_file_pairs(wm_dir, clean_dir, count, seed):
    wm_files = {source_id(path, "wm"): path for path in sorted(Path(wm_dir).glob("*_wm.txt"))}
    clean_files = {source_id(path, "clean"): path for path in sorted(Path(clean_dir).glob("*_clean.txt"))}
    common_ids = sorted(set(wm_files) & set(clean_files))
    random.Random(seed).shuffle(common_ids)
    return [(wm_files[item_id], clean_files[item_id]) for item_id in common_ids[:count]]


def build_aligned_dataset(args):
    paths = aligned_paths(args.work_dir)
    if paths["root"].exists():
        if aligned_dataset_ready(args.work_dir) and not args.force_data:
            print(f"Using existing aligned dataset: {paths['root']}")
            return
        if args.force_data:
            shutil.rmtree(paths["root"])

    paths["watermarked"].mkdir(parents=True, exist_ok=True)
    paths["clean"].mkdir(parents=True, exist_ok=True)
    pairs = sample_aligned_file_pairs(args.wm_dir, args.clean_dir, args.count, args.seed)
    if not pairs:
        raise FileNotFoundError("No aligned watermarked/clean files were found.")

    for wm_file, clean_file in pairs:
        shutil.copy(wm_file, paths["watermarked"] / wm_file.name)
        shutil.copy(clean_file, paths["clean"] / clean_file.name)
    args.rebuild_index = True
    print(f"Built aligned dataset with {len(pairs)} file pairs: {paths['root']}")


def prepare_rag(rag: ReActRAG, db_dir, rebuild_index: bool):
    if rebuild_index and Path(db_dir).exists():
        shutil.rmtree(db_dir)
    if rebuild_index:
        rag.build_index()
    else:
        rag.load_index()


def group_totals(records, pair_records: bool = False):
    flag_name = "from_watermarked_pair" if pair_records else "from_watermarked"
    totals = {"watermarked": 0, "clean": 0, "all": 0}
    for record in records:
        group = "watermarked" if record.get(flag_name) else "clean"
        tokens = detection_tokens(record)
        totals[group] += tokens
        totals["all"] += tokens
    return totals


def serialize_trace(trace):
    return [asdict(step) for step in trace]


def rotated_index(index: int, length: int, rotate: bool):
    if length == 0:
        return 0, None
    if rotate:
        return index // length, index % length
    if index >= length:
        return 0, None
    return 0, index


class WardExperiment:
    def __init__(self, args):
        self.args = args
        self.paths = aligned_paths(args.work_dir)
        self.provider = BackboneLLM.infer_provider(args.backbone_model, args.backbone_provider)
        self.query_model = BackboneLLM(
            model=args.backbone_model,
            provider=args.backbone_provider,
            temperature=args.backbone_temperature,
            reasoning_effort=args.openai_reasoning_effort,
        )
        self.detector = KGW(Qwen3_5("Qwen/Qwen3.5-2B", device=args.detector_device))
        self.rags = {
            "watermarked": self.build_rag("watermarked"),
            "clean": self.build_rag("clean"),
        }

    def build_rag(self, group):
        return ReActRAG(
            data_dir=str(self.paths[group]),
            db_dir=str(self.paths[f"{group}_db"]),
            top_k=self.args.top_k,
            chat_model=self.args.backbone_model,
            chat_provider=self.provider,
            chat_temperature=self.args.backbone_temperature,
            chat_enable_thinking=self.args.rag_enable_thinking,
            chat_reasoning_effort=self.args.openai_reasoning_effort,
        )

    def prepare(self):
        for group, rag in self.rags.items():
            prepare_rag(rag, self.paths[f"{group}_db"], self.args.rebuild_index)

    def generate_query(self, text: str) -> str:
        return self.query_model.generate(
            system_prompt=WARD_QUERY_PROMPT,
            user_prompt=f"Document:\n{text}\n\nQuestion:",
            model=self.args.backbone_model,
            enable_thinking=False,
        ).strip()

    def run(self, out_path):
        self.prepare()
        query_files = sorted(self.paths["watermarked"].glob("*_wm.txt"))
        if not query_files:
            raise FileNotFoundError(f"No *_wm.txt files in {self.paths['watermarked']}.")

        records = load_existing(out_path)
        totals = group_totals(records)
        while totals["watermarked"] < self.args.token_budget or totals["clean"] < self.args.token_budget:
            index = max(
                sum(1 for record in records if record.get("from_watermarked")),
                sum(1 for record in records if not record.get("from_watermarked")),
            )
            rotation, file_index = rotated_index(index, len(query_files), self.args.rotate_inputs)
            if file_index is None:
                break

            query_file = query_files[file_index]
            query = self.generate_query(query_file.read_text(encoding="utf-8"))
            for group, from_wm in [("watermarked", True), ("clean", False)]:
                if totals[group] >= self.args.token_budget:
                    continue
                answer, trace = self.rags[group].answer(query, return_trace=True)
                db_file = query_file.name if from_wm else query_file.name.replace("_wm.txt", "_clean.txt")
                records.append(
                    {
                        "file": db_file,
                        "query_source_file": query_file.name,
                        "database_group": group,
                        "query_source": "watermarked",
                        "from_watermarked": from_wm,
                        "source_index": file_index,
                        "rotation": rotation,
                        "query": query,
                        "answer": answer,
                        "detection_result": safe_detect(self.detector, answer),
                        "trace": serialize_trace(trace),
                    }
                )
                totals = group_totals(records)
                save_json(out_path, self.payload(records, "running"))
                tqdm.write(f"Ward totals: {totals}")

        save_json(out_path, self.payload(records, "completed"))
        print(f"Saved Ward results to {out_path}")
        return out_path

    def payload(self, records, status):
        return {
            "method": "scaled_ward_react_aligned_control",
            "status": status,
            "params": common_params(self.args, self.paths),
            "token_totals": group_totals(records),
            "results": records,
        }


def run_bca(args, out_path):
    paths = aligned_paths(args.work_dir)
    runner = BCA(
        data_dir=str(paths["watermarked"]),
        db_dir=str(paths["watermarked_db"]),
        pair_count=args.pair_count,
        top_k=args.top_k,
        pair_strategy=args.pair_strategy,
        pair_alpha=args.pair_alpha,
        pair_seed=args.seed,
        embedding_device=args.embedding_device,
        detector_device=args.detector_device,
        backbone_model=args.backbone_model,
        backbone_provider=args.backbone_provider,
        backbone_temperature=args.backbone_temperature,
        backbone_reasoning_effort=args.openai_reasoning_effort,
        rag_enable_thinking=args.rag_enable_thinking,
    )
    if args.rebuild_index and paths["watermarked_db"].exists():
        shutil.rmtree(paths["watermarked_db"])
    runner.build_rag(rebuild_index=args.rebuild_index)
    clean_rag = ReActRAG(
        data_dir=str(paths["clean"]),
        db_dir=str(paths["clean_db"]),
        top_k=args.top_k,
        chat_model=args.backbone_model,
        chat_provider=runner.backbone_provider,
        chat_temperature=args.backbone_temperature,
        chat_enable_thinking=args.rag_enable_thinking,
        chat_reasoning_effort=args.openai_reasoning_effort,
    )
    prepare_rag(clean_rag, paths["clean_db"], args.rebuild_index)

    pairs = runner.find_pairs(runner.load_entries("wm"))
    if not pairs:
        raise RuntimeError("No BCA pairs were generated.")

    query_cache = {}
    records = load_existing(out_path)
    totals = group_totals(records, pair_records=True)
    while totals["watermarked"] < args.token_budget or totals["clean"] < args.token_budget:
        progressed = False
        for from_wm, group in [(True, "watermarked"), (False, "clean")]:
            if totals[group] >= args.token_budget:
                continue
            index = sum(1 for record in records if record.get("from_watermarked_pair") == from_wm)
            rotation, pair_index = rotated_index(index, len(pairs), args.rotate_inputs)
            if pair_index is None:
                continue
            pair = pairs[pair_index]
            cache_key = query_cache_key(pair)
            if cache_key not in query_cache:
                query_cache[cache_key] = runner.generate_query_bundle(pair["documents"])
            rag = runner.rag if from_wm else clean_rag
            records.append(answer_pair(runner, rag, pair, query_cache[cache_key], from_wm, group, pair_index, rotation))
            totals = group_totals(records, pair_records=True)
            save_json(out_path, bca_payload(args, paths, runner, records, "running"))
            tqdm.write(f"BCA totals: {totals}")
            progressed = True
        if not progressed:
            break

    save_json(out_path, bca_payload(args, paths, runner, records, "completed"))
    print(f"Saved BCA results to {out_path}")
    return out_path


def query_cache_key(pair):
    documents = tuple(
        (document.get("file"), hashlib.sha256(document.get("text", "").encode("utf-8")).hexdigest()[:16])
        for document in pair["documents"]
    )
    return documents


def answer_pair(runner, rag, pair, query_bundle, from_wm, group, pair_index, rotation):
    answer, trace = rag.answer(
        query_bundle["query"],
        return_trace=True,
        retrieval_query=query_bundle.get("retrieval_queries") or query_bundle.get("retrieval_query"),
    )
    query_source_files = [document["file"] for document in pair["documents"]]
    database_files = query_source_files if from_wm else clean_pair_files(query_source_files)
    detection = safe_detect(runner.detector, answer)
    record = {
        "from_watermarked_pair": from_wm,
        "database_group": group,
        "query_source": "watermarked",
        "query_source_files": query_source_files,
        "files": database_files,
        "database_files": database_files,
        "pair_index": pair_index,
        "rotation": rotation,
        "query_prompt_style": "anchor_comparison",
        "query": query_bundle["query"],
        "retrieval_query": query_bundle.get("retrieval_query"),
        "retrieval_queries": query_bundle.get("retrieval_queries"),
        "query_plan": query_bundle.get("query_plan"),
        "query_plan_raw": query_bundle.get("query_plan_raw"),
        "answer": answer,
        "detection_result": detection,
        "detected_as_watermarked": runner.is_detected(detection),
        "trace": serialize_trace(trace),
    }
    record.update(pair_metadata(pair))
    return record


def clean_pair_files(files):
    return [file_name.replace("_wm.txt", "_clean.txt") for file_name in files]


def pair_metadata(pair):
    keys = [
        "similarity",
        "ranking_score",
        "target_similarity",
        "centered",
        "raw_similarity",
        "centered_similarity",
        "pair_alpha",
    ]
    return {key: pair[key] for key in keys if key in pair}


def bca_payload(args, paths, runner, records, status):
    return {
        "method": "anchor_comparison_pairs_aligned_control",
        "status": status,
        "params": {
            **common_params(args, paths),
            "pair_count": args.pair_count,
            "pair_strategy": args.pair_strategy,
            "pair_alpha": args.pair_alpha,
            "pair_seed": args.seed,
            "pairing_backend": runner.pairing_backend,
            "backbone_provider": runner.backbone_provider,
            "backbone_model": runner.backbone_model,
            "query_prompt_style": "anchor_comparison",
            "soft_refusal_suffix": "",
            "soft_refusal_enabled": False,
        },
        "summary": summarize_bca(records),
        "results": records,
    }


def summarize_bca(records):
    summary = {}
    for flag, key in [(True, "watermarked_pairs"), (False, "clean_pairs")]:
        group = [record for record in records if record.get("from_watermarked_pair") == flag]
        scores = [
            record["detection_result"].get("score")
            for record in group
            if isinstance(record.get("detection_result"), dict)
            and record["detection_result"].get("score") is not None
        ]
        summary[key] = {
            "count": len(group),
            "tokens": sum(detection_tokens(record) for record in group),
            "detected": sum(record.get("detected_as_watermarked", False) for record in group),
            "avg_score": sum(scores) / len(scores) if scores else None,
        }
    return summary


def load_existing(path):
    if not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    records = payload.get("results", [])
    if records:
        print(f"Resuming from {path}: {len(records)} records.")
    return records


def common_params(args, paths):
    return {
        "token_budget": args.token_budget,
        "top_k": args.top_k,
        "work_dir": args.work_dir,
        "count": args.count,
        "seed": args.seed,
        "rotate_inputs": args.rotate_inputs,
        "aligned_control": True,
        "query_source": "watermarked",
        "backbone_provider": args.backbone_provider,
        "backbone_model": args.backbone_model,
        "backbone_temperature": args.backbone_temperature,
        "openai_reasoning_effort": args.openai_reasoning_effort,
        "rag_enable_thinking": args.rag_enable_thinking,
        "watermarked_data_dir": str(paths["watermarked"]),
        "clean_data_dir": str(paths["clean"]),
        "watermarked_db_dir": str(paths["watermarked_db"]),
        "clean_db_dir": str(paths["clean_db"]),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run the main aligned Ward/BCA RAG watermark experiment.")
    parser.add_argument("--method", choices=["ward", "bca", "both"], default="both")
    parser.add_argument("--token-budget", "--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--wm-dir", default="Data/watermarked_texts")
    parser.add_argument("--clean-dir", default="Data/original_texts")
    parser.add_argument("--out-dir", default="Data0518")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--detector-device", default="cuda")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--backbone-provider", choices=["auto", "deepseek", "gemini", "openai"], default="auto")
    parser.add_argument("--backbone-model", default="deepseek-v4-flash")
    parser.add_argument("--backbone-temperature", type=float, default=0.0)
    parser.add_argument("--rag-enable-thinking", action="store_true")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument("--query-style", choices=["anchor_comparison"], default="anchor_comparison")
    parser.add_argument(
        "--pair-strategy",
        choices=["random", "nearest", "centered_orthogonal", "hybrid_centered_orthogonal"],
        default="hybrid_centered_orthogonal",
    )
    parser.add_argument("--pair-alpha", type=float, default=1.0)
    parser.add_argument("--pair-count", type=int, default=130)
    parser.add_argument("--rotate-inputs", action="store_true")
    return parser


def contrastive_method_name(args):
    alpha = f"{args.pair_alpha:g}".replace(".", "p")
    model = args.backbone_model.replace("/", "-").replace(".", "p")
    return f"contrastive_{args.pair_strategy}_a{alpha}_{args.query_style}_{model}_aligned"


def main():
    args = build_parser().parse_args()
    if args.token_budget < 1:
        raise ValueError("--token-budget must be positive.")
    if args.count < 2:
        raise ValueError("--count must be at least 2.")
    args.run_id = args.run_id or dt.now().strftime("%m_%d-%H_%M")

    if args.force_data or not aligned_dataset_ready(args.work_dir):
        build_aligned_dataset(args)
    if args.build_only:
        return
    if not aligned_indexes_ready(args.work_dir):
        args.rebuild_index = True

    if args.method in {"ward", "both"}:
        WardExperiment(args).run(output_path(args, "ward_aligned"))
        args.rebuild_index = False
    if args.method in {"bca", "both"}:
        run_bca(args, output_path(args, contrastive_method_name(args)))


if __name__ == "__main__":
    main()
