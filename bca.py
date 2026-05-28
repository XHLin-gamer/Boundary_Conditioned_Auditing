from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

import numpy as np

from models import BackboneLLM, Qwen3_5
from prompts import ANCHOR_COMPARISON_QUERY_PROMPT, ANCHOR_PLAN_PROMPT
from rag import ReActRAG
from watermarking import KGW


try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None


@dataclass(frozen=True)
class PairStrategy:
    name: str
    needs_centered_similarity: bool = False
    needs_raw_similarity: bool = False
    target_similarity: float | None = None


PAIR_STRATEGIES = {
    "random": PairStrategy("random"),
    "nearest": PairStrategy("nearest"),
    "centered_orthogonal": PairStrategy(
        "centered_orthogonal",
        needs_centered_similarity=True,
        target_similarity=0.0,
    ),
    "hybrid_centered_orthogonal": PairStrategy(
        "hybrid_centered_orthogonal",
        needs_centered_similarity=True,
        needs_raw_similarity=True,
    ),
}


class BCA:
    def __init__(
        self,
        data_dir: str,
        db_dir: str,
        pair_count: int = 130,
        top_k: int = 5,
        pair_strategy: str = "hybrid_centered_orthogonal",
        pair_alpha: float = 1.0,
        pair_seed: int = 42,
        embedding_device: str = "cuda:1",
        detector_device: str = "cuda",
        backbone_model: str = "deepseek-v4-flash",
        backbone_provider: str = "auto",
        backbone_temperature: float = 0,
        backbone_reasoning_effort: str | None = None,
        rag_enable_thinking: bool = False,
    ):
        if pair_strategy not in PAIR_STRATEGIES:
            options = ", ".join(sorted(PAIR_STRATEGIES))
            raise ValueError(f"pair_strategy must be one of: {options}")

        self.data_dir = data_dir
        self.db_dir = db_dir
        self.pair_count = pair_count
        self.top_k = top_k
        self.pair_strategy = pair_strategy
        self.pair_alpha = pair_alpha
        self.pair_seed = pair_seed
        self.backbone_model = backbone_model
        self.backbone_provider = BackboneLLM.infer_provider(backbone_model, backbone_provider)
        self.backbone_temperature = backbone_temperature
        self.backbone_reasoning_effort = backbone_reasoning_effort
        self.rag_enable_thinking = rag_enable_thinking
        self.pairing_backend = "lexical"

        self.model = BackboneLLM(
            model=backbone_model,
            provider=backbone_provider,
            temperature=backbone_temperature,
            reasoning_effort=backbone_reasoning_effort,
        )
        self.detector = KGW(Qwen3_5("Qwen/Qwen3.5-2B", device=detector_device))
        self.embedding_model = (
            SentenceTransformer("Qwen/Qwen3-Embedding-4B", device=embedding_device)
            if SentenceTransformer is not None
            else None
        )
        self.rag = None

    def build_rag(self, rebuild_index: bool = False):
        self.rag = ReActRAG(
            data_dir=self.data_dir,
            db_dir=self.db_dir,
            top_k=self.top_k,
            chat_model=self.backbone_model,
            chat_provider=self.backbone_provider,
            chat_temperature=self.backbone_temperature,
            chat_enable_thinking=self.rag_enable_thinking,
            chat_reasoning_effort=self.backbone_reasoning_effort,
        )
        if rebuild_index:
            self.rag.build_index()
        else:
            self.rag.load_index()
        return self.rag

    def load_entries(self, label: str):
        entries = []
        for file_name in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, file_name)
            if os.path.isdir(path) or label not in file_name:
                continue
            with open(path, encoding="utf-8") as handle:
                entries.append({"file": file_name, "text": handle.read()})
        return entries

    def find_pairs(self, entries):
        if len(entries) < 2:
            return []
        strategy = PAIR_STRATEGIES[self.pair_strategy]
        if strategy.name == "random":
            return self.random_pairs(entries)
        candidates = self.pair_candidates(entries, strategy)
        return self.select_disjoint_pairs(entries, self.rank_candidates(candidates, strategy), strategy)

    def random_pairs(self, entries):
        indices = list(range(len(entries)))
        random.Random(self.pair_seed).shuffle(indices)
        pairs = []
        for offset in range(0, len(indices) - 1, 2):
            i, j = indices[offset], indices[offset + 1]
            similarity = self.lexical_similarity(entries[i], entries[j])
            pairs.append(
                {
                    "documents": [entries[i], entries[j]],
                    "similarity": similarity,
                    "ranking_score": similarity,
                    "centered": False,
                    "pair_seed": self.pair_seed,
                }
            )
            if len(pairs) >= self.pair_count:
                break
        self.pairing_backend = "random"
        return pairs

    def pair_candidates(self, entries, strategy: PairStrategy):
        embeddings = self.embed_texts([entry["text"] for entry in entries])
        if embeddings is None:
            similarities = self.lexical_similarity_matrix(entries)
            candidates = self.all_pair_candidates(similarities)
            if strategy.needs_raw_similarity:
                for candidate in candidates:
                    candidate["raw_similarity"] = candidate["similarity"]
                    candidate["centered_similarity"] = 0.0
            return candidates

        raw_similarities = embeddings @ embeddings.T
        centered_similarities = None
        if strategy.needs_centered_similarity:
            centered = self.center_embeddings(embeddings)
            centered_similarities = centered @ centered.T
            self.pairing_backend = f"centered_{self.pairing_backend}"
        if strategy.needs_raw_similarity:
            return self.hybrid_pair_candidates(raw_similarities, centered_similarities)
        similarities = centered_similarities if centered_similarities is not None else raw_similarities
        return self.all_pair_candidates(similarities)

    def embed_texts(self, texts):
        if self.embedding_model is None:
            return None
        self.pairing_backend = "sentence_transformer"
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    @staticmethod
    def all_pair_candidates(similarities):
        candidates = []
        for i in range(similarities.shape[0]):
            for j in range(i + 1, similarities.shape[0]):
                similarity = float(similarities[i, j])
                candidates.append(
                    {
                        "i": i,
                        "j": j,
                        "similarity": similarity,
                        "ranking_score": similarity,
                    }
                )
        return candidates

    @staticmethod
    def hybrid_pair_candidates(raw_similarities, centered_similarities):
        candidates = []
        for i in range(raw_similarities.shape[0]):
            for j in range(i + 1, raw_similarities.shape[0]):
                candidates.append(
                    {
                        "i": i,
                        "j": j,
                        "similarity": float(raw_similarities[i, j]),
                        "raw_similarity": float(raw_similarities[i, j]),
                        "centered_similarity": float(centered_similarities[i, j]),
                    }
                )
        return candidates

    def rank_candidates(self, candidates, strategy: PairStrategy):
        if strategy.name == "nearest":
            return sorted(candidates, key=lambda item: item["similarity"], reverse=True)
        if strategy.name == "centered_orthogonal":
            target = strategy.target_similarity
            for candidate in candidates:
                candidate["ranking_score"] = -abs(candidate["similarity"] - target)
            return sorted(candidates, key=lambda item: item["ranking_score"], reverse=True)
        if strategy.name == "hybrid_centered_orthogonal":
            for candidate in candidates:
                candidate["pair_alpha"] = self.pair_alpha
                candidate["ranking_score"] = (
                    candidate["raw_similarity"]
                    - self.pair_alpha * abs(candidate["centered_similarity"])
                )
            return sorted(candidates, key=lambda item: item["ranking_score"], reverse=True)
        raise ValueError(f"Unknown pair strategy: {strategy.name}")

    def select_disjoint_pairs(self, entries, candidates, strategy: PairStrategy):
        pairs = []
        used = set()
        for candidate in candidates:
            i, j = candidate["i"], candidate["j"]
            if i in used or j in used:
                continue
            pair = {
                "documents": [entries[i], entries[j]],
                "similarity": candidate["similarity"],
                "ranking_score": candidate["ranking_score"],
                "centered": strategy.needs_centered_similarity,
            }
            if strategy.target_similarity is not None:
                pair["target_similarity"] = strategy.target_similarity
            for key in ["raw_similarity", "centered_similarity", "pair_alpha"]:
                if key in candidate:
                    pair[key] = candidate[key]
            pairs.append(pair)
            used.update([i, j])
            if len(pairs) >= self.pair_count:
                break
        return pairs

    def generate_query_bundle(self, documents):
        plan, raw_plan = self.generate_anchor_plan(documents)
        query = self.generate_anchor_query(plan, raw_plan)
        retrieval_queries = self.build_retrieval_queries(plan, raw_plan)
        return {
            "query_plan": plan,
            "query_plan_raw": raw_plan,
            "query": query,
            "retrieval_query": self.format_retrieval_queries(retrieval_queries),
            "retrieval_queries": retrieval_queries,
        }

    def generate_anchor_plan(self, documents):
        user_prompt = "\n\n".join(
            f"Document {index + 1}:\n{document['text']}"
            for index, document in enumerate(documents)
        )
        raw_plan = self.model.generate(
            system_prompt=ANCHOR_PLAN_PROMPT,
            user_prompt=f"{user_prompt}\n\nJSON:",
            model=self.backbone_model,
            enable_thinking=False,
        ).strip()
        return self.parse_json_object(raw_plan), raw_plan

    def generate_anchor_query(self, plan, raw_plan):
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2) if isinstance(plan, dict) else raw_plan
        query = self.model.generate(
            system_prompt=ANCHOR_COMPARISON_QUERY_PROMPT,
            user_prompt=f"Retrieval plan:\n{plan_text}\n\nQuestion:",
            model=self.backbone_model,
            enable_thinking=False,
        ).strip()
        return query if query.endswith("?") else f"{query.rstrip('.')}?"

    def build_retrieval_queries(self, plan, raw_plan):
        if not isinstance(plan, dict):
            return [raw_plan]
        doc1_anchors = self.plan_anchors(plan, "doc1_anchors", document_number=1)
        doc2_anchors = self.plan_anchors(plan, "doc2_anchors", document_number=2)
        axes = self.as_list(plan.get("axes") or plan.get("comparison_axes") or [])
        shared_topic = plan.get("shared_topic", "")
        common = "\n".join(
            [
                f"Shared topic: {shared_topic}",
                f"Comparison axes: {', '.join(str(axis) for axis in axes)}",
            ]
        )
        queries = []
        if doc1_anchors:
            queries.append(f"{common}\nDocument 1 anchors: {', '.join(doc1_anchors)}")
        if doc2_anchors:
            queries.append(f"{common}\nDocument 2 anchors: {', '.join(doc2_anchors)}")
        return queries or [raw_plan]

    @staticmethod
    def center_embeddings(embeddings):
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.maximum(norms, 1e-12)

    @staticmethod
    def tokenize(text: str):
        return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower()))

    def lexical_similarity_matrix(self, entries):
        token_sets = [self.tokenize(entry["text"]) for entry in entries]
        similarities = np.zeros((len(entries), len(entries)), dtype=float)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                union = token_sets[i] | token_sets[j]
                similarity = len(token_sets[i] & token_sets[j]) / len(union) if union else 0.0
                similarities[i, j] = similarity
                similarities[j, i] = similarity
        return similarities

    def lexical_similarity(self, entry_a, entry_b):
        tokens_a = self.tokenize(entry_a["text"])
        tokens_b = self.tokenize(entry_b["text"])
        union = tokens_a | tokens_b
        return float(len(tokens_a & tokens_b) / len(union)) if union else 0.0

    @staticmethod
    def format_retrieval_queries(queries):
        return "\n\n".join(
            f"[Retrieval query {index + 1}]\n{query}"
            for index, query in enumerate(queries)
        )

    @staticmethod
    def plan_anchors(plan, key, document_number):
        if key in plan and isinstance(plan[key], list):
            return [str(anchor) for anchor in plan[key]]
        for item in plan.get("document_anchors", []):
            if isinstance(item, dict) and item.get("document") == document_number:
                return [str(anchor) for anchor in item.get("anchors", [])]
        return []

    @staticmethod
    def as_list(value):
        if isinstance(value, list):
            return value
        return [value] if value else []

    @staticmethod
    def parse_json_object(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {"raw": text}

    @staticmethod
    def is_detected(result):
        return bool(result.get("is_watermarked", False)) if isinstance(result, dict) else bool(result)
