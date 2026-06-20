from __future__ import annotations

"""
Generative evaluation: Grounding and Faithfulness via LLM-as-judge.
Runs test queries through the generator and scores outputs against retrieved triplets.
"""
import asyncio
import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import Config, logger
from src.generator import DiscoveryGenerator
from src.helpers import utc_now_iso as _utc_now_iso


def _load_queries() -> list[str]:
    """Load test queries from file or env."""
    queries_env = os.getenv("EVALUATION_QUERIES")
    if queries_env:
        try:
            return json.loads(queries_env)
        except json.JSONDecodeError:
            logger.warning("EVALUATION_QUERIES invalid JSON, using file")
    path = Config.EVALUATION_QUERIES_FILE
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(q) for q in data if q]
        if isinstance(data, dict) and "queries" in data:
            return [str(q) for q in data["queries"] if q]
    return [
        "What are the key findings?",
        "Who are the main researchers?",
        "What methods were used?",
        "What are the main results?",
        "What datasets or concepts are discussed?",
    ]


def _parse_json_from_response(text: str) -> dict | None:
    """Extract JSON object from LLM response."""
    text = text.strip()
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass  # Fallback to next parse strategy
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass  # Fallback to next parse strategy
    return None


async def _score_grounding(narrative: str, triplets: list, llm) -> float | None:
    """Score 1-5 per claim, return average normalized to 0-1."""
    if not narrative or not triplets:
        return None
    triplet_str = "\n".join(
        f"({t.get('source')})-[{t.get('relation')}]->({t.get('target')})"
        for t in triplets[:30]
    )
    prompt = f"""You are a strict grounding evaluator. Given a narrative and knowledge graph triplets, identify each distinct factual claim in the narrative and rate how well it is supported by the triplets.

SCORING RUBRIC:
- 5: Claim directly maps to one or more triplets (entity names and relationship match exactly)
- 4: Claim is strongly implied by combining 2-3 triplets in a logical chain
- 3: Claim references entities from the triplets but states a relationship not explicitly present
- 2: Claim is loosely related to triplet content but adds unsupported interpretation
- 1: Claim has no traceable connection to any triplet

A "claim" is any statement asserting a fact, relationship, or conclusion — typically one per sentence. Ignore stylistic or transitional text.

NARRATIVE:
{narrative}

TRIPLETS:
{triplet_str}

Return ONLY valid JSON: {{"scores": [5,4,3,...], "average": X}} where average is the mean of all scores (0-5 scale)."""
    try:
        resp = await llm.ainvoke(prompt)
        data = _parse_json_from_response(resp.content)
        if data and "average" in data:
            avg = float(data["average"])
            return min(1.0, max(0.0, avg / 5.0))
        if data and "scores" in data:
            scores = [float(s) for s in data["scores"] if isinstance(s, (int, float))]
            if scores:
                return min(1.0, max(0.0, sum(scores) / len(scores) / 5.0))
    except Exception as e:
        logger.warning("Grounding score parse failed: %s", e)
    return None


async def _score_faithfulness(narrative: str, triplets: list, llm) -> float | None:
    """Ratio of claims supported by context."""
    if not narrative or not triplets:
        return None
    triplet_str = "\n".join(
        f"({t.get('source')})-[{t.get('relation')}]->({t.get('target')})"
        for t in triplets[:30]
    )
    prompt = f"""You are a strict faithfulness evaluator. Extract every distinct factual claim from the narrative and determine whether each is supported by the provided triplets.

RULES:
- A "claim" is any statement asserting a fact, relationship, attribution, or conclusion. Extract one claim per distinct assertion (typically one per sentence, but compound sentences may have multiple).
- A claim is "supported" (true) ONLY if the entities AND relationship can be traced to one or more triplets, either directly or through a short logical chain (2-3 triplets).
- A claim is "unsupported" (false) if it introduces entities, relationships, or conclusions not present in the triplets — even if plausible.
- Ignore transitional phrases, questions, and stylistic language — only evaluate factual assertions.

NARRATIVE:
{narrative}

TRIPLETS:
{triplet_str}

Return ONLY valid JSON: {{"claims": [{{"text": "...", "supported": true/false}}], "ratio": X}} where ratio = supported_count / total_count (0.0-1.0)."""
    try:
        resp = await llm.ainvoke(prompt)
        data = _parse_json_from_response(resp.content)
        if data and "ratio" in data:
            return min(1.0, max(0.0, float(data["ratio"])))
        if data and "claims" in data:
            claims = data["claims"]
            if claims:
                supported = sum(1 for c in claims if c.get("supported") is True)
                return supported / len(claims)
    except Exception as e:
        logger.warning("Faithfulness score parse failed: %s", e)
    return None


async def run_grounding_evaluation(
    mode: str = "full_stack",
    grounding_threshold: float | None = None,
    persist_to_ablation: bool = True,
) -> dict:
    """
    Run grounding and faithfulness evaluation on test queries.
    Returns and persists results to evaluation_results.json.

    Args:
        mode: 'full_stack' | 'graph_no_gnn' | 'prompt_only'.
              'full_stack'   — GNN-validated retrieval + synthesis (default).
              'graph_no_gnn' — graph retrieval WITHOUT plausibility filter.
                               Isolates the GNN integrity layer's contribution.
              'prompt_only'  — chunk RAG, no graph (weakest baseline).
        persist_to_ablation: Write results into evaluation_results.json's
              top-level + ablation[mode] section. Set False when called from a
              threshold sweep so transient thresholds don't clobber the primary
              ablation numbers.
    """
    queries = _load_queries()
    if not queries:
        logger.warning("No evaluation queries found")
        return {"grounding_score": None, "faithfulness_score": None, "sample_count": 0}

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0,
    )
    generator = DiscoveryGenerator()

    # Bound concurrent Gemini calls so we don't trip per-minute rate limits.
    sem = asyncio.Semaphore(Config.EVALUATION_MAX_CONCURRENCY)

    async def _evaluate_one(q: str) -> dict | None:
        async with sem:
            try:
                if mode == "prompt_only":
                    result = await generator.generate_answer_prompt_only(q)
                elif mode == "graph_no_gnn":
                    result = await generator.generate_answer_graph_no_gnn(q, explain=True)
                else:
                    result = await generator.generate_answer(
                        q, explain=True, grounding_threshold=grounding_threshold
                    )
                narrative = result.get("narrative_text", "")
                triplets = result.get("triplets", [])
                if not narrative:
                    return None
                # Grounding + faithfulness judges are independent — fan them out.
                g, f = await asyncio.gather(
                    _score_grounding(narrative, triplets, llm),
                    _score_faithfulness(narrative, triplets, llm),
                )
                return {
                    "query": q,
                    "grounding_score": g,
                    "faithfulness_score": f,
                }
            except Exception as e:
                logger.warning("Evaluation failed for query %s: %s", q[:50], e)
                return None

    per_query_raw = await asyncio.gather(*[_evaluate_one(q) for q in queries])
    per_query_results = [r for r in per_query_raw if r is not None]
    grounding_scores = [
        r["grounding_score"] for r in per_query_results if r["grounding_score"] is not None
    ]
    faithfulness_scores = [
        r["faithfulness_score"] for r in per_query_results if r["faithfulness_score"] is not None
    ]

    grounding_score = sum(grounding_scores) / len(grounding_scores) if grounding_scores else None
    faithfulness_score = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

    output = {
        "grounding_score": grounding_score,
        "faithfulness_score": faithfulness_score,
        "completed_at": _utc_now_iso(),
        "sample_count": len(grounding_scores) or len(faithfulness_scores),
        "mode": mode,
        "grounding_threshold": grounding_threshold or Config.GROUNDING_MIN_SCORE,
        "per_query": per_query_results,
    }

    # Persist per-mode results for ablation comparison
    path = Config.EVALUATION_RESULTS_PATH
    if path and persist_to_ablation:
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        if "ablation" not in existing:
            existing["ablation"] = {}
        existing["ablation"][mode] = {
            "grounding_score": grounding_score,
            "faithfulness_score": faithfulness_score,
            "completed_at": _utc_now_iso(),
            "sample_count": output["sample_count"],
            "per_query": per_query_results,
        }
        # Top-level reflects the PRIMARY pipeline (full_stack). Ablation modes
        # (prompt_only) are recorded only under ablation[mode] so the dashboard's
        # headline KPI never shows the ablation baseline.
        if mode == "full_stack":
            existing["grounding_score"] = grounding_score
            existing["faithfulness_score"] = faithfulness_score
            existing["completed_at"] = _utc_now_iso()
            existing["sample_count"] = output["sample_count"]
            existing["per_query"] = per_query_results
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Evaluation results written to %s (mode=%s)", path, mode)

    return output


def persist_gnn_metrics(training_history: dict) -> None:
    """Merge GNN training metrics into evaluation_results.json."""
    path = Config.EVALUATION_RESULTS_PATH
    if not path:
        return
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["gnn"] = {
        "auc_roc": training_history.get("final_auc_roc"),
        "mrr": training_history.get("final_mrr"),
        "best_epoch": training_history.get("best_epoch"),
        "early_stop_epoch": training_history.get("early_stop_epoch"),
        "completed_at": _utc_now_iso(),
    }
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("GNN metrics written to %s", path)


async def run_threshold_sweep(thresholds: list[float] | None = None) -> dict:
    """Run evaluation across multiple plausibility thresholds for sensitivity analysis."""
    if thresholds is None:
        # BPR-trained model: scores span [0, 1]. Paper's τ=0.95 is meaningful.
        # Sweep spans both BCE and BPR usable bands.
        thresholds = [0.30, 0.50, 0.85, 0.95]
    results = {}
    for tau in thresholds:
        logger.info("Threshold sweep: running evaluation at tau=%.2f", tau)
        result = await run_grounding_evaluation(
            mode="full_stack",
            grounding_threshold=tau,
            persist_to_ablation=False,
        )
        results[str(tau)] = {
            "grounding_score": result.get("grounding_score"),
            "faithfulness_score": result.get("faithfulness_score"),
            "sample_count": result.get("sample_count"),
            "per_query": result.get("per_query", []),
        }
    path = Config.EVALUATION_RESULTS_PATH
    if path:
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing["threshold_sweep"] = {
            "thresholds": thresholds,
            "results": results,
            "completed_at": _utc_now_iso(),
        }
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Threshold sweep results written to %s", path)
    return results


if __name__ == "__main__":
    asyncio.run(run_grounding_evaluation())
