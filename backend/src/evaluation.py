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
        for t in triplets[:15]
    )
    prompt = f"""You are an evaluator. Given this narrative and these knowledge graph triplets, rate each substantive claim in the narrative from 1-5 (5=fully traceable to the triplets, 1=not traceable).

NARRATIVE:
{narrative}

TRIPLETS:
{triplet_str}

Return ONLY valid JSON: {{"scores": [1,5,4,...], "average": X}} where average is the mean of scores. Use 0-5 scale for average."""
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
        for t in triplets[:15]
    )
    prompt = f"""You are an evaluator. List each factual claim in the narrative. For each, say if it is supported by the triplets (yes/no).

NARRATIVE:
{narrative}

TRIPLETS:
{triplet_str}

Return ONLY valid JSON: {{"claims": [{{"text": "...", "supported": true/false}}], "ratio": X}} where ratio = supported_count / total_count (0-1)."""
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


async def run_grounding_evaluation(mode: str = "full_stack") -> dict:
    """
    Run grounding and faithfulness evaluation on test queries.
    Returns and persists results to evaluation_results.json.

    Args:
        mode: 'full_stack' | 'prompt_only' (graph-only mode not yet implemented).
              'full_stack' uses GNN-validated retrieval + synthesis (default).
              'prompt_only' bypasses the graph entirely (ablation baseline).
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

    grounding_scores = []
    faithfulness_scores = []

    for q in queries:
        try:
            if mode == "prompt_only":
                result = await generator.generate_answer_prompt_only(q)
            else:
                result = await generator.generate_answer(q, explain=True)
            narrative = result.get("narrative_text", "")
            triplets = result.get("triplets", [])
            if not narrative:
                continue
            g = await _score_grounding(narrative, triplets, llm)
            f = await _score_faithfulness(narrative, triplets, llm)
            if g is not None:
                grounding_scores.append(g)
            if f is not None:
                faithfulness_scores.append(f)
        except Exception as e:
            logger.warning("Evaluation failed for query %s: %s", q[:50], e)

    grounding_score = sum(grounding_scores) / len(grounding_scores) if grounding_scores else None
    faithfulness_score = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

    output = {
        "grounding_score": grounding_score,
        "faithfulness_score": faithfulness_score,
        "completed_at": _utc_now_iso(),
        "sample_count": len(grounding_scores) or len(faithfulness_scores),
        "mode": mode,
    }

    # Persist per-mode results for ablation comparison
    path = Config.EVALUATION_RESULTS_PATH
    if path:
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
        }
        # Keep top-level scores as the latest run
        existing["grounding_score"] = grounding_score
        existing["faithfulness_score"] = faithfulness_score
        existing["completed_at"] = _utc_now_iso()
        existing["sample_count"] = output["sample_count"]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Evaluation results written to %s (mode=%s)", path, mode)

    return output


if __name__ == "__main__":
    asyncio.run(run_grounding_evaluation())
