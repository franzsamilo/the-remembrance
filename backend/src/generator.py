"""
Discovery generator: retrieves graph context, filters validated triplets,
and synthesizes auditable narrative responses via the synthesis layer.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config, logger
from src.retriever import GraphRetriever
from src.synthesis import generate_narrative_response, generate_chunk_response
from src.chunk_retriever import retrieve_chunks


class DiscoveryGenerator:
    def __init__(self):
        self.retriever = GraphRetriever()

    async def generate_answer(self, query, explain: bool = False):
        """
        Generates an auditable answer using the Synthesis Layer.
        Returns filtered_triplets for UI transparency.
        """
        # 1. Retrieve Subgraph Context (Triples) + Discovery Leads
        _, triplets, leads = self.retriever.retrieve(query)
        validated_statuses = {"trained_experimental", "validated"}
        validated_triplets = [
            triplet
            for triplet in triplets
            if triplet.get("audit") is not None and triplet.get("audit_status") in validated_statuses
        ]
        # Track what was filtered out
        validated_ids = {id(t) for t in validated_triplets}
        filtered_triplets = [t for t in triplets if id(t) not in validated_ids]
        # Fallback: if no validated triplets (e.g. unaudited graph), use all with audit score >= threshold
        if not validated_triplets and triplets:
            min_score = Config.GROUNDING_MIN_SCORE
            validated_triplets = [
                t for t in triplets
                if t.get("audit") is not None and (t.get("audit") or 0) >= min_score
            ]
            validated_ids = {id(t) for t in validated_triplets}
            filtered_triplets = [t for t in triplets if id(t) not in validated_ids]

        # Grounding error: no validated triplets at all
        if not validated_triplets:
            lead_objects = self._normalize_leads(leads)
            return {
                "narrative_text": None,
                "grounding_error": True,
                "message": "No validated evidence found. The system refuses to generate an unsupported answer.",
                "triplets": [],
                "filtered_triplets": filtered_triplets,
                "leads": lead_objects,
                "context_summary": "",
                "suggested_actions": [],
            }

        context = "\n".join(
            f"({t['source']})-[{t['relation']}]->({t['target']})"
            for t in validated_triplets
        )

        lead_objects = self._normalize_leads(leads)

        # 2. Synthesis: Convert Triples + Leads to Auditable Narrative
        narrative_package = await generate_narrative_response(
            query,
            validated_triplets,
            lead_objects,
            include_explanations=explain,
        )

        if explain:
            triplet_explanations = narrative_package.get("triplet_explanations", [])
            explanation_map = {}
            for item in triplet_explanations:
                key = (item.get("source"), item.get("relation"), item.get("target"))
                explanation_map[key] = item.get("explanation")

            lead_explanations = narrative_package.get("lead_explanations", [])
            lead_explanation_map = {
                item.get("name"): item.get("explanation") for item in lead_explanations
            }

            for t in validated_triplets:
                key = (t.get("source"), t.get("relation"), t.get("target"))
                t["explanation"] = explanation_map.get(key)

            for lead in lead_objects:
                lead["explanation"] = lead_explanation_map.get(lead.get("name"))

        return {
            "narrative_text": narrative_package.get("narrative_text", ""),
            "triplets": validated_triplets,
            "filtered_triplets": filtered_triplets,
            "leads": lead_objects,
            "context_summary": context,
            "suggested_actions": narrative_package.get("suggested_actions", []),
        }

    def _normalize_leads(self, leads):
        """Convert raw lead strings into structured objects."""
        lead_objects = []
        for lead in leads:
            if not lead:
                continue
            if ":" in lead:
                name, desc = lead.split(":", 1)
                lead_objects.append({"name": name.strip(), "description": desc.strip()})
            else:
                lead_objects.append({"name": lead.strip(), "description": None})
        return lead_objects

    async def generate_answer_prompt_only(self, query: str):
        """
        Ablation: chunk-based RAG only. No graph retrieval, no GNN audit.
        Returns narrative with empty triplets/leads for comparison.
        """
        context_str, chunks = retrieve_chunks(query, top_k=5)
        narrative = await generate_chunk_response(query, context_str)
        # Convert chunks to minimal triplet-like structure for UI compatibility
        pseudo_triplets = [
            {
                "source": c.get("source", "Chunk"),
                "relation": "EXCERPT",
                "target": (c.get("text", "")[:80] + "…") if len(c.get("text", "")) > 80 else c.get("text", ""),
                "audit": None,
                "description": None,
                "source_docs": [c.get("source", "")],
                "target_docs": [],
                "cross_document": False,
            }
            for c in chunks
        ]
        return {
            "narrative_text": narrative,
            "triplets": pseudo_triplets,
            "leads": [],
            "context_summary": context_str[:500] + "…" if len(context_str) > 500 else context_str,
            "suggested_actions": [],
        }


if __name__ == "__main__":
    import asyncio
    import json
    generator = DiscoveryGenerator()
    result = asyncio.run(generator.generate_answer("Who are the researchers?", explain=True))
    logger.info("%s", json.dumps(result, indent=2))
