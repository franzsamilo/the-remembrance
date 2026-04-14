"""
Synthesis layer: converts graph triplets and leads into analytical briefings
using Gemini. Returns narrative text plus per-triplet and per-lead explanations.
"""
from typing import List, Dict, Any, Optional
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config, logger


async def generate_chunk_response(query: str, context_chunks: str) -> str:
    """
    Ablation: Answer from raw document chunks only. No graph, no triplets.
    Used for prompt_only mode to compare against full stack.
    """
    if not context_chunks or not context_chunks.strip():
        return "I have no document context to answer from. Upload PDFs and try again."

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0,  # Deterministic: reproducible synthesis for evaluation
    )
    prompt = f"""Answer the user's question using ONLY the following document excerpts. Do not use outside knowledge. If the excerpts do not contain enough information, say so clearly. Be direct and cite which document (the [filename] prefix) supports your answer when relevant.

DOCUMENT EXCERPTS:
{context_chunks}

USER QUESTION: {query}

Answer:"""
    try:
        response = await llm.ainvoke(prompt)
        return response.content.strip() or "No response generated."
    except Exception as e:
        logger.error("Chunk synthesis failed: %s", e)
        return f"Synthesis failed: {e}"


async def generate_narrative_response(
    query: str,
    triplets: List[Dict[str, Any]],
    leads: List[Dict[str, Optional[str]]] = [],
    include_explanations: bool = True,
):
    """
    Step 5: Synthesis Layer
    Converts graph triplets and community leads into an 'Analytical Briefing'.
    """
    if not triplets:
        return {
            "narrative_text": "I do not have record of that in my validated knowledge base. No supporting evidence was found in the graph.",
            "triplet_explanations": [],
            "lead_explanations": [],
        }

    # Format triplets for the prompt
    triplet_strings = []
    for t in triplets:
        triplet_strings.append(f"({t['source']})-[{t['relation']}]->({t['target']})")
    
    context_text = "\n".join(triplet_strings)
    
    # Format leads
    leads_text = (
        "\n".join(
            [
                f"- {lead.get('name', 'Unknown')}"
                + (f": {lead.get('description')}" if lead.get("description") else "")
                for lead in leads
            ]
        )
        if leads
        else "No specific community leads identified."
    )

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0  # Deterministic: reproducible synthesis for evaluation
    )

    persona_line = (
        f'You are the Lead {Config.SYNTHESIS_PERSONA} for "{Config.SYNTHESIS_FRAMEWORK_NAME}" framework.'
        if Config.SYNTHESIS_FRAMEWORK_NAME
        else f"You are the Lead {Config.SYNTHESIS_PERSONA} providing analytical briefings."
    )

    system_instruction = f"""
    PERSONA: {persona_line}
    OBJECTIVE: Provide a precise, evidence-grounded Analytical Briefing. You are a meticulous analyst: every claim must be directly traceable to the provided graph evidence.

    GROUNDING RULES:
    - ONLY state facts that are explicitly present in the provided Graph Triples.
    - Do NOT infer, extrapolate, or synthesize connections that are not directly represented in the triples.
    - If the evidence is sparse or covers only part of the question, state what IS known and stop. Do not fill gaps with plausible-sounding claims.
    - Each sentence in your narrative should map to one or more specific triples.

    SOURCE USAGE:
    - Use the provided Graph Triples as your sole factual evidence.
    - Use the provided "Discovery Leads" for broader context and investigative paths, but clearly distinguish them from triple-backed facts.
    - DO NOT mention 'triples', 'nodes', 'edges', 'Leiden', or 'clusters'.
    
    RESPONSE STRUCTURE (Fluid & Spaced):
    
    1. **The Direct Answer**: Start immediately with the answer. Just give the fluid narrative without preamble. Tell the story of the evidence.
    
    2. **Separator**: accurate use of a markdown horizontal rule (---) to visually separate the answer from the deeper analysis.
    
    3. **Contextual Analysis**: 
       - If you found relevant Discovery Leads, discuss them here.
       - Explain exactly *why* these leads matter based on the evidence.
       - Use a subtle header like "### **Deeper Context & Leads**"
    
    TONE: Investigative, expert, direct, unhedged, and exceptionally thorough. Use straight english and wide evidencing. Use **bolding** for key concepts.
    
    OUTPUT FORMAT:
    - Return ONLY valid JSON.
    - Use keys: narrative_text, triplet_explanations, lead_explanations, suggested_actions
    - triplet_explanations is a list of objects: {{source, relation, target, explanation}}
    - lead_explanations is a list of objects: {{name, explanation}}
    - suggested_actions is a list of 2-4 bold, actionable next steps (strings) for the professional. CRITICAL: Each action MUST be directly tied to the USER QUERY — guide the user on what to do next given what they asked. Ask yourself: "Given that the user asked this, what should they do next with this evidence?" Use imperative, concrete language. Examples: "File a motion to compel discovery on [entity from their question]", "Request production of [Document X] — it supports your position on [topic they asked about]", "Depose [witness] regarding the [relationship/claim they inquired about]", "Draft a brief citing [source] to answer the [specific question they posed]". Be specific to both the evidence AND the original query. Avoid generic suggestions; every action should clearly connect back to what the user asked.
    
    CRITICAL EXPLANATION RULES for triplet_explanations and lead_explanations:
    - NEVER use generic relevance phrases (e.g. "this node is here because it's relevant to this", "this connection shows relevance").
    - You MUST provide hard evidencing, detailing exactly WHAT the connection is and WHY it exists based on the exact relationship data provided.
    - Treat each explanation as a detailed piece of hard evidence in an investigation brief.
    """

    system_instruction_simple = f"""
    PERSONA: {persona_line}
    OBJECTIVE: Provide a precise, evidence-grounded Analytical Briefing. You are a meticulous analyst: every claim must be directly traceable to the provided graph evidence.

    GROUNDING RULES:
    - ONLY state facts that are explicitly present in the provided Graph Triples.
    - Do NOT infer, extrapolate, or synthesize connections that are not directly represented in the triples.
    - If the evidence is sparse or covers only part of the question, state what IS known and stop. Do not fill gaps with plausible-sounding claims.
    - Each sentence in your narrative should map to one or more specific triples.

    SOURCE USAGE:
    - Use the provided Graph Triples as your sole factual evidence.
    - Use the provided "Discovery Leads" for broader context and investigative paths, but clearly distinguish them from triple-backed facts.
    - DO NOT mention 'triples', 'nodes', 'edges', 'Leiden', or 'clusters'.
    
    RESPONSE STRUCTURE (Fluid & Spaced):
    
    1. **The Direct Answer**: Start immediately with the answer. Just give the fluid narrative without preamble. Tell the story of the evidence.
    
    2. **Separator**: accurate use of a markdown horizontal rule (---) to visually separate the answer from the deeper analysis.
    
    3. **Contextual Analysis**: 
       - If you found relevant Discovery Leads, discuss them here.
       - Explain exactly *why* these leads matter based on the evidence.
       - Use a subtle header like "### **Deeper Context & Leads**"
    
    TONE: Investigative, expert, direct, unhedged, and exceptionally thorough. Use straight english and wide evidencing. Use **bolding** for key concepts.
    """

    prompt = f"""
    SYSTEM: {system_instruction}
    
    USER QUERY: {query}
    
    VALIDATED GRAPH TRIPLETS (Evidence):
    {context_text}
    
    DISCOVERY LEADS (Community Context):
    {leads_text}
    
    ANALYTICAL BRIEFING:
    """

    prompt_simple = f"""
    SYSTEM: {system_instruction_simple}
    
    USER QUERY: {query}
    
    VALIDATED GRAPH TRIPLETS (Evidence):
    {context_text}
    
    DISCOVERY LEADS (Community Context):
    {leads_text}
    
    ANALYTICAL BRIEFING:
    """

    try:
        logger.info(f"Synthesizing narrative for query: {query}")
        response = await llm.ainvoke(prompt if include_explanations else prompt_simple)
        content = response.content.strip()
        if not include_explanations:
            return {
                "narrative_text": content,
                "triplet_explanations": [],
                "lead_explanations": [],
                "suggested_actions": [],
            }
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    payload = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    return {
                        "narrative_text": content,
                        "triplet_explanations": [],
                        "lead_explanations": [],
                        "suggested_actions": [],
                    }
            else:
                return {
                    "narrative_text": content,
                    "triplet_explanations": [],
                    "lead_explanations": [],
                    "suggested_actions": [],
                }

        if not isinstance(payload, dict):
            return {
                "narrative_text": content,
                "triplet_explanations": [],
                "lead_explanations": [],
                "suggested_actions": [],
            }

        expl = payload.get("triplet_explanations")
        lead_expl = payload.get("lead_explanations")
        actions = payload.get("suggested_actions")
        return {
            "narrative_text": str(payload.get("narrative_text") or ""),
            "triplet_explanations": expl if isinstance(expl, list) else [],
            "lead_explanations": lead_expl if isinstance(lead_expl, list) else [],
            "suggested_actions": actions if isinstance(actions, list) else [],
        }
    except Exception as e:
        logger.error(f"Synthesis Error: {str(e)}")
        return {
            "narrative_text": "Synthesis layer failed to perform discovery analysis.",
            "triplet_explanations": [],
            "lead_explanations": [],
            "suggested_actions": [],
        }
