"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import GraphInfoCard, { NodeInfo, EdgeInfo } from "@/components/GraphInfoCard";
import type { Triplet, Lead } from "@/lib/types";

interface GraphNode {
  id: string;
  name: string;
  type: "query" | "entity" | "lead";
  x: number;
  y: number;
  vx: number;
  vy: number;
  description?: string;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  audit: number;
  description?: string;
  explanation?: string | null;
  cross_document?: boolean;
}

interface KnowledgeGraphProps {
  triplets: Triplet[];
  leads?: Lead[];
  query: string;
  showInlineCard?: boolean;
  onSelectNode?: (nodeInfo: NodeInfo | null) => void;
  onSelectEdge?: (edgeInfo: EdgeInfo | null) => void;
  /** When true, use larger min dimensions for better readability */
  large?: boolean;
}

// Deterministic seeded RNG for consistent jitter across refreshes
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// Simple force-directed layout with deterministic seeding
function useForceLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number
) {
  const nodesRef = useRef<GraphNode[]>([]);
  const [positions, setPositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const frameRef = useRef<number>(0);
  const iterRef = useRef(0);

  useEffect(() => {
    if (nodes.length === 0) return;

    // Init node positions deterministically
    const cx = width / 2;
    const cy = height / 2;
    // Seed based on node count + names for consistency
    const seed = nodes.reduce((s, n) => s + n.id.length * 7, nodes.length * 31);
    const rng = seededRandom(seed);

    nodesRef.current = nodes.map((n, i) => {
      if (n.type === "query") {
        return { ...n, x: cx, y: cy, vx: 0, vy: 0 };
      }
      // Deterministic layout: nodes ordered by appearance in evidence flow
      const nonQueryCount = nodes.length - 1;
      const angle = nonQueryCount > 0 ? ((i - 1) / nonQueryCount) * Math.PI * 2 - Math.PI / 2 : 0;
      const radius = n.type === "lead" ? 200 : 150;
      // Small deterministic jitter to break symmetry consistently
      const jitterX = (rng() - 0.5) * 10;
      const jitterY = (rng() - 0.5) * 10;
      return {
        ...n,
        x: cx + Math.cos(angle) * radius + jitterX,
        y: cy + Math.sin(angle) * radius + jitterY,
        vx: 0,
        vy: 0,
      };
    });

    iterRef.current = 0;

    const simulate = () => {
      const ns = nodesRef.current;
      const dampening = 0.85;
      const repulsionStrength = 2000;
      const attractionStrength = 0.015;
      const centerPull = 0.002;
      const idealDist = 120;

      // Repulsion: all nodes repel each other
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          const dx = ns[i].x - ns[j].x;
          const dy = ns[i].y - ns[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsionStrength / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          ns[i].vx += fx;
          ns[i].vy += fy;
          ns[j].vx -= fx;
          ns[j].vy -= fy;
        }
      }

      // Attraction along edges
      for (const edge of edges) {
        const src = ns.find((n) => n.id === edge.source);
        const tgt = ns.find((n) => n.id === edge.target);
        if (!src || !tgt) continue;
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - idealDist) * attractionStrength;
        src.vx += (dx / dist) * force;
        src.vy += (dy / dist) * force;
        tgt.vx -= (dx / dist) * force;
        tgt.vy -= (dy / dist) * force;
      }

      // Center gravity
      for (const n of ns) {
        n.vx += (cx - n.x) * centerPull;
        n.vy += (cy - n.y) * centerPull;
      }

      // Apply velocities
      for (const n of ns) {
        if (n.type === "query") {
          n.x = cx;
          n.y = cy;
          n.vx = 0;
          n.vy = 0;
          continue;
        }
        n.vx *= dampening;
        n.vy *= dampening;
        n.x += n.vx;
        n.y += n.vy;
        // Boundary clamping
        n.x = Math.max(40, Math.min(width - 40, n.x));
        n.y = Math.max(40, Math.min(height - 40, n.y));
      }

      const newPos: Record<string, { x: number; y: number }> = {};
      for (const n of ns) {
        newPos[n.id] = { x: n.x, y: n.y };
      }
      setPositions(newPos);

      iterRef.current++;
      const maxIter = nodes.length > 15 ? 100 : 150;
      if (iterRef.current < maxIter) {
        frameRef.current = requestAnimationFrame(simulate);
      }
    };

    frameRef.current = requestAnimationFrame(simulate);

    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [nodes, edges, width, height]);

  return positions;
}

export default function KnowledgeGraph({
  triplets,
  leads = [],
  query,
  showInlineCard = true,
  onSelectNode,
  onSelectEdge,
  large = false,
}: KnowledgeGraphProps) {
  // Defensive guards for potentially null/invalid inputs from backend
  const safeTriplets = React.useMemo(
    () =>
      Array.isArray(triplets)
        ? triplets.filter(
            (t) =>
              t &&
              typeof t.source === "string" &&
              t.source.trim().length > 0 &&
              typeof t.target === "string" &&
              t.target.trim().length > 0
          )
        : [],
    [triplets]
  );

  const safeLeads = React.useMemo(
    () =>
      Array.isArray(leads)
        ? leads.filter(
            (lead): lead is Lead =>
              !!lead &&
              typeof lead.name === "string" &&
              lead.name.trim().length > 0
          )
        : [],
    [leads]
  );

  const safeQuery = typeof query === "string" ? query : String(query ?? "");

  const containerRef = useRef<HTMLDivElement>(null);
  const defaultH = large ? 520 : 420;
  const [dimensions, setDimensions] = useState({ width: large ? 800 : 600, height: defaultH });
  const [selectedNode, setSelectedNode] = useState<NodeInfo | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<EdgeInfo | null>(null);
  const [cardPos, setCardPos] = useState({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const [dragNode, setDragNode] = useState<string | null>(null);
  const [dragPositions, setDragPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [mounted, setMounted] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ panX: 0, panY: 0, mouseX: 0, mouseY: 0 });

  useEffect(() => {
    setMounted(true);
  }, []);

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      const minW = large ? 600 : 400;
      const minH = large ? 420 : 350;
      setDimensions({ width: Math.max(minW, width), height: Math.max(minH, height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [large]);

  // Build graph data from triplets
  const { graphNodes, graphEdges } = React.useMemo(() => {
    const nodeMap = new Map<string, GraphNode>();
    const edgeList: GraphEdge[] = [];

    // Query node at center
    nodeMap.set("__query__", {
      id: "__query__",
      name:
        safeQuery.length > 30
          ? safeQuery.substring(0, 30) + "…"
          : safeQuery,
      type: "query",
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      description: safeQuery,
    });

    // Entity nodes from triplets
    for (const t of safeTriplets) {
      const source = (t.source ?? "").trim();
      const target = (t.target ?? "").trim();
      if (!source || !target) continue;

      if (!nodeMap.has(source)) {
        nodeMap.set(source, {
          id: source,
          name: source,
          type: "entity",
          x: 0,
          y: 0,
          vx: 0,
          vy: 0,
        });
      }
      if (!nodeMap.has(target)) {
        nodeMap.set(target, {
          id: target,
          name: target,
          type: "entity",
          x: 0,
          y: 0,
          vx: 0,
          vy: 0,
        });
      }

      const relation = (t.relation ?? "").trim();

      edgeList.push({
        id: `${source}--${relation}--${target}`,
        source,
        target,
        relation,
        audit: t.audit || 1.0,
        description: t.description,
        explanation: t.explanation ?? undefined,
        cross_document: t.cross_document ?? false,
      });
    }

    // Hidden edges from query to all entity nodes
    for (const [key, node] of nodeMap) {
      if (key !== "__query__" && node.type === "entity") {
        edgeList.push({
          id: `__query__--relevance--${key}`,
          source: "__query__",
          target: key,
          relation: "relevant_to",
          audit: 1.0,
          description: "Retrieved as relevant to the user query.",
        });
      }
    }

    // Lead nodes
    for (const lead of safeLeads) {
      const leadName = lead.name.trim();
      if (!nodeMap.has(leadName)) {
        nodeMap.set(leadName, {
          id: leadName,
          name: leadName,
          type: "lead",
          x: 0,
          y: 0,
          vx: 0,
          vy: 0,
          description: lead.description || undefined,
        });
        edgeList.push({
          id: `__query__--lead--${leadName}`,
          source: "__query__",
          target: leadName,
          relation: "discovery_lead",
          audit: 0.7,
          description: "Community-based discovery lead.",
        });
      }
    }

    return {
      graphNodes: Array.from(nodeMap.values()),
      graphEdges: edgeList,
    };
  }, [safeTriplets, safeLeads, safeQuery]);

  const positions = useForceLayout(
    graphNodes,
    graphEdges,
    dimensions.width,
    dimensions.height
  );

  // Build node info for the card
  const buildNodeInfo = useCallback(
    (nodeId: string): NodeInfo => {
      const node = graphNodes.find((n) => n.id === nodeId)!;
      const connections: NodeInfo["connections"] = [];

      for (const e of graphEdges) {
        if (e.source === "__query__" || e.target === "__query__") continue;
        if (e.source === nodeId) {
          connections.push({ name: e.target, relation: e.relation, direction: "out" });
        }
        if (e.target === nodeId) {
          connections.push({ name: e.source, relation: e.relation, direction: "in" });
        }
      }

      // Find a direct triplet involving this node to build queryRelation
      const relatedTriplet = safeTriplets.find(
        (t) => t.source === nodeId || t.target === nodeId
      );
      let queryRelation: string | undefined;
      if (relatedTriplet) {
        const rel = relatedTriplet.relation ?? "";
        const src = relatedTriplet.source ?? "";
        const tgt = relatedTriplet.target ?? "";
        queryRelation = `Connected via "${rel}" between ${src} and ${tgt}.`;
      }

      let explanation: string | undefined;
      if (node.type === "lead") {
        explanation = safeLeads.find((lead) => lead.name === nodeId)?.explanation || undefined;
      } else {
        explanation = relatedTriplet?.explanation || undefined;
      }

      return {
        id: node.id,
        name: node.name,
        type: node.type,
        description: node.description,
        explanation,
        connections,
        queryRelation,
      };
    },
    [graphNodes, graphEdges, safeTriplets, safeLeads]
  );

  const buildEdgeInfo = useCallback(
    (edgeId: string): EdgeInfo | null => {
      const edge = graphEdges.find((e) => e.id === edgeId);
      if (!edge) return null;
      return {
        id: edge.id,
        source: edge.source === "__query__" ? query : edge.source,
        target: edge.target,
        relation: edge.relation,
        audit: edge.audit,
        description: edge.description,
        explanation: edge.explanation || undefined,
        cross_document: edge.cross_document,
      };
    },
    [graphEdges, query]
  );

  const effectivePositions = React.useMemo(() => {
    const out = { ...positions };
    for (const [id, pos] of Object.entries(dragPositions)) {
      out[id] = pos;
    }
    return out;
  }, [positions, dragPositions]);

  const handleNodeClick = (nodeId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (dragNode) return;
    const pos = effectivePositions[nodeId];
    if (!pos) return;

    const info = buildNodeInfo(nodeId);
    setSelectedEdge(null);
    setSelectedNode(info);
    setCardPos({ x: pos.x, y: pos.y });
    if (onSelectNode) onSelectNode(info);
    if (onSelectEdge) onSelectEdge(null);
  };

  const handleEdgeClick = (edgeId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    // Don't show info cards for hidden query-entity edges
    if (edgeId.startsWith("__query__")) return;

    const edge = buildEdgeInfo(edgeId);
    if (!edge) return;
    const srcPos = effectivePositions[graphEdges.find((e) => e.id === edgeId)!.source];
    const tgtPos = effectivePositions[graphEdges.find((e) => e.id === edgeId)!.target];
    if (!srcPos || !tgtPos) return;

    setSelectedNode(null);
    setSelectedEdge(edge);
    setCardPos({
      x: (srcPos.x + tgtPos.x) / 2,
      y: (srcPos.y + tgtPos.y) / 2,
    });
    if (onSelectEdge) onSelectEdge(edge);
    if (onSelectNode) onSelectNode(null);
  };

  const closeCard = () => {
    setSelectedNode(null);
    setSelectedEdge(null);
    if (onSelectNode) onSelectNode(null);
    if (onSelectEdge) onSelectEdge(null);
  };

  // Drag handlers
  const handleMouseDown = (nodeId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragNode(nodeId);
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragNode || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setDragPositions((prev) => ({ ...prev, [dragNode]: { x, y } }));
    },
    [dragNode]
  );

  const handleMouseUp = useCallback(() => {
    setDragNode(null);
    setDragPositions({});
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.max(0.4, Math.min(3, z + delta)));
  }, []);

  const handlePanStart = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget || (e.target as SVGElement).tagName === "svg") {
      setIsPanning(true);
      panStartRef.current = { panX: pan.x, panY: pan.y, mouseX: e.clientX, mouseY: e.clientY };
    }
  }, [pan]);

  useEffect(() => {
    if (!isPanning) return;
    const onMove = (e: MouseEvent) => setPan({
      x: panStartRef.current.panX + (e.clientX - panStartRef.current.mouseX),
      y: panStartRef.current.panY + (e.clientY - panStartRef.current.mouseY),
    });
    const onUp = () => setIsPanning(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [isPanning]);

  useEffect(() => {
    if (dragNode) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [dragNode, handleMouseMove, handleMouseUp]);

  // Node colors and sizes (larger when large=true)
  const scale = large ? 1.2 : 1;
  const nodeStyle = (type: string, id: string) => {
    const isHovered = hoveredNode === id;
    switch (type) {
      case "query":
        return {
          r: (isHovered ? 28 : 24) * scale,
          fill: "var(--gilded-gold)",
          stroke: "#A68A1E",
          strokeWidth: 2.5,
          className: "graph-node-query",
        };
      case "lead":
        return {
          r: (isHovered ? 16 : 13) * scale,
          fill: "var(--validated-green)",
          stroke: "#2A4A30",
          strokeWidth: 1.5,
          className: "graph-node-lead",
        };
      default:
        return {
          r: (isHovered ? 20 : 17) * scale,
          fill: "var(--seal-red)",
          stroke: "#6B0A0A",
          strokeWidth: 2,
          className: "graph-node-entity",
        };
    }
  };

  if (!mounted) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.4 }}
      className="w-full"
    >
      <div
        ref={containerRef}
        className="relative w-full bg-[#FAFAF8]/60 border border-[#E5E5E3]/50 rounded-lg overflow-hidden"
        style={{ minHeight: large ? 420 : 350, height: "100%", cursor: isPanning ? "grabbing" : dragNode ? "grabbing" : "default" }}
        onClick={closeCard}
        onWheel={handleWheel}
      >
        {/* SVG paper texture background */}
        <div className="absolute inset-0 paper-texture pointer-events-none opacity-50" />

        <svg
          width={dimensions.width}
          height={dimensions.height}
          className="relative z-10"
          style={{ cursor: dragNode ? "grabbing" : isPanning ? "grabbing" : "default" }}
        >
          <g transform={`translate(${dimensions.width / 2 + pan.x}, ${dimensions.height / 2 + pan.y}) scale(${zoom}) translate(${-dimensions.width / 2}, ${-dimensions.height / 2})`}>
          {/* Invisible background for panning */}
          <rect x={0} y={0} width={dimensions.width} height={dimensions.height} fill="transparent" onMouseDown={handlePanStart} style={{ cursor: isPanning ? "grabbing" : "grab" }} />
          <defs>
            {/* Glow filter for query node */}
            <filter id="queryGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {/* Arrow marker */}
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="6"
              refX="8"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#C5A028" opacity="0.7" />
            </marker>
            <marker
              id="arrowheadHover"
              markerWidth="8"
              markerHeight="6"
              refX="8"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 8 3, 0 6" fill="#C5A028" />
            </marker>
          </defs>

          {/* Edges */}
          {graphEdges.map((edge) => {
            const srcPos = effectivePositions[edge.source];
            const tgtPos = effectivePositions[edge.target];
            if (!srcPos || !tgtPos) return null;

            const isQueryEdge = edge.source === "__query__";
            const isHovered = hoveredEdge === edge.id;

            // Calculate edge endpoint offset to stop at node border
            const dx = tgtPos.x - srcPos.x;
            const dy = tgtPos.y - srcPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const srcNode = graphNodes.find((n) => n.id === edge.source);
            const tgtNode = graphNodes.find((n) => n.id === edge.target);
            const srcR = srcNode?.type === "query" ? 24 : srcNode?.type === "lead" ? 13 : 17;
            const tgtR = tgtNode?.type === "query" ? 24 : tgtNode?.type === "lead" ? 13 : 17;

            const x1 = srcPos.x + (dx / dist) * srcR;
            const y1 = srcPos.y + (dy / dist) * srcR;
            const x2 = tgtPos.x - (dx / dist) * (tgtR + 10);
            const y2 = tgtPos.y - (dy / dist) * (tgtR + 10);

            // Midpoint for label
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;

            return (
              <g key={edge.id}>
                {/* Invisible wider path for easier clicking */}
                {!isQueryEdge && (
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="transparent"
                    strokeWidth="16"
                    className="cursor-pointer"
                    onClick={(e) => handleEdgeClick(edge.id, e)}
                    onMouseEnter={() => setHoveredEdge(edge.id)}
                    onMouseLeave={() => setHoveredEdge(null)}
                  />
                )}
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={
                    isQueryEdge
                      ? "rgba(212,175,55,0.15)"
                      : edge.cross_document
                      ? isHovered
                        ? "#C5A028"
                        : "rgba(212,175,55,0.7)"
                      : isHovered
                      ? "#C5A028"
                      : "rgba(82,82,82,0.4)"
                  }
                  strokeWidth={isQueryEdge ? 1 : isHovered ? 2.5 : edge.cross_document ? 2 : 1.5}
                  strokeDasharray={isQueryEdge || edge.cross_document ? "6,4" : "none"}
                  markerEnd={
                    isQueryEdge
                      ? undefined
                      : isHovered
                      ? "url(#arrowheadHover)"
                      : "url(#arrowhead)"
                  }
                  className={`transition-all duration-200 ${
                    !isQueryEdge ? "cursor-pointer" : ""
                  }`}
                  pointerEvents={isQueryEdge ? "none" : "auto"}
                />
                {/* Edge label with tooltip for full relation name */}
                {!isQueryEdge && (
                  <g>
                    <title>
                      {edge.relation ?? ""}
                      {edge.cross_document ? " (Cross-document connection)" : ""}
                    </title>
                    {edge.cross_document && (
                      <text
                        x={mx}
                        y={my - 18}
                        textAnchor="middle"
                        className="fill-[#8B6914] text-[7px] font-semibold pointer-events-none select-none"
                        fontFamily="system-ui, sans-serif"
                      >
                        Cross-document
                      </text>
                    )}
                    <text
                      x={mx}
                      y={my - 6}
                      textAnchor="middle"
                      className="fill-[#737373] text-[8px] pointer-events-none select-none"
                      fontFamily="JetBrains Mono, monospace"
                      opacity={isHovered ? 1 : 0.7}
                    >
                      {(() => {
                        const label = edge.relation ?? "";
                        return label.length > 18
                          ? label.substring(0, 18) + "…"
                          : label;
                      })()}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {graphNodes.map((node, idx) => {
            const pos = effectivePositions[node.id];
            if (!pos) return null;
            const style = nodeStyle(node.type, node.id);
            const isHovered = hoveredNode === node.id;

            return (
              <g
                key={node.id}
                className="cursor-pointer"
                onClick={(e) => handleNodeClick(node.id, e)}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onMouseDown={(e) => handleMouseDown(node.id, e)}
              >
                {/* Glow circle for hovered/query nodes */}
                {(node.type === "query" || isHovered) && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={style.r + 8}
                    fill="none"
                    stroke={style.fill}
                    strokeWidth="1"
                    opacity={node.type === "query" ? 0.3 : 0.4}
                    className={
                      node.type === "query" ? "graph-pulse-ring" : ""
                    }
                  />
                )}
                {/* Main circle */}
                <motion.circle
                  cx={pos.x}
                  cy={pos.y}
                  initial={{ r: 0 }}
                  animate={{ r: style.r }}
                  transition={{
                    delay: 0.1 + idx * 0.06,
                    type: "spring",
                    stiffness: 300,
                    damping: 20,
                  }}
                  fill={style.fill}
                  stroke={style.stroke}
                  strokeWidth={style.strokeWidth}
                  filter={node.type === "query" ? "url(#queryGlow)" : undefined}
                  opacity={node.type === "lead" ? 0.7 : 1}
                  className="transition-all duration-150"
                />
                {/* Label */}
                <text
                  x={pos.x}
                  y={pos.y + style.r + (large ? 18 : 14)}
                  textAnchor="middle"
                  className="select-none pointer-events-none"
                  fontFamily="EB Garamond, serif"
                  fontWeight={node.type === "query" ? 700 : 600}
                  fontSize={node.type === "query" ? (large ? 13 : 11) : (large ? 11 : 10)}
                  fill={
                    node.type === "query"
                      ? "#C5A028"
                      : isHovered
                      ? "#1A1A1A"
                      : "#525252"
                  }
                >
                  {node.name.length > (large ? 24 : 20)
                    ? node.name.substring(0, large ? 24 : 20) + "…"
                    : node.name}
                </text>
                {/* Type icon inside node */}
                <text
                  x={pos.x}
                  y={pos.y + 4}
                  textAnchor="middle"
                  fontFamily="system-ui"
                  fontSize={node.type === "query" ? 14 : 11}
                  fill="#FFFFFF"
                  className="pointer-events-none select-none"
                >
                  {node.type === "query" ? "?" : node.type === "lead" ? "◇" : "●"}
                </text>
              </g>
            );
          })}
          </g>
        </svg>

        {/* Zoom reset button */}
        <button
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
          className="absolute top-2 right-2 px-2 py-1 bg-white border border-[#E5E5E3] rounded text-[#737373] hover:text-[#1A1A1A] text-xs font-mono z-10"
          title="Reset zoom"
        >
          Reset
        </button>

        {/* Legend — more prominent when large */}
        <div className={`absolute bottom-3 left-3 right-3 flex flex-wrap items-center gap-4 z-20 ${large ? "gap-6" : ""}`}>
          <div className="flex items-center gap-2">
            <div className={`rounded-full bg-[#C5A028] shadow-sm ${large ? "w-4 h-4" : "w-3 h-3"}`} />
            <span className={`font-mono text-[#737373] uppercase tracking-wider ${large ? "text-xs" : "text-[9px]"}`}>
              Your question
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`rounded-full bg-[#7A1A1A] shadow-sm ${large ? "w-4 h-4" : "w-3 h-3"}`} />
            <span className={`font-mono text-[#737373] uppercase tracking-wider ${large ? "text-xs" : "text-[9px]"}`}>
              Evidence
            </span>
          </div>
          {safeLeads.length > 0 && (
            <div className="flex items-center gap-2">
              <div className={`rounded-full bg-[#2D6A4F] shadow-sm ${large ? "w-4 h-4" : "w-3 h-3"}`} />
              <span className={`font-mono text-[#737373] uppercase tracking-wider ${large ? "text-xs" : "text-[9px]"}`}>
                Leads
              </span>
            </div>
          )}
          {safeTriplets.some((t) => t.cross_document) && (
            <div className="flex items-center gap-2">
              <div className={`${large ? "w-8" : "w-6"} h-0.5 rounded`} style={{ borderBottom: "2px dashed #C5A028" }} />
              <span className={`font-mono text-[#737373] uppercase tracking-wider ${large ? "text-xs" : "text-[9px]"}`}>
                Cross-document
              </span>
            </div>
          )}
          <span className={`text-[#737373] italic ml-auto ${large ? "text-xs" : "text-[9px]"}`}>
            Click any node or line for details
          </span>
        </div>

        {/* Info Cards */}
        <div className="absolute inset-0 pointer-events-none z-30">
          {showInlineCard && (
            <AnimatePresence>
              {(selectedNode || selectedEdge) && (
                <GraphInfoCard
                  nodeInfo={selectedNode}
                  edgeInfo={selectedEdge}
                  position={cardPos}
                  onClose={closeCard}
                  query={query}
                />
              )}
            </AnimatePresence>
          )}
        </div>
      </div>
    </motion.div>
  );
}
