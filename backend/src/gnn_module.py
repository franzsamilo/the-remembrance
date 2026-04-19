from __future__ import annotations

import copy
import json
import os
import threading
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config, logger
from src.helpers import utc_now_iso as _utc_now_iso

CHECKPOINT_DIR = os.path.join(Config.BASE_DIR, "run_logs")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "compgcn_best.pt")
CHECKPOINT_META_PATH = os.path.join(CHECKPOINT_DIR, "compgcn_best_meta.json")

# Training history for UI visualization (module-level, survives across requests)
_training_history: dict = {"epochs": [], "early_stop_epoch": None, "best_epoch": None}
_training_history_lock = threading.Lock()


def _compute_auc_roc(labels: torch.Tensor, probabilities: torch.Tensor) -> float | None:
    if labels.numel() == 0:
        return None

    labels = labels.float()
    pos_total = labels.sum()
    neg_total = labels.numel() - pos_total
    if pos_total.item() == 0 or neg_total.item() == 0:
        return None

    sort_idx = torch.argsort(probabilities, descending=True)
    sorted_labels = labels[sort_idx]
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1.0 - sorted_labels, dim=0)

    tpr = tps / pos_total
    fpr = fps / neg_total
    tpr = torch.cat([torch.tensor([0.0], device=tpr.device), tpr])
    fpr = torch.cat([torch.tensor([0.0], device=fpr.device), fpr])
    return float(torch.trapz(tpr, fpr).item())


def _split_edge_indices(num_edges: int, validation_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(num_edges)
    if num_edges < 4:
        return indices, torch.empty(0, dtype=torch.long)

    validation_size = max(1, int(num_edges * validation_split))
    validation_size = min(validation_size, num_edges - 1)
    return indices[validation_size:], indices[:validation_size]


def _build_type_pools(node_type: torch.Tensor) -> dict[int, torch.Tensor]:
    """Partition node indices by schema-label id.

    Pools with fewer than 2 nodes are dropped — the sampler falls back to
    uniform sampling for endpoints whose label has no pool. The sentinel
    label -1 (unlabeled / not-in-schema nodes) is always excluded.
    """
    pools: dict[int, torch.Tensor] = {}
    unique_labels = torch.unique(node_type).tolist()
    for label_id in unique_labels:
        if label_id == -1:
            continue
        mask = node_type == label_id
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            continue
        pools[int(label_id)] = indices.to(torch.long)
    return pools


def _sample_negative_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_negatives: int = 1,
    positive_triples: set[tuple[int, int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample negatives by corrupting head or tail. Optionally filter out false negatives."""
    if edge_index.size(1) == 0:
        return edge_index, edge_type

    neg_edges_list = []
    neg_types_list = []
    # Pre-materialize edge_type once to Python list; per-element .item() inside
    # the hot inner loop triggers PyTorch SymInt errors / memory faults on
    # Windows when called ~1.9M times per epoch.
    rel_cpu = edge_type.tolist() if positive_triples is not None else None
    for _ in range(num_negatives):
        negative_edges = edge_index.clone()
        corrupt_tail_mask = torch.rand(edge_index.size(1)) > 0.5
        max_retries = 20
        for attempt in range(max_retries):
            random_nodes = torch.randint(0, num_nodes, (edge_index.size(1),), dtype=torch.long)
            negative_edges[1, corrupt_tail_mask] = random_nodes[corrupt_tail_mask]
            negative_edges[0, ~corrupt_tail_mask] = random_nodes[~corrupt_tail_mask]
            if positive_triples is None:
                break
            # Batch convert each row once per attempt (instead of per-index .item())
            srcs = negative_edges[0].tolist()
            dsts = negative_edges[1].tolist()
            bad = [
                i
                for i in range(len(rel_cpu))
                if (srcs[i], dsts[i], rel_cpu[i]) in positive_triples
            ]
            if not bad:
                break
            # Keep per-index randint to preserve RNG-progression determinism
            # with the original implementation.
            for i in bad:
                random_nodes[i] = int(torch.randint(0, num_nodes, (1,), dtype=torch.long).item())
            negative_edges[1, corrupt_tail_mask] = random_nodes[corrupt_tail_mask]
            negative_edges[0, ~corrupt_tail_mask] = random_nodes[~corrupt_tail_mask]
        neg_edges_list.append(negative_edges)
        neg_types_list.append(edge_type.clone())

    neg_edges = torch.cat(neg_edges_list, dim=1)
    neg_types = torch.cat(neg_types_list, dim=0)
    return neg_edges, neg_types


class CompGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.node_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.rel_linear = nn.Linear(out_channels, out_channels, bias=False)
        self.self_linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        rel_emb: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.node_linear(x[src]) + self.rel_linear(rel_emb[edge_type])

        aggregated = torch.zeros(
            x.size(0),
            messages.size(1),
            device=x.device,
            dtype=messages.dtype,
        )
        aggregated.index_add_(0, dst, messages)
        degrees = torch.bincount(dst, minlength=x.size(0)).clamp_min(1).unsqueeze(1)
        aggregated = aggregated / degrees
        return aggregated + self.self_linear(x)


class CompGCNAuditModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.node_projection = nn.Linear(in_channels, hidden_channels)
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        self.layer1 = CompGCNLayer(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.layer2 = CompGCNLayer(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.layer3 = CompGCNLayer(hidden_channels, hidden_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.node_projection(x)
        x = self.dropout(F.relu(self.norm1(self.layer1(x, edge_index, edge_type, self.rel_emb.weight))))
        x = self.dropout(F.relu(self.norm2(self.layer2(x, edge_index, edge_type, self.rel_emb.weight))))
        x = self.layer3(x, edge_index, edge_type, self.rel_emb.weight)
        return self.dropout(x)

    def edge_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = self.rel_emb(edge_type)
        return torch.sum(x[src] * rel * x[dst], dim=1)

    def edge_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.edge_logits(x, edge_index, edge_type))


def _build_positive_triples(data) -> set[tuple[int, int, int]]:
    """Set of (src, dst, rel) for all edges in the graph."""
    triples = set()
    for i in range(data.edge_index.size(1)):
        s = int(data.edge_index[0, i].item())
        t = int(data.edge_index[1, i].item())
        r = int(data.edge_type[i].item())
        triples.add((s, t, r))
    return triples


def _evaluate_auc(model: CompGCNAuditModel, data, edge_indices: torch.Tensor, neg_ratio: int = 1, positive_triples: set[tuple[int, int, int]] | None = None) -> float | None:
    if edge_indices.numel() == 0:
        return None

    if positive_triples is None:
        positive_triples = _build_positive_triples(data)

    encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
    pos_logits = model.edge_logits(
        encoded_nodes,
        data.edge_index[:, edge_indices],
        data.edge_type[edge_indices],
    )
    neg_edges, neg_types = _sample_negative_edges(
        data.edge_index[:, edge_indices],
        data.edge_type[edge_indices],
        data.x.size(0),
        num_negatives=neg_ratio,
        positive_triples=positive_triples,
    )
    neg_logits = model.edge_logits(encoded_nodes, neg_edges, neg_types)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    probabilities = torch.sigmoid(torch.cat([pos_logits, neg_logits]))
    return _compute_auc_roc(labels, probabilities)


def _evaluate_mrr(model: CompGCNAuditModel, data, edge_indices: torch.Tensor, neg_ratio: int = 5, positive_triples: set[tuple[int, int, int]] | None = None) -> float | None:
    """Mean Reciprocal Rank: for each positive edge, rank it among negatives; MRR = mean(1/rank)."""
    if edge_indices.numel() == 0:
        return None

    if positive_triples is None:
        positive_triples = _build_positive_triples(data)

    encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
    reciprocal_ranks = []

    for i in range(edge_indices.numel()):
        idx = edge_indices[i].item()
        pos_edge = data.edge_index[:, idx : idx + 1]
        pos_type = data.edge_type[idx : idx + 1]
        pos_logit = model.edge_logits(encoded_nodes, pos_edge, pos_type)

        neg_edges, neg_types = _sample_negative_edges(
            pos_edge, pos_type, data.x.size(0), num_negatives=max(neg_ratio, 5), positive_triples=positive_triples
        )
        neg_logits = model.edge_logits(encoded_nodes, neg_edges, neg_types)

        all_scores = torch.cat([pos_logit, neg_logits])
        sorted_desc = torch.argsort(all_scores, descending=True)
        pos_rank = (sorted_desc == 0).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / pos_rank)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else None


def run_audit():
    """Train and evaluate a local CompGCN-style plausibility model and sync scores."""
    from src.db import DatabaseManager
    from src.gnn_loader import GNNLoader

    # Reproducible runs: same seed yields same AUC/MRR for panel evaluation
    seed = Config.COMPGCN_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = GNNLoader()
    result = loader.fetch_graph_data()

    if not result:
        logger.error("Audit failed: No graph data found.")
        return

    data, rel_types, _node_id_map = result
    num_rels = len(rel_types)
    if num_rels == 0:
        logger.warning("Audit skipped: graph has no auditable relationship types.")
        return

    audit_run_id = str(uuid.uuid4())
    audit_started_at = _utc_now_iso()
    train_idx, val_idx = _split_edge_indices(
        data.edge_index.size(1),
        Config.COMPGCN_VALIDATION_SPLIT,
    )
    positive_triples = _build_positive_triples(data)

    neg_ratio = max(1, Config.COMPGCN_NEG_RATIO)
    label_smoothing = Config.COMPGCN_LABEL_SMOOTHING
    pos_target = 1.0 - label_smoothing
    neg_target = label_smoothing
    loss_mode = Config.COMPGCN_LOSS
    bpr_margin = Config.COMPGCN_BPR_MARGIN

    model = CompGCNAuditModel(
        in_channels=data.x.size(1),
        hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
        num_relations=num_rels,
        dropout=Config.COMPGCN_DROPOUT,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.COMPGCN_LEARNING_RATE,
        weight_decay=Config.COMPGCN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()
    logger.info("CompGCN training loss=%s (margin=%s)", loss_mode, bpr_margin)

    best_state = copy.deepcopy(model.state_dict())
    best_auc = float("-inf")
    best_epoch_idx = None  # Track best epoch (1-indexed)
    final_train_loss = None
    patience_counter = 0
    epoch_metrics = []

    for epoch in range(Config.COMPGCN_EPOCHS):
        model.train()
        optimizer.zero_grad()

        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        pos_logits = model.edge_logits(
            encoded_nodes,
            data.edge_index[:, train_idx],
            data.edge_type[train_idx],
        )
        neg_edges, neg_types = _sample_negative_edges(
            data.edge_index[:, train_idx],
            data.edge_type[train_idx],
            data.x.size(0),
            num_negatives=neg_ratio,
            positive_triples=positive_triples,
        )
        neg_logits = model.edge_logits(encoded_nodes, neg_edges, neg_types)

        if loss_mode == "bpr":
            # Pairwise: neg_edges = neg_ratio repetitions of pos ordering
            # (see _sample_negative_edges: list-cat preserves [pos_0..pos_N] grouped by rep)
            pos_expanded = pos_logits.repeat(neg_ratio)
            diff = pos_expanded - neg_logits - bpr_margin
            loss = -F.logsigmoid(diff).mean()
        else:
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat(
                [torch.full_like(pos_logits, pos_target), torch.full_like(neg_logits, neg_target)]
            )
            loss = criterion(logits, labels)
        loss.backward()
        if Config.COMPGCN_GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.COMPGCN_GRAD_CLIP)
        optimizer.step()
        final_train_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            current_auc = _evaluate_auc(model, data, val_idx, neg_ratio, positive_triples)
        if current_auc is None:
            current_auc = _evaluate_auc(model, data, train_idx, neg_ratio, positive_triples)

        if current_auc is not None:
            scheduler.step(current_auc)
        if current_auc is not None and current_auc > best_auc:
            best_auc = current_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch_idx = epoch + 1
            # Persist best state to disk so partial progress survives Windows
            # transient PyTorch crashes during long training runs.
            try:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save(best_state, CHECKPOINT_PATH)
                with open(CHECKPOINT_META_PATH, "w") as f:
                    json.dump({
                        "best_epoch": best_epoch_idx,
                        "best_auc": float(best_auc),
                        "train_loss": final_train_loss,
                        "hidden_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                        "loss_mode": loss_mode,
                        "num_relations": num_rels,
                        "in_channels": int(data.x.size(1)),
                        "saved_at": _utc_now_iso(),
                    }, f)
            except Exception as e:
                logger.warning("Checkpoint save failed at epoch %s: %s", epoch + 1, e)
        else:
            patience_counter += 1

        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": round(final_train_loss, 4) if final_train_loss is not None else None,
            "auc_roc": round(current_auc, 4) if current_auc is not None else None,
        })

        if patience_counter >= Config.COMPGCN_PATIENCE:
            logger.info(
                "CompGCN early stopping at epoch %s (no improvement for %s epochs)",
                epoch + 1,
                Config.COMPGCN_PATIENCE,
            )
            break

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == Config.COMPGCN_EPOCHS - 1:
            logger.info(
                "CompGCN epoch %s/%s loss=%.4f auc_roc=%s",
                epoch + 1,
                Config.COMPGCN_EPOCHS,
                final_train_loss,
                f"{current_auc:.4f}" if current_auc is not None else "unavailable",
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        plausibility_scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)
        final_auc = _evaluate_auc(model, data, val_idx, neg_ratio, positive_triples)
        if final_auc is None:
            final_auc = _evaluate_auc(model, data, train_idx, neg_ratio, positive_triples)
        final_mrr = _evaluate_mrr(model, data, val_idx, neg_ratio, positive_triples)
        if final_mrr is None:
            final_mrr = _evaluate_mrr(model, data, train_idx, neg_ratio, positive_triples)

    # Store training history for UI
    with _training_history_lock:
        global _training_history
        _training_history = {
            "epochs": epoch_metrics,
            "early_stop_epoch": (epoch + 1) if patience_counter >= Config.COMPGCN_PATIENCE else None,
            "best_epoch": best_epoch_idx,
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "loss_mode": loss_mode,
            "bpr_margin": bpr_margin,
        }

    logger.info(
        "CompGCN audit completed for %s relationships with auc_roc=%s mrr=%s",
        len(plausibility_scores),
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{final_mrr:.4f}" if final_mrr is not None else "unavailable",
    )

    # Training ran compute-only for many minutes; Aura Free may have dropped the
    # pooled TCP socket. Force a fresh driver before the write-back to avoid
    # "connection reset" / session-expired failures on the first sync query.
    driver = DatabaseManager.refresh()
    audit_completed_at = _utc_now_iso()

    with driver.session(database=Config.NEO4J_DATABASE) as session:
        logger.info("Syncing CompGCN plausibility scores to Neo4j...")
        # Batch update all scores in a single query instead of N+1 individual calls
        updates = []
        for i in range(data.edge_index.size(1)):
            updates.append({
                "rel_id": data.edge_rel_id[i],
                "score": float(plausibility_scores[i].item()),
            })
        batch_size = 500
        for batch_start in range(0, len(updates), batch_size):
            batch = updates[batch_start:batch_start + batch_size]
            session.run(
                """
                UNWIND $updates AS update
                MATCH ()-[r]->()
                WHERE elementId(r) = update.rel_id
                SET r.plausibility_score = update.score,
                    r.audit_score = update.score,
                    r.experimental_score = update.score,
                    r.audit_status = 'trained_experimental',
                    r.audit_mode = 'compgcn',
                    r.audit_model = 'CompGCNAuditModel',
                    r.audit_updated_at = $updated_at
                """,
                updates=batch,
                updated_at=audit_completed_at,
            )

        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = 'trained_experimental',
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $started_at,
                run.completed_at = $completed_at,
                run.audited_relationships = $audited_relationships,
                run.graph_nodes = $graph_nodes,
                run.graph_relationship_types = $graph_relationship_types,
                run.hidden_channels = $hidden_channels,
                run.epochs = $epochs,
                run.learning_rate = $learning_rate,
                run.weight_decay = $weight_decay,
                run.validation_split = $validation_split,
                run.patience = $patience,
                run.dropout = $dropout,
                run.label_smoothing = $label_smoothing,
                run.grad_clip = $grad_clip,
                run.neg_ratio = $neg_ratio,
                run.loss = $loss,
                run.bpr_margin = $bpr_margin,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.train_loss = $train_loss
            """,
            run_id=audit_run_id,
            started_at=audit_started_at,
            completed_at=audit_completed_at,
            audited_relationships=int(data.edge_index.size(1)),
            graph_nodes=int(data.x.size(0)),
            graph_relationship_types=int(num_rels),
            hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
            epochs=Config.COMPGCN_EPOCHS,
            learning_rate=Config.COMPGCN_LEARNING_RATE,
            weight_decay=Config.COMPGCN_WEIGHT_DECAY,
            validation_split=Config.COMPGCN_VALIDATION_SPLIT,
            patience=Config.COMPGCN_PATIENCE,
            dropout=Config.COMPGCN_DROPOUT,
            label_smoothing=Config.COMPGCN_LABEL_SMOOTHING,
            grad_clip=Config.COMPGCN_GRAD_CLIP,
            neg_ratio=neg_ratio,
            loss=loss_mode,
            bpr_margin=bpr_margin,
            auc_roc=final_auc,
            mrr=final_mrr,
            train_loss=final_train_loss,
        )
        logger.info("Neo4j CompGCN audit sync complete.")


def get_training_history() -> dict:
    """Return the last training run's per-epoch metrics."""
    with _training_history_lock:
        return copy.deepcopy(_training_history)


def recover_from_checkpoint() -> dict | None:
    """Load best_state from disk and sync plausibility scores to Neo4j without retraining.

    Used when a training run crashed partway through but a checkpoint was saved
    at a previous best-AUC epoch. Returns metrics dict or None if no checkpoint.
    """
    from src.db import DatabaseManager
    from src.gnn_loader import GNNLoader

    if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_META_PATH)):
        logger.warning("recover_from_checkpoint: no checkpoint at %s", CHECKPOINT_PATH)
        return None

    with open(CHECKPOINT_META_PATH) as f:
        meta = json.load(f)
    logger.info(
        "Recovering from checkpoint: epoch=%s auc=%.4f loss=%s saved_at=%s",
        meta.get("best_epoch"),
        meta.get("best_auc", 0.0),
        meta.get("loss_mode"),
        meta.get("saved_at"),
    )

    seed = Config.COMPGCN_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = GNNLoader()
    result = loader.fetch_graph_data()
    if not result:
        logger.error("recover_from_checkpoint: no graph data")
        return None
    data, rel_types, _node_id_map = result
    num_rels = len(rel_types)

    model = CompGCNAuditModel(
        in_channels=data.x.size(1),
        hidden_channels=meta.get("hidden_channels", Config.COMPGCN_HIDDEN_CHANNELS),
        num_relations=num_rels,
        dropout=Config.COMPGCN_DROPOUT,
    )
    best_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(best_state)
    model.eval()

    neg_ratio = max(1, Config.COMPGCN_NEG_RATIO)
    positive_triples = _build_positive_triples(data)
    _, val_idx = _split_edge_indices(
        data.edge_index.size(1), Config.COMPGCN_VALIDATION_SPLIT
    )
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)
        final_auc = _evaluate_auc(model, data, val_idx, neg_ratio, positive_triples)
        final_mrr = _evaluate_mrr(model, data, val_idx, neg_ratio, positive_triples)

    logger.info(
        "Checkpoint eval: auc=%s mrr=%s",
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{final_mrr:.4f}" if final_mrr is not None else "unavailable",
    )

    driver = DatabaseManager.refresh()
    audit_run_id = str(uuid.uuid4())
    audit_completed_at = _utc_now_iso()
    with driver.session(database=Config.NEO4J_DATABASE) as session:
        logger.info("Syncing recovered plausibility scores to Neo4j...")
        updates = []
        for i in range(data.edge_index.size(1)):
            updates.append({"rel_id": data.edge_rel_id[i], "score": float(scores[i].item())})
        batch_size = 500
        for batch_start in range(0, len(updates), batch_size):
            batch = updates[batch_start:batch_start + batch_size]
            session.run(
                """
                UNWIND $updates AS update
                MATCH ()-[r]->()
                WHERE elementId(r) = update.rel_id
                SET r.plausibility_score = update.score,
                    r.audit_score = update.score,
                    r.experimental_score = update.score,
                    r.audit_status = 'recovered_checkpoint',
                    r.audit_mode = 'compgcn',
                    r.audit_model = 'CompGCNAuditModel',
                    r.audit_updated_at = $updated_at
                """,
                updates=batch,
                updated_at=audit_completed_at,
            )
        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = 'recovered_checkpoint',
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $saved_at,
                run.completed_at = $completed_at,
                run.best_epoch = $best_epoch,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.loss = $loss
            """,
            run_id=audit_run_id,
            saved_at=meta.get("saved_at"),
            completed_at=audit_completed_at,
            best_epoch=meta.get("best_epoch"),
            auc_roc=final_auc,
            mrr=final_mrr,
            loss=meta.get("loss_mode"),
        )
        logger.info("Neo4j recovery sync complete.")

    # Update training history for UI
    with _training_history_lock:
        global _training_history
        _training_history = {
            "epochs": _training_history.get("epochs", []),
            "best_epoch": meta.get("best_epoch"),
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "loss_mode": meta.get("loss_mode"),
            "recovered": True,
        }

    return {
        "best_epoch": meta.get("best_epoch"),
        "final_auc_roc": final_auc,
        "final_mrr": final_mrr,
        "saved_at": meta.get("saved_at"),
    }


if __name__ == "__main__":
    run_audit()
