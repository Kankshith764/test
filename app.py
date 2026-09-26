"""
FastAPI fraud-scoring service. Scores a transaction's local subgraph in real time using
the trained SentinelFraudNet (GATv2 + Transformer) checkpoint, returns a calibrated
0-1000 risk score plus an attention-weight-based explanation, and hashes that explanation
for the caller to submit on-chain via PaymentEscrow.submitScore().

If checkpoints/sentinelfraud.pt doesn't exist yet (i.e. you haven't run train.py), this
falls back to an untrained model with a clear warning in every response -- so the API
contract is usable for integration testing before training finishes.
"""
import hashlib
import json
import os
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from data import FEATURE_NAMES, NODE_DIM
from model import load_model, edges_to_adj_mask

CKPT_PATH = os.environ.get("MODEL_CKPT", "checkpoints/sentinelfraud.pt")
RISK_THRESHOLD = int(os.environ.get("RISK_THRESHOLD", "700"))

app = FastAPI(title="SentinelPay Fraud Engine", version="1.0.0")

_model = load_model(NODE_DIM, ckpt_path=CKPT_PATH)
_is_trained = os.path.exists(CKPT_PATH)
_temperature = 1.0
if _is_trained:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    _temperature = float(ckpt.get("temperature", 1.0))


class Node(BaseModel):
    features: List[float]  # must match len(FEATURE_NAMES) -- see /schema


class Edge(BaseModel):
    src: int
    dst: int


class ScoreRequest(BaseModel):
    tx_id: str
    nodes: List[Node]
    edges: List[Edge]
    target_node_index: int


class ScoreResponse(BaseModel):
    tx_id: str
    risk_score: int
    flagged: bool
    calibrated_probability: float
    audit_hash: str
    top_contributing_neighbors: List[dict]
    model_trained: bool
    explanation_note: Optional[str] = None


@app.get("/schema")
def schema():
    return {"feature_names": FEATURE_NAMES, "node_dim": NODE_DIM, "risk_threshold": RISK_THRESHOLD}


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": _is_trained, "checkpoint": CKPT_PATH}


@app.get("/metrics")
def metrics():
    """Serves the last training run's evaluation report, if train.py has been run."""
    metrics_path = os.path.join(os.path.dirname(CKPT_PATH), "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    baseline_path = os.path.join(os.path.dirname(CKPT_PATH), "baseline_metrics.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            return {"note": "GNN not yet trained -- showing feature-only sanity baseline. "
                             "Run `python train.py` to populate metrics.json.", **json.load(f)}
    return {"note": "No metrics available yet. Run `python train.py`."}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if len(req.nodes) == 0 or req.target_node_index >= len(req.nodes):
        return ScoreResponse(
            tx_id=req.tx_id, risk_score=0, flagged=False, calibrated_probability=0.0,
            audit_hash="0x0", top_contributing_neighbors=[], model_trained=_is_trained,
            explanation_note="empty or invalid subgraph",
        )

    x = torch.tensor([n.features for n in req.nodes], dtype=torch.float32)
    import numpy as np
    edge_index = np.array([[e.src for e in req.edges], [e.dst for e in req.edges]], dtype=np.int64) \
        if req.edges else np.zeros((2, 0), dtype=np.int64)
    adj = edges_to_adj_mask(edge_index, x.size(0))

    with torch.no_grad():
        logits, attn = _model(x, adj, return_attention=True)
        calibrated_prob = torch.sigmoid(logits[req.target_node_index] / _temperature).item()

    risk_score = int(round(calibrated_prob * 1000))
    flagged = risk_score >= RISK_THRESHOLD

    # explainability: which neighbors' attention weight into the target node was highest
    layer2_attn = attn["layer2"][req.target_node_index].mean(dim=-1)  # avg over heads -> [N]
    top_k = min(3, x.size(0))
    top_idx = torch.topk(layer2_attn, top_k).indices.tolist()
    top_neighbors = [
        {"node_index": i, "attention_weight": round(float(layer2_attn[i]), 4)}
        for i in top_idx if float(layer2_attn[i]) > 0
    ]

    explanation = {
        "tx_id": req.tx_id, "calibrated_probability": calibrated_prob,
        "top_contributing_neighbors": top_neighbors, "model_head": _model.head_type,
    }
    audit_hash = "0x" + hashlib.sha256(json.dumps(explanation, sort_keys=True).encode()).hexdigest()

    return ScoreResponse(
        tx_id=req.tx_id, risk_score=risk_score, flagged=flagged,
        calibrated_probability=round(calibrated_prob, 4), audit_hash=audit_hash,
        top_contributing_neighbors=top_neighbors, model_trained=_is_trained,
        explanation_note=None if _is_trained else "WARNING: serving an untrained model -- run train.py first",
    )
