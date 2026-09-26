# SentinelPay
**Quantum-Attentive Fraud Shield for Real-Time, Blockchain-Settled Payments in India**

Built for **CHL-7007 — India Blockchain Forum** (Problem Statement 1: Real-Time Payments)

Every transaction is scored in real time by a Graph Attention Transformer before it
settles. Low-risk payments clear instantly through a Drunix/EVM escrow contract;
high-risk ones are auto-held for review, with every decision hashed on-chain for an
auditable trail.

## What's actually been verified vs. what needs a local run
Being upfront about this because "it compiles" and "it works" aren't the same claim:

| Component | Status |
|---|---|
| `PaymentEscrow.sol` | **Compiles clean with real solc 0.8.20** (zero errors/warnings) |
| Backend orchestrator (`/pay`, `/transactions`, `/release`, `/reverse`, `/stats`, WebSocket) | **Smoke-tested end-to-end in this sandbox** — pay → score → hold → release → settle, all verified working |
| Synthetic fraud data pipeline (`data.py` + `metrics.py`) | **Run in-sandbox**: produces a non-trivial (not 100%-separable) labeled dataset; a GradientBoosting sanity baseline scores ROC-AUC 0.898 / F1 0.83 / precision 0.91 on it — see `fraud-engine/checkpoints/baseline_metrics.json` |
| `SentinelFraudNet` (GATv2 + Transformer + optional quantum head) | Code complete, syntax-checked, **not yet trained** — this sandbox has no disk headroom to install PyTorch. Run `python train.py` locally (see below) to produce the real checkpoint + metrics |
| Hardhat contract tests (`test/PaymentEscrow.test.js`) | Written and ready — `npx hardhat test` needs network access to `binaries.soliditylang.org`, which this sandbox blocks. Will run in a normal dev environment |
| Frontend dashboard | Built (React via CDN, no build step needed) — open directly against a running backend |

**Before you demo this, run `python train.py` in `fraud-engine/`** so `/score` is serving
a real trained model instead of an untrained one (the API clearly flags untrained
responses so this can't be missed by accident).

## Repo layout
```
contracts/        PaymentEscrow.sol + Hardhat tests, targets Drunix/EVM
fraud-engine/      GATv2 + Transformer fraud model: data.py, model.py, losses.py,
                   metrics.py, train.py, app.py (FastAPI serving)
backend/           Payment orchestrator: db.py (SQLite/Postgres), chain.py (web3 bridge,
                   auto-simulates if no RPC configured), app.py (FastAPI + WebSocket)
frontend/          Single-file React dashboard (live feed, held-transaction review)
docs/architecture.md
docker-compose.yml
```

## Quickstart (local, no Docker)
```bash
# 1. Fraud engine — train first, then serve
cd fraud-engine
pip install -r requirements.txt        # see note below if this pulls a CUDA build you don't want
python train.py --epochs 60            # ~a few minutes on CPU; writes checkpoints/sentinelfraud.pt + metrics.json
uvicorn app:app --reload --port 8001

# 2. Backend orchestrator (new terminal)
cd ../backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# No RPC_URL/CONTRACT_ADDRESS/ORACLE_PRIVATE_KEY set -> runs in SIMULATED chain mode,
# logging exactly what it would submit on-chain. Set those three env vars to go live.

# 3. Frontend (new terminal)
cd ../frontend
python3 -m http.server 3000
# open http://localhost:3000 — dashboard talks to http://localhost:8000 by default;
# override with window.SENTINELPAY_API before the script tag if needed.

# 4. Contracts
cd ../contracts
npm install
npx hardhat compile && npx hardhat test
```

Or `docker compose up --build` once you've trained a checkpoint (the fraud-engine
container mounts `fraud-engine/checkpoints`, so train locally first, then containerize).

### If `pip install torch` pulls a huge CUDA build
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```
then `pip install -r requirements.txt` again for the rest.

## Pushing this to your own GitHub repo
```bash
git init && git add . && git commit -m "SentinelPay: hackathon submission"
git branch -M main
git remote add origin https://github.com/<your-username>/sentinelpay.git
git push -u origin main
```
Paste that URL into the submission form's "GitHub Repository URL" field.

## Design notes
- **Why dense-adjacency attention instead of torch-geometric**: real-time scoring runs
  against a small local neighborhood (tens of accounts), not the whole ledger, and it
  avoids a fiddly torch-geometric install for anyone judging/running this live.
- **Why focal loss**: fraud is a small minority class; plain BCE lets it get ignored.
- **Why temperature scaling**: the escrow contract's `RISK_THRESHOLD = 700` is only
  meaningful if the model's output is a calibrated probability, not a raw logit score.
- **Why a SIMULATED chain mode by default**: lets the full pay → score → hold → release
  flow be demoed without needing a live Drunix RPC endpoint in front of a judge.
