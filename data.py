"""
Synthetic transaction-graph dataset generator.

Real IEEE-CIS / UPI transaction logs aren't bundled here (privacy + size), so this
generates a structurally realistic stand-in: accounts are graph nodes, transactions are
edges, and fraud is injected as a *pattern* (fast fan-out from a newly created or recently
compromised account, at odd hours, to previously-unconnected payees) rather than a single
threshold on one feature -- so the model actually has to learn relational structure, not
just "amount > X". Swap `build_dataset()` for a loader over your real labeled data and
every downstream file (model.py, train.py, app.py) is unaffected as long as the feature
schema (FEATURE_NAMES) is preserved.
"""
import numpy as np

FEATURE_NAMES = [
    "amount_norm", "hour_sin", "hour_cos", "account_age_days_norm",
    "txn_velocity_1h", "txn_velocity_24h", "avg_amount_ratio",
    "is_new_payee", "device_change", "geo_distance_norm",
    "degree_norm", "in_out_ratio", "prior_fraud_flag_neighbor",
]
NODE_DIM = len(FEATURE_NAMES)


def _make_graph(n_accounts: int, fraud_ring_frac: float, rng: np.random.Generator):
    """One synthetic transaction subgraph: returns (features, edge_index, labels)."""
    features = np.zeros((n_accounts, NODE_DIM), dtype=np.float32)
    labels = np.zeros(n_accounts, dtype=np.int64)

    n_fraud = max(1, int(n_accounts * fraud_ring_frac))
    fraud_idx = rng.choice(n_accounts, size=n_fraud, replace=False)
    labels[fraud_idx] = 1

    # A slice of "hard negative" legit accounts deliberately behaves like fraud on 1-2
    # surface features (e.g. a legit new-payee gift, a genuine travel-driven geo jump) --
    # without this, the label is trivially separable on single features and the eval
    # numbers are meaningless. Real fraud detection lives or dies on cases like these.
    hard_negative = (labels == 0) & (rng.random(n_accounts) < 0.18)
    label_noise = rng.random(n_accounts) < 0.03  # 3% mislabeled, as in real review queues
    labels_noisy = labels.copy()
    labels_noisy[label_noise] = 1 - labels_noisy[label_noise]

    for i in range(n_accounts):
        is_fraud = labels[i] == 1
        hard_neg = hard_negative[i]
        amount = rng.lognormal(mean=6.1 if is_fraud else 5.6, sigma=1.2 if is_fraud else 0.9)
        hour = rng.normal(6 if is_fraud else 14, 5 if (is_fraud or hard_neg) else 4) % 24
        account_age = rng.exponential(45 if is_fraud else (60 if hard_neg else 400))
        velocity_1h = rng.poisson(3.0 if is_fraud else (1.5 if hard_neg else 0.4))
        velocity_24h = velocity_1h + rng.poisson(6 if is_fraud else 2)
        avg_ratio = amount / (rng.lognormal(5.3, 0.7) + 1e-6)
        is_new_payee = rng.random() < (0.65 if is_fraud else (0.55 if hard_neg else 0.15))
        device_change = rng.random() < (0.4 if is_fraud else (0.25 if hard_neg else 0.05))
        geo_dist = rng.exponential(500 if is_fraud else (300 if hard_neg else 25))
        degree = rng.poisson(6 if is_fraud else 3)
        in_out_ratio = rng.uniform(0.05, 0.55) if is_fraud else rng.uniform(0.25, 1.0)

        features[i] = [
            np.log1p(amount) / 12.0,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            min(account_age / 500.0, 1.0),
            min(velocity_1h / 10.0, 1.0), min(velocity_24h / 30.0, 1.0),
            min(avg_ratio / 5.0, 1.0),
            float(is_new_payee), float(device_change),
            min(geo_dist / 2000.0, 1.0),
            min(degree / 20.0, 1.0), in_out_ratio,
            0.0,  # prior_fraud_flag_neighbor filled in below
        ]

    # Build edges: fraud accounts fan out to random targets (ring-like); legit accounts
    # form small clustered friend/merchant groups. This creates the *relational* signal.
    edges = []
    for i in fraud_idx:
        fanout = rng.integers(3, 9)
        targets = rng.choice(n_accounts, size=min(fanout, n_accounts - 1), replace=False)
        for t in targets:
            if t != i:
                edges.append((i, t))
                edges.append((t, i))
    n_clusters = max(1, n_accounts // 6)
    cluster_of = rng.integers(0, n_clusters, size=n_accounts)
    for c in range(n_clusters):
        members = np.where(cluster_of == c)[0]
        for a in range(len(members)):
            for b in range(a + 1, min(a + 3, len(members))):
                edges.append((members[a], members[b]))
                edges.append((members[b], members[a]))
    if not edges:
        edges = [(0, 0)]
    edge_index = np.array(edges, dtype=np.int64).T

    # Propagate a soft "neighbor was fraud" signal one hop (relational feature).
    neighbor_fraud = np.zeros(n_accounts, dtype=np.float32)
    for src, dst in edges:
        neighbor_fraud[dst] += labels[src]
    # slight noise on the propagated signal too -- it's inferred from partially-noisy
    # upstream labels in any real pipeline, not an oracle feature.
    features[:, -1] = np.minimum(neighbor_fraud / 3.0, 1.0) + rng.normal(0, 0.05, n_accounts)
    features[:, -1] = np.clip(features[:, -1], 0, 1)

    return features.astype(np.float32), edge_index, labels_noisy


def build_dataset(n_graphs=400, n_accounts_range=(20, 60), fraud_ring_frac=0.12, seed=42):
    """Returns a list of (features, edge_index, labels) synthetic graphs."""
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        n = int(rng.integers(*n_accounts_range))
        graphs.append(_make_graph(n, fraud_ring_frac, rng))
    return graphs


def train_val_test_split(graphs, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    n = len(graphs)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    return (
        [graphs[i] for i in idx[:train_end]],
        [graphs[i] for i in idx[train_end:val_end]],
        [graphs[i] for i in idx[val_end:]],
    )
