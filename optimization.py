"""
File: optimization.py
Role: Bayesian Hyperparameter Tuning for XGBRanker via Optuna

Workflow:
    1. Load training cache (X, y, qids) built by XGboost.py
    2. Split into train / val / test  (70 / 15 / 15 by unique query ID)
    3. Optuna searches hyperparams on val  — train is used to fit each trial model
    4. Best params saved to best_params.json
    5. Final model (train + val) evaluated on test set — exactly once

Metrics:
    P@5  : fraction of top-5 results with normalized label > RELEVANCE_THRESHOLD
    MRR  : reciprocal rank of the first relevant result in top-5
    Objective: P@5 + MRR  (equal weight; adjust ALPHA/BETA below if needed)

Run from project root:
    python src/optimization.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from XGboost import (
    CACHE_PATH,
    BEST_PARAMS_PATH,
    RANDOM_SEED,
    FEATURE_NAMES,
    three_way_split,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIALS           = 50
RELEVANCE_THRESHOLD = 0.5   # normalized label > 0.5 → relevant
ALPHA = 1.0                 # weight for P@5
BETA  = 1.0                 # weight for MRR


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def compute_metrics(model, X, y, qids):
    """
    Compute mean P@5 and mean MRR for a dataset split.
    Relevance oracle: normalized label y > RELEVANCE_THRESHOLD.
    """
    scores   = model.predict(X)
    p5_list  = []
    mrr_list = []

    for qid in np.unique(qids):
        mask   = qids == qid
        s      = scores[mask]
        labels = y[mask]

        top5 = np.argsort(s)[::-1][:5]
        rel  = labels[top5] > RELEVANCE_THRESHOLD

        p5_list.append(rel.sum() / len(rel))
        mrr = next((1.0 / (r + 1) for r, v in enumerate(rel) if v), 0.0)
        mrr_list.append(mrr)

    return float(np.mean(p5_list)), float(np.mean(mrr_list))


def sort_by_qid(X, y, qids):
    """XGBRanker requires all rows of the same query to be contiguous."""
    idx = np.argsort(qids, kind="stable")
    return X[idx], y[idx], qids[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── Load cache ───────────────────────────────────────────────────────────
    if not os.path.exists(CACHE_PATH):
        print(f"Cache not found at {CACHE_PATH}.")
        print("Run XGboost.py first to build the training data.")
        return

    print(f"Loading training cache from {CACHE_PATH}...")
    cache = np.load(CACHE_PATH)
    X     = cache["X"]
    y     = cache["y"]
    qids  = cache["qids"]
    print(f"  {X.shape[0]} samples · {X.shape[1]} features · {len(np.unique(qids))} queries")

    # ── 70/15/15 split by query ──────────────────────────────────────────────
    train_mask, val_mask, test_mask = three_way_split(qids)

    X_train, y_train, qids_train = sort_by_qid(X[train_mask], y[train_mask], qids[train_mask])
    X_val,   y_val,   qids_val   = sort_by_qid(X[val_mask],   y[val_mask],   qids[val_mask])
    X_test,  y_test,  qids_test  = sort_by_qid(X[test_mask],  y[test_mask],  qids[test_mask])

    print(f"  Train : {X_train.shape[0]} samples ({len(np.unique(qids_train))} queries)")
    print(f"  Val   : {X_val.shape[0]} samples ({len(np.unique(qids_val))} queries)")
    print(f"  Test  : {X_test.shape[0]} samples ({len(np.unique(qids_test))} queries)  ← held out")

    # ── Optuna objective ─────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int  ("n_estimators",     100, 500),
            "max_depth":        trial.suggest_int  ("max_depth",         2,   6),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.2,  log=True),
            "subsample":        trial.suggest_float("subsample",       0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.5, 1.0),
            "min_child_weight": trial.suggest_int  ("min_child_weight",  1,  20),
            "reg_alpha":        trial.suggest_float("reg_alpha",        0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",       0.0, 5.0),
        }

        model = xgb.XGBRanker(
            objective="rank:pairwise",
            random_state=RANDOM_SEED,
            verbosity=0,
            **params,
        )
        model.fit(X_train, y_train, qid=qids_train)

        p5, mrr = compute_metrics(model, X_val, y_val, qids_val)
        trial.set_user_attr("p5",  p5)
        trial.set_user_attr("mrr", mrr)
        return ALPHA * p5 + BETA * mrr

    # ── Run study ────────────────────────────────────────────────────────────
    print(f"\nRunning Optuna — {N_TRIALS} trials  (objective: {ALPHA}×P@5 + {BETA}×MRR on val)")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_trial  = study.best_trial
    p5_val      = best_trial.user_attrs["p5"]
    mrr_val     = best_trial.user_attrs["mrr"]

    print(f"\n{'='*60}")
    print(f"Optuna complete — best trial: #{best_trial.number}")
    print(f"  Val  P@5 : {p5_val:.4f}")
    print(f"  Val  MRR : {mrr_val:.4f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    # ── Save best params ─────────────────────────────────────────────────────
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved → {BEST_PARAMS_PATH}")

    # ── Test evaluation — run ONCE ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION  (run once, never used for tuning)")

    # Train final model on train + val combined
    X_tv, y_tv, qids_tv = sort_by_qid(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
        np.concatenate([qids_train, qids_val]),
    )

    final_model = xgb.XGBRanker(
        objective="rank:pairwise",
        random_state=RANDOM_SEED,
        verbosity=0,
        **best_params,
    )
    final_model.fit(X_tv, y_tv, qid=qids_tv)

    p5_test, mrr_test = compute_metrics(final_model, X_test, y_test, qids_test)

    print(f"  Test P@5 : {p5_test:.4f}")
    print(f"  Test MRR : {mrr_test:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
