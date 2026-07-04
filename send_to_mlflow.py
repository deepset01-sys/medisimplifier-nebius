"""
send_to_mlflow.py — Log MediSimplifier evaluation results to Nebius Managed MLflow.

Usage:
  1. Create a Managed MLflow cluster in Nebius Console (AI Services → MLflow)
  2. Set environment variables:
       MLFLOW_TRACKING_URI=<your-cluster-public-tracking-uri>
       MLFLOW_TRACKING_USERNAME=admin
       MLFLOW_TRACKING_PASSWORD=<your-cluster-password>
  3. Run: python send_to_mlflow.py

All 4 evaluation runs (3 models × seed=42, plus seed=2 validation) will be
logged to the 'medisimplifier' experiment with full params and metrics.
"""

import os
import json
import mlflow
from pathlib import Path

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not TRACKING_URI:
    raise EnvironmentError(
        "MLFLOW_TRACKING_URI not set. "
        "Create a Nebius Managed MLflow cluster and set the public tracking URI."
    )

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("medisimplifier")

EVIDENCE_DIR = Path("results/nebius_evidence")

runs = [
    {
        "name": "openbio-seed42",
        "file": "results_openbio.json",
        "params": {"model": "openbio", "rank": 32, "modules": "all_attn",
                   "epochs": 3, "data_size": 7999, "seed": 42, "use_rslora": True}
    },
    {
        "name": "mistral-seed42",
        "file": "results_mistral.json",
        "params": {"model": "mistral", "rank": 32, "modules": "all_attn",
                   "epochs": 3, "data_size": 7999, "seed": 42, "use_rslora": True}
    },
    {
        "name": "biomistral-seed42",
        "file": "results_biomistral.json",
        "params": {"model": "biomistral", "rank": 32, "modules": "all_attn",
                   "epochs": 3, "data_size": 7999, "seed": 42, "use_rslora": True}
    },
    {
        "name": "openbio-seed2",
        "file": "eval_seed2.json",
        "params": {"model": "openbio", "rank": 32, "modules": "all_attn",
                   "epochs": 3, "data_size": 7999, "seed": 2, "use_rslora": True}
    },
]

for run in runs:
    data = json.loads((EVIDENCE_DIR / run["file"]).read_text())
    with mlflow.start_run(run_name=run["name"]):
        mlflow.log_params(run["params"])
        mlflow.log_metrics({
            "rouge_l":   data["rouge_l"],
            "bertscore": data["bertscore"],
            "sari":      data["sari"],
            "fk_grade":  data["fk_grade"],
        })
    print(f"Logged: {run['name']}")

print("Done! Check your MLflow experiment at:", TRACKING_URI)
