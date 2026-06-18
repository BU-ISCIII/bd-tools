#!/usr/bin/env python3
"""
Three-level independent hierarchical classifier for sepsis outcomes.

Pipeline
--------
  Level 1 – sepsis (binary)
    Trained on original features only.

  Level 2 – resultado_hemo_grouped (multiclass)
    Trained on original features only (no Level-1 OOF cascade).
    Very rare blood-culture classes are dropped automatically.
    SHAP-RFECV is applied to select the best feature subset.

  Level 3 – resistente_cefalosporina (binary)
    Trained on original features only (no Level-1 or Level-2 OOF cascade).
    Row scope: only patients with a positive blood culture
               (resultado_hemo_grouped != "NEGATIVE").
    SHAP-RFECV is applied to select the best feature subset.

Each level is trained independently – OOF predictions from one level are NOT
injected as features into the next level.  This allows a clean comparison
against the cascade variant.

SHAP-RFECV is run at every level; the selected feature list and full score
history are saved in each level's summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import shap
from .config import data_processing_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
JOB_ID = os.environ.get("SLURM_JOB_ID", 1)
TODAY = datetime.today().strftime("%Y%m%d%H%M%S")

_DATA_CFG = data_processing_config()
FOCUS_MAP = _DATA_CFG["focus_map"]

# Metadata columns to include in the output predictions file for analysis
METADATA_COLUMNS = _DATA_CFG["metadata_columns"]

# All potential target columns – none of these should appear as features
TARGET_REMOVE = _DATA_CFG["target_remove"]

DELETE_COLUMNS = _DATA_CFG["delete_columns"]

FOCUS_TO_EXCLUDE = set(_DATA_CFG["focus_to_exclude"])

GATE_RECALL = float(_DATA_CFG["gate_recall"])

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

