#!/usr/bin/env python3
# coding: utf-8
"""Two-stage BACTHECOM mortality modelling pipeline orchestration."""

from __future__ import annotations

import time
import traceback

import optuna
import pandas as pd
from warnings import filterwarnings

from .metrics import binary_metrics, save_predictions
from .oof import generate_oof_stage1_proba
from .optuna import optimise_and_fit_binary_model
from .utils import (
    compute_imbalance,
    load_data,
    prepare_directories,
    preprocess_train_test_features,
    read_drop_columns,
    save_config,
    save_json,
    setup_logging,
    train_test_split_data,
)

filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_training(args) -> None:
    run_dir, stage1_dir, stage2_dir = prepare_directories(args.results_dir, str(args.job_id), args.target1, args.target2)
    logger = setup_logging(run_dir, str(args.array_id))
    save_json(run_dir / "processing" / "args.json", {k: str(v) for k, v in vars(args).items()})

    try:
        if str(args.array_id) != "0":
            time.sleep(15)

        drop_cols = read_drop_columns(args.utils_json)
        if args.target2 == "mortalidad_14_dias":
            drop_cols.append("mortalidad_30_dias")
        elif args.target2 == "mortalidad_30_dias":
            drop_cols.append("mortalidad_14_dias")

        drop_cols = list(dict.fromkeys(drop_cols))
        pd.Series(drop_cols, name="requested_drop_column").to_csv(run_dir / "processing" / "features" / "requested_drop_columns.csv", index=False)

        X, y1, y2, load_info = load_data(args.data, drop_cols, args.target1, args.target2, logger=logger)
        save_json(run_dir / "processing" / "load_info.json", load_info)

        X_train_raw, X_test_raw, y_train1, y_test1, y_train2, y_test2 = train_test_split_data(X=X, y1=y1, y2=y2, test_size=args.testing_split_size, seed=args.seed)

        split_info = {
            "X_train_raw_shape": list(X_train_raw.shape),
            "X_test_raw_shape": list(X_test_raw.shape),
            "stage1_train_counts": y_train1.value_counts().to_dict(),
            "stage1_test_counts": y_test1.value_counts().to_dict(),
            "stage2_train_counts_before_stage_filter": y_train2.value_counts().to_dict(),
            "stage2_test_counts_before_stage_filter": y_test2.value_counts().to_dict(),
        }
        save_json(run_dir / "processing" / "split_info.json", split_info)

        X_train, X_test, preprocessing_info = preprocess_train_test_features(
            X_train_raw,
            X_test_raw,
            na_perc_limit=args.na_perc_limit,
            iqr_multiplier=args.iqr_multiplier,
            max_corr=args.max_corr,
            run_dir=run_dir,
            logger=logger,
        )

        imbalance_ratio1 = compute_imbalance(y_train1)
        stage1_model, stage1_test_proba, metrics1, best_params1 = optimise_and_fit_binary_model(
            X_train=X_train,
            y_train=y_train1,
            X_test=X_test,
            y_test=y_test1,
            target=args.target1,
            algorithm=args.algorithm,
            n_splits=args.training_n_splits,
            n_trials=args.n_trials,
            storage=args.storage,
            imbalance_ratio=imbalance_ratio1,
            n_cpus=args.n_cpus,
            seed=args.seed,
            array_id=str(args.array_id),
            job_id=str(args.job_id),
            run_dir=stage1_dir,
            study_name=f"{args.study_name}_{args.algorithm}_stage1",
            logger=logger,
        )

        stage1_train_oof_proba = generate_oof_stage1_proba(
            X=X_train,
            y=y_train1,
            algorithm=args.algorithm,
            params=best_params1,
            n_splits=args.training_n_splits,
            seed=args.seed,
        )
        stage1_test_proba_series = pd.Series(stage1_test_proba, index=X_test.index, name="stage1_proba")

        mask_train2 = y_train1 == 1
        mask_test2_true_positive = y_test1 == 1
        if int(mask_train2.sum()) < 10:
            raise ValueError(f"Too few Stage-1-positive training rows for Stage 2: {int(mask_train2.sum())}")

        X_train2 = X_train.loc[mask_train2].copy()
        X_test2 = X_test.loc[mask_test2_true_positive].copy()
        y_train2_stage = y_train2.loc[mask_train2].copy()
        y_test2_stage = y_test2.loc[mask_test2_true_positive].copy()
        X_train2["stage1_proba"] = stage1_train_oof_proba.loc[X_train2.index].values
        X_test2["stage1_proba"] = stage1_test_proba_series.loc[X_test2.index].values

        imbalance_ratio2 = compute_imbalance(y_train2_stage)
        stage2_model, _, metrics2, _ = optimise_and_fit_binary_model(
            X_train=X_train2,
            y_train=y_train2_stage,
            X_test=X_test2,
            y_test=y_test2_stage,
            target=args.target2,
            algorithm=args.algorithm,
            n_splits=args.training_n_splits,
            n_trials=args.n_trials,
            storage=args.storage,
            imbalance_ratio=imbalance_ratio2,
            n_cpus=args.n_cpus,
            seed=args.seed,
            array_id=str(args.array_id),
            job_id=str(args.job_id),
            run_dir=stage2_dir,
            study_name=f"{args.study_name}_{args.algorithm}_stage2",
            logger=logger,
            early_stopping_rounds=args.early_stopping_rounds,
            fast_search=args.fast_search,
        )

        cascade_metrics = None
        if args.stage2_use_predicted_positive_test:
            mask_test2_predicted_positive = pd.Series(stage1_test_proba >= 0.5, index=X_test.index)
            X_test2_cascade = X_test.loc[mask_test2_predicted_positive].copy()
            y_test2_cascade = y_test2.loc[mask_test2_predicted_positive].copy()
            if len(X_test2_cascade) > 0 and y_test2_cascade.nunique() > 1:
                X_test2_cascade["stage1_proba"] = stage1_test_proba_series.loc[X_test2_cascade.index].values
                cascade_proba = stage2_model.predict_proba(X_test2_cascade)[:, 1]
                cascade_metrics = binary_metrics(y_test2_cascade, cascade_proba, threshold=0.5)
                save_json(stage2_dir / f"metrics_cascade_{args.target2}_array_{args.array_id}.json", cascade_metrics)
                save_predictions(stage2_dir / "predictions" / f"predictions_cascade_{args.target2}_array_{args.array_id}.csv", X_test2_cascade.index, y_test2_cascade, cascade_proba)

        aggregate_summary = {
            "workflow": "two_stage_mortality",
            "run_dir": str(run_dir),
            "stage1": metrics1,
            "stage2_true_mortality_subset": metrics2,
            "stage2_deployed_cascade_subset": cascade_metrics,
        }
        save_json(run_dir / f"aggregate_summary_{args.target2}_array_{args.array_id}.json", aggregate_summary)
        save_config(args, drop_cols, preprocessing_info, run_dir)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        save_json(
            run_dir / "logs" / f"error_array_{args.array_id}.json",
            {"error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()},
        )
        raise
