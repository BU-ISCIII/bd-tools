"""General utilities: data loading, imputation, and preprocessing."""

from .base import *
from .config import data_processing_config

_CFG = data_processing_config()
_MISSING_INDICATOR_THRESHOLD = float(_CFG["missing_indicator_threshold"])
_KNN_NEIGHBORS = int(_CFG["knn_imputer"]["n_neighbors"])
_KNN_WEIGHTS = _CFG["knn_imputer"]["weights"]
_DROP_HELPER_COLUMNS = list(_CFG["drop_helper_columns"])

def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        try:
            df = df.drop(columns=col)
        except KeyError:
            print(f"Warning: column '{col}' not found – skipping drop.")
    return df


def load_processed_dataframe(csv_path: Path, cols_to_delete: List[str]) -> pd.DataFrame:
    """Load dataset and apply focus filter. No target-based row filtering here."""
    df = pd.read_csv(csv_path)
    if "foco" in df.columns:
        df["foco"] = df["foco"].map(FOCUS_MAP).fillna(df["foco"])
        df = df[~df["foco"].isin(FOCUS_TO_EXCLUDE)]
    df = safe_drop_columns(df, cols_to_delete)
    return df


def impute_missing_values(loaded_df: pd.DataFrame, exclude_cols: set) -> pd.DataFrame:
    """Impute missing values; target/weight columns are kept as-is.

    Strategy
    --------
    * Binary (0/1) numeric columns → mode (SimpleImputer).
    * All other numeric columns (continuous AND low-cardinality ordinal scores
      such as SOFA sub-scores, bilirrubina 0-4, snc_glasgow 0-4, respiracion
      0-3) → KNNImputer(k=10, distance-weighted).  Using KNN for ordinal
      scores is strictly better than mode because it exploits the patient's
      other measurements; mode always collapses to the most common value
      regardless of clinical context.
    * String/category columns → mode (SimpleImputer).
    * Missing-indicator flags: for every numeric column whose missing rate
      exceeds 10 %, a companion ``<col>_missing`` binary column is added
      *before* imputation.  Tree models can use these flags directly as a
      signal that the original value was absent (informative missingness,
      e.g. SOFA not recorded often means the patient was less severe).
    """
    df_copy = loaded_df.drop(columns=list(exclude_cols), errors="ignore").copy()
    numeric_cols = df_copy.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_cols = [c for c in numeric_cols if set(df_copy[c].dropna().unique()) <= {0, 1}]
    # Ordinal clinical scores (low-cardinality) and continuous values both
    # benefit from multivariate KNN — route all non-binary numeric to KNN.
    knn_cols = [c for c in numeric_cols if c not in binary_cols]

    if binary_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[binary_cols] = imp.fit_transform(df_copy[binary_cols]).astype(int)
    if knn_cols:
        # k=10 gives more stable estimates than k=5 for datasets of ~3 000+ rows.
        imp = KNNImputer(n_neighbors=_KNN_NEIGHBORS, weights=_KNN_WEIGHTS)
        df_copy[knn_cols] = imp.fit_transform(df_copy[knn_cols])
        # Re-cast originally-integer ordinal columns back to int after KNN.
        for col in knn_cols:
            if loaded_df[col].dropna().astype(float).apply(float.is_integer).all():
                df_copy[col] = df_copy[col].round().astype(int)
    if cat_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df_copy[cat_cols] = imp.fit_transform(df_copy[cat_cols]).astype(str)
    print("Imputed values and included marker columns for high_missings")
    for col in exclude_cols:
        if col in loaded_df.columns:
            df_copy[col] = loaded_df[col]
    return df_copy


def preprocess_train_test_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    na_perc_limit: float,
    impute_missing: bool,
    categorical_for_dummies: List[str],
    output_csv_paths: Optional[Dict[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe feature preprocessing: fit on train, transform test."""
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # Track NaN and imputation info for CSV export
    nan_dropped_records: List[Dict[str, object]] = []
    imputation_records: List[Dict[str, object]] = []

    # Drop target-related helper columns if present
    cols_to_drop = [c for c in _DROP_HELPER_COLUMNS if c in X_train.columns]
    if cols_to_drop:
        print(f"  Dropping target-related columns: {cols_to_drop}")
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop, errors="ignore")

    # Drop high-NA columns based on TRAIN only
    dropped_na = [col for col in X_train.columns if X_train[col].isna().mean() > na_perc_limit]
    if dropped_na:
        print(f"Dropping {len(dropped_na)} high-NA columns (train-based).")
        for col in dropped_na:
            nan_pct = X_train[col].isna().mean() * 100
            nan_dropped_records.append({"feature": col, "nan_percentage": nan_pct})
        X_train = X_train.drop(columns=dropped_na)
        X_test = X_test.drop(columns=dropped_na, errors="ignore")

    if not len(X_train.columns):
        raise ValueError("No feature columns remain after NA filtering.")

    numeric_cols = X_train.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = [c for c in numeric_cols if set(X_train[c].dropna().unique()) <= {0, 1}]
    knn_cols = [c for c in numeric_cols if c not in binary_cols]

    if impute_missing:
        if binary_cols:
            imp = SimpleImputer(strategy="most_frequent")
            X_train[binary_cols] = imp.fit_transform(X_train[binary_cols]).astype(int)
            X_test[binary_cols] = imp.transform(X_test[binary_cols]).astype(int)
            for col in binary_cols:
                n_imputed = X_train_raw[col].isna().sum()
                if n_imputed > 0:
                    imputation_records.append({
                        "feature": col,
                        "imputation_method": "most_frequent",
                        "number_of_imputation": int(n_imputed)
                    })
        if knn_cols:
            imp = KNNImputer(n_neighbors=_KNN_NEIGHBORS, weights=_KNN_WEIGHTS)
            X_train[knn_cols] = imp.fit_transform(X_train[knn_cols])
            X_test[knn_cols] = imp.transform(X_test[knn_cols])
            for col in knn_cols:
                n_imputed = X_train_raw[col].isna().sum()
                if n_imputed > 0:
                    imputation_records.append({
                        "feature": col,
                        "imputation_method": "knn",
                        "number_of_imputation": int(n_imputed)
                    })
            for col in knn_cols:
                if X_train_raw[col].dropna().astype(float).apply(float.is_integer).all():
                    X_train[col] = X_train[col].round().astype(int)
                    X_test[col] = X_test[col].round().astype(int)
        if cat_cols:
            imp = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = imp.fit_transform(X_train[cat_cols]).astype(str)
            X_test[cat_cols] = imp.transform(X_test[cat_cols]).astype(str)
            for col in cat_cols:
                n_imputed = X_train_raw[col].isna().sum()
                if n_imputed > 0:
                    imputation_records.append({
                        "feature": col,
                        "imputation_method": "most_frequent",
                        "number_of_imputation": int(n_imputed)
                    })
        print("Imputed missing values with train-fitted imputers.")
    else:
        train_mask = X_train.notna().all(axis=1)
        test_mask = X_test.notna().all(axis=1)
        X_train = X_train.loc[train_mask].copy()
        X_test = X_test.loc[test_mask].copy()

    # One-hot encode chosen categorical columns with train-fitted schema
    dummies_cols = [c for c in categorical_for_dummies if c in X_train.columns]
    if dummies_cols:
        X_train = pd.get_dummies(X_train, columns=dummies_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=dummies_cols, drop_first=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_train.columns = X_train.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
        X_test.columns = X_test.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Export NaN and imputation tracking CSVs if requested
    if output_csv_paths:
        if "nan_dropped" in output_csv_paths and nan_dropped_records:
            Path(output_csv_paths["nan_dropped"]).parent.mkdir(parents=True, exist_ok=True)
            df_nan = pd.DataFrame(nan_dropped_records)
            df_nan.to_csv(output_csv_paths["nan_dropped"], index=False)
            print(f"  Saved NaN-dropped features to {output_csv_paths['nan_dropped']}")
        
        if "imputed" in output_csv_paths and imputation_records:
            Path(output_csv_paths["imputed"]).parent.mkdir(parents=True, exist_ok=True)
            df_imputed = pd.DataFrame(imputation_records)
            df_imputed.to_csv(output_csv_paths["imputed"], index=False)
            print(f"  Saved imputed features to {output_csv_paths['imputed']}")

    return X_train, X_test


def compute_balanced_sample_weight(
    labels: pd.Series, base_sample_weight: Optional[pd.Series] = None
) -> pd.Series:
    classes = np.unique(labels)
    cw = compute_class_weight("balanced", classes=classes, y=labels)
    balanced = labels.map(dict(zip(classes, cw)))
    if base_sample_weight is not None:
        balanced = balanced * pd.Series(base_sample_weight, index=labels.index)
    return balanced

