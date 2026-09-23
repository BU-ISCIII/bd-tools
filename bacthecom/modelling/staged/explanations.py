"""Native tree SHAP values in raw log-odds space, with reproducible rankings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export_explanations(pipeline, X, algorithm, settings, directory, seed, population):
    if not settings.get('enabled', True):
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(X) == 0:
        raise ValueError('Cannot explain an empty population')
    sample = X.sample(n=min(len(X), settings.get('max_samples', 1000)), random_state=seed).sort_index()
    features = pipeline['postprocessing'].transform(sample)
    model = pipeline['model']
    if algorithm == 'catb':
        from catboost import Pool
        contributions = model.get_feature_importance(Pool(features), type='ShapValues', thread_count=1)
        raw = model.predict(features, prediction_type='RawFormulaVal')
    elif algorithm == 'lgbm':
        contributions = model.predict(features, pred_contrib=True)
        raw = model.predict(features, raw_score=True)
    elif algorithm == 'xgb':
        from xgboost import DMatrix
        matrix = DMatrix(features)
        contributions = model.get_booster().predict(matrix, pred_contribs=True)
        raw = model.get_booster().predict(matrix, output_margin=True)
    else:
        raise ValueError(f'Unsupported SHAP algorithm: {algorithm}')
    contributions = np.asarray(contributions)
    if contributions.shape != (len(features), features.shape[1] + 1):
        raise ValueError(f'Unexpected binary SHAP shape: {contributions.shape}')
    values, base = contributions[:, :-1], contributions[:, -1]
    error = float(np.max(np.abs(contributions.sum(axis=1) - raw)))
    if not np.allclose(contributions.sum(axis=1), raw, rtol=1e-4, atol=1e-5):
        raise ValueError(f'SHAP additivity check failed: max error={error}')
    pd.DataFrame(values, index=sample.index, columns=features.columns).to_csv(directory / 'shap_values.csv', index_label='row_index')
    features.to_csv(directory / 'explained_features.csv', index_label='row_index')
    pd.DataFrame({'base_value': base, 'raw_margin': raw, 'probability': model.predict_proba(features)[:, 1]}, index=sample.index).to_csv(directory / 'shap_base_values.csv', index_label='row_index')
    ranked = pd.DataFrame({'feature': features.columns, 'mean_abs_shap': np.abs(values).mean(axis=0),
                           'mean_signed_shap': values.mean(axis=0),
                           'model_importance': model.feature_importances_}).sort_values('mean_abs_shap', ascending=False, kind='stable')
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked.to_csv(directory / 'ranked_variables.csv', index=False)
    native = ranked[['feature', 'model_importance']].sort_values('model_importance', ascending=False, kind='stable')
    native.insert(0, 'rank', np.arange(1, len(native) + 1))
    native.to_csv(directory / 'model_importance_ranked.csv', index=False)
    # Missingness indicators are separate model inputs; this additional table
    # aggregates the magnitudes for each source variable and its indicator.
    grouped = ranked.assign(variable=ranked.feature.str.replace(r'__missing_indicator$', '', regex=True)).groupby('variable')['mean_abs_shap'].sum().sort_values(ascending=False).reset_index()
    grouped.insert(0, 'rank', np.arange(1, len(grouped) + 1))
    grouped.to_csv(directory / 'ranked_source_variables.csv', index=False)
    top = ranked.head(settings.get('top_features', 30)).iloc[::-1]
    height = max(4, len(top) * .28)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(top.feature, top.mean_abs_shap)
    ax.set_xlabel('Mean absolute SHAP value (log-odds)')
    ax.set_title(population)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'shap_importance.{ext}', bbox_inches='tight', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, height))
    rng = np.random.default_rng(seed)
    for row, col in enumerate(top.feature):
        i = features.columns.get_loc(col)
        x = features[col].to_numpy()
        span = np.ptp(x)
        colors = (x - x.min()) / span if span else np.full(len(x), .5)
        ax.scatter(values[:, i], row + rng.uniform(-.24, .24, len(x)), c=colors, cmap='coolwarm', vmin=0, vmax=1, s=9, alpha=.65, rasterized=True)
    ax.set_yticks(range(len(top)), top.feature)
    ax.axvline(0, color='grey', linewidth=.6)
    ax.set_xlabel('SHAP value (log-odds); blue = low feature value, red = high')
    ax.set_title(population)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'shap_summary.{ext}', bbox_inches='tight', dpi=180)
    plt.close(fig)
    metadata = {'population': population, 'population_rows': len(X), 'explained_rows': len(sample),
                'seed': seed, 'output_space': 'raw_log_odds', 'algorithm': algorithm,
                'max_additivity_error': error, 'scope': 'individual stage model, not combined cascade',
                'source_ranking': 'sum of mean absolute contributions of source feature and missingness indicator',
                'native_importance_type': {'catb': 'PredictionValuesChange', 'lgbm': model.get_params().get('importance_type', 'split'), 'xgb': model.get_params().get('importance_type') or 'gain'}[algorithm]}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
