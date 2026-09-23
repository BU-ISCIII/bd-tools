"""Merge four hospital XLSX exports using reviewed episode decisions.

Reference: playground/notebooks/cohort_preprocessing.ipynb. Review joins use
identity rather than row position; prior treatments stay attached to their own
culture, with one complete course per slot. No notebook execution is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = ROOT / 'ANALYSIS/01-PREPROCESS/00-data'
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'config/merging_cohort.yml'
EPISODE_KEYS = ['record_id', 'fecha_ingreso', 'fecha_hemocultivo']
TREATMENT_FIELDS = ['antimicrobiano_previo', 'atc_antib_prev',
                    'fecha_inicio_antib_previo', 'fecha_fin_antib_previo',
                    'dias_trat_antimicrobiano', 'via_administ_antib_prev']
TREATMENT_PATTERN = re.compile(r'^(' + '|'.join(TREATMENT_FIELDS) + r')_(\d+)$')


def canonical_name(value, aliases):
    name = re.sub(r'\s+', '_', str(value).strip().lower())
    name = aliases.get(name, name)
    name = re.sub(r'^(antimicrobiano_previo|atc_antib_prev|dias_trat_antimicrobiano|via_administ_antib_prev)_prev_(\d+)$', r'\1_\2', name)
    return re.sub(r'^fecha_antib_previo(?=_|$)', 'fecha_inicio_antib_previo', name)


def identifier(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def date_value(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float)):
        # Excel serial dates, not nanoseconds since 1970.
        if 1 <= value <= 100000:
            return (pd.Timestamp('1899-12-30') + pd.to_timedelta(value, unit='D')).normalize()
        return pd.NaT
    text = str(value).strip()
    parsed = pd.to_datetime(text, dayfirst=not bool(re.match(r'^\d{4}-', text)), errors='coerce')
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


def normalize_frame(frame, config, audit):
    columns = {}
    names = [canonical_name(c, config.get('column_aliases', {})) for c in frame.columns]
    seen = {}
    for original, roles in config.get('duplicate_column_roles', {}).items():
        if names.count(original) > 1 and names.count(original) != len(roles):
            raise ValueError(f'Unexpected number of {original} columns; review duplicate_column_roles')
    for pos, source in enumerate(frame.columns):
        name = names[pos]
        roles = config.get('duplicate_column_roles', {}).get(name)
        if roles and names.count(name) > 1:
            occurrence = seen.get(name, 0)
            seen[name] = occurrence + 1
            audit.setdefault('disambiguated_columns', []).append(
                {'source': str(source), 'position': pos, 'target': roles[occurrence]})
            name = roles[occurrence]
        values = frame.iloc[:, pos].map(lambda v: v.strip() if isinstance(v, str) else v)
        values = values.mask(values.isin(config.get('missing_tokens', [''])))
        if name.startswith('fecha_'):
            dates = values.map(date_value)
            audit.setdefault('invalid_dates', {})[name] = int((values.notna() & dates.isna()).sum())
            values = pd.to_datetime(dates)
        if name in {'record_id', 'id_hemocultivo'}:
            values = values.map(identifier)
        if name in columns:
            previous = columns[name]
            conflict = previous.notna() & values.notna() & ~previous.eq(values).fillna(False)
            if conflict.any():
                raise ValueError(f'Conflicting duplicate/alias column {name}: {int(conflict.sum())} rows')
            values = previous.combine_first(values)
            audit.setdefault('coalesced_columns', []).append(name)
        if name != str(source):
            audit.setdefault('renamed_columns', {})[str(source)] = name
        columns[name] = values
    return pd.DataFrame(columns, index=frame.index)


def read_sheet(path, sheet, config, audit):
    # Preserve duplicate names for conflict checks instead of pandas .1 mangling.
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object, engine='openpyxl')
    if raw.empty:
        raise ValueError(f'Empty sheet: {path.name}:{sheet}')
    header = raw.iloc[0]
    populated = header.notna()
    if raw.iloc[1:, (~populated).to_numpy()].notna().any().any():
        raise ValueError(f'Populated columns without headers: {path.name}:{sheet}')
    frame = raw.iloc[1:, populated.to_numpy()].copy()
    frame.columns = header[populated].tolist()
    frame = frame.dropna(how='all').reset_index(drop=True)
    audit['input_rows'] = len(frame)
    return normalize_frame(frame, config, audit)


def require_keys(frame, keys, context):
    absent = set(keys) - set(frame)
    if absent:
        raise ValueError(f'{context}: missing columns {sorted(absent)}')
    if frame[keys].isna().any().any():
        raise ValueError(f'{context}: missing/invalid episode keys {keys}')


def select_reviewed(frame, review, config, audit):
    keys = [*EPISODE_KEYS, 'microorganismo']
    if 'id_hemocultivo' in frame and 'id_hemocultivo' in review:
        keys.append('id_hemocultivo')
    require_keys(frame, EPISODE_KEYS, 'cohort')
    require_keys(review, EPISODE_KEYS, 'review')
    if not set([*keys, 'estado']).issubset(review):
        raise ValueError('Review lacks episode identity columns or estado')
    # Reviewed duplicates can have different decisions (keep one, remove its
    # duplicate). The review's explicit row ID disambiguates them, but only when
    # every patient/admission/culture/isolate identity also matches.
    frame = frame.copy()
    review = review.copy()
    if '_id_fila' in review:
        row_ids = pd.to_numeric(review['_id_fila'], errors='coerce')
        if row_ids.isna().any() or not row_ids.eq(row_ids.round()).all() or row_ids.duplicated().any():
            raise ValueError('Review _id_fila must contain unique integer source-row IDs')
        review['_source_row_id'] = row_ids.astype(int)
        if '_source_row_id' not in frame:
            frame['_source_row_id'] = range(len(frame))
        keys.append('_source_row_id')
        audit['review_matching'] = 'Explicit _id_fila plus validated patient/admission/culture/isolate identity'
    else:
        audit['review_matching'] = 'Unique patient/admission/culture/isolate identity'
    decisions = review[[*keys, 'estado']].copy()
    decisions['estado'] = decisions.estado.astype('string').str.strip().str.upper()
    if not decisions.estado.isin([config['review_keep'], config['review_drop']]).all():
        raise ValueError('Missing or unrecognized review status')
    decisions = decisions.drop_duplicates()
    if decisions.duplicated(keys).any():
        raise ValueError('Conflicting review decisions for the same culture/isolate')
    joined = frame.merge(decisions, on=keys, how='left', validate='many_to_one', indicator=True)
    if joined['_merge'].ne('both').any():
        raise ValueError('Source rows lack matching reviewed episodes; review files may be stale')
    matched = decisions.merge(frame[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    if matched['_merge'].ne('both').any():
        raise ValueError('Review contains episodes absent from the source workbook')
    keep = joined.estado.eq(config['review_keep'])
    audit['review_kept_rows'] = int(keep.sum())
    audit['review_removed_rows'] = int((~keep).sum())
    if '_source_row_id' in joined:
        audit['review_removed_source_row_ids'] = joined.loc[~keep, '_source_row_id'].tolist()
    return joined.loc[keep].drop(columns=['estado', '_merge', '_source_row_id'], errors='ignore').reset_index(drop=True)


def attach_treatments(frame, treatment, layout, audit):
    if layout == 'none':
        return frame
    keys = [*EPISODE_KEYS, 'id_hemocultivo']
    require_keys(frame, keys, 'treatment parent')
    require_keys(treatment, keys, 'treatment sheet')
    # Discharge and organism text are mutable descriptors, not culture identity.
    # One real VR row has different organism text between the two sheets.
    audit['treatment_join_keys'] = keys
    for descriptor in ['fecha_alta', 'microorganismo']:
        if descriptor in frame and descriptor in treatment:
            diagnostic = treatment[keys + [descriptor]].drop_duplicates().merge(
                frame[keys + [descriptor]].drop_duplicates(), on=keys + [descriptor],
                how='left', indicator=True)
            audit.setdefault('treatment_descriptor_mismatches', {})[descriptor] = int(diagnostic['_merge'].ne('both').sum())
    absent = set(keys) - set(treatment) | (set(keys) - set(frame))
    if absent:
        raise ValueError(f'Treatment join columns missing: {sorted(absent)}')
    audit['exact_duplicate_treatment_rows_removed'] = int(treatment.duplicated().sum())
    treatment = treatment.drop_duplicates().copy()
    if layout == 'long':
        absent = set(TREATMENT_FIELDS) - set(treatment)
        if absent:
            raise ValueError(f'Long treatment table lacks fields: {sorted(absent)}')
        treatment = treatment.sort_values(keys + ['fecha_inicio_antib_previo', 'fecha_fin_antib_previo'], kind='stable')
        # Nullable non-key values remain individual fields, not packed lists.
        treatment['_slot'] = treatment.groupby(keys, dropna=False).cumcount() + 1
        wide = treatment.set_index(keys + ['_slot'])[TREATMENT_FIELDS].unstack('_slot')
        wide.columns = [f'{field}_{slot}' for field, slot in wide.columns]
        treatment = wide.reset_index()
    else:
        treatment = treatment[keys + [c for c in treatment if TREATMENT_PATTERN.fullmatch(c)]].drop_duplicates()
        if treatment.duplicated(keys).any():
            raise ValueError('Conflicting wide treatment rows; cannot perform a many-to-many join')
    collision = (set(treatment) & set(frame)) - set(keys)
    if collision:
        raise ValueError(f'Treatment slots exist in both sheets: {sorted(collision)}')
    joined = frame.merge(treatment, on=keys, how='left', validate='many_to_one', indicator=True)
    audit['rows_without_treatment_sheet_match'] = int(joined['_merge'].eq('left_only').sum())
    unmatched = treatment[keys].merge(frame[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    audit['treatment_keys_without_parent'] = int(unmatched['_merge'].eq('left_only').sum())
    if audit['treatment_keys_without_parent']:
        raise ValueError('Treatment sheet has unmatched parent cultures; check source versions/join keys')
    return joined.drop(columns='_merge')


def unique_text(values):
    parts = [part.strip() for value in values.dropna() for part in str(value).split(';') if part.strip()]
    return '; '.join(dict.fromkeys(parts)) if parts else pd.NA


def group_episodes(frame, hospital, config, audit):
    require_keys(frame, EPISODE_KEYS, hospital)
    if 'id_hemocultivo' not in frame:
        frame = frame.copy()
        frame['id_hemocultivo'] = frame.fecha_hemocultivo.dt.strftime('%Y_%m_%d') + '_' + frame.record_id
    before = len(frame)
    frame = frame.drop_duplicates()
    audit['exact_duplicate_episode_rows_removed'] = before - len(frame)
    rows, conflicts = [], {}
    treatment_columns = [c for c in frame if TREATMENT_PATTERN.fullmatch(c)]
    slots = sorted({int(c.rsplit('_', 1)[1]) for c in treatment_columns})
    for identity, group in frame.groupby(EPISODE_KEYS, sort=True, dropna=False):
        if len(group) == 1:
            row = group.iloc[0].drop(labels=treatment_columns).to_dict()
        else:
            row = dict(zip(EPISODE_KEYS, identity))
            for column in frame.columns.difference([*EPISODE_KEYS, *treatment_columns], sort=False):
                values = group[column].dropna().drop_duplicates()
                if column in {'id_hemocultivo', 'microorganismo', 'fenotipo_resistencia'} or column.startswith('resultado_antibiograma'):
                    row[column] = unique_text(values)
                else:
                    if len(values) > 1:
                        conflicts[column] = conflicts.get(column, 0) + 1
                        if config['clinical_conflict_policy'] == 'error' or ('_prev' in column and re.search(r'_\d+$', column)):
                            raise ValueError(f'{hospital}: conflicting {column} in a same-day episode')
                    row[column] = values.iloc[0] if len(values) else pd.NA
        # Deduplicate whole courses across isolates, retaining attribute alignment.
        courses = {}
        for source in group[treatment_columns].to_dict('records'):
            for slot in slots:
                course = {f: source.get(f'{f}_{slot}', pd.NA) for f in TREATMENT_FIELDS}
                if all(pd.isna(v) for v in course.values()):
                    continue
                key = tuple(None if pd.isna(v) else str(v) for v in course.values())
                courses.setdefault(key, course)
        for slot, course in enumerate(courses.values(), 1):
            row.update({f'{field}_{slot}': value for field, value in course.items()})
        rows.append(row)
    if not rows:
        raise ValueError(f'{hospital}: no reviewed episodes remain')
    episodes = pd.DataFrame(rows)
    # Ensure optional all-missing dates still have datetime dtype for CSV export.
    for column in episodes:
        if column.startswith('fecha_'):
            episodes[column] = pd.to_datetime(episodes[column])
    episodes = episodes.copy()  # Consolidate date-column blocks before adding metadata.
    audit['same_day_rows_collapsed'] = len(frame) - len(episodes)
    audit['clinical_conflicting_episode_counts'] = conflicts
    episodes['fecha_ingreso_original'] = episodes.fecha_ingreso
    gap = episodes.groupby(['record_id', 'fecha_ingreso_original']).fecha_hemocultivo.diff().dt.days
    limit = config['follow_up_days']
    new = gap.gt(limit)
    follow = gap.between(1, limit) if config['apply_gap_rule_after_review'] else pd.Series(False, index=episodes.index)
    episodes['diff_days_hemocultivo'] = gap
    episodes[f'episode_later_{limit}d'] = new.astype(int)
    episodes['episode_classification'] = 'first_culture'
    episodes.loc[new, 'episode_classification'] = 'new_sparse_episode'
    if config['apply_gap_rule_after_review']:
        episodes.loc[new, 'fecha_ingreso'] = episodes.loc[new, 'fecha_hemocultivo']
    audit['follow_up_episodes_removed'] = int(follow.sum())
    audit['later_episodes'] = int(new.sum())
    episodes = episodes.loc[~follow].copy()
    episodes.insert(0, 'hospital', hospital)
    episodes['episode_id'] = episodes.apply(lambda r: json.dumps(
        [hospital, r.record_id, r.fecha_ingreso.strftime('%Y-%m-%d'), r.fecha_hemocultivo.strftime('%Y-%m-%d')],
        ensure_ascii=False, separators=(',', ':')), axis=1)
    # Preserve raw outcomes; preprocessing recalculates mortality targets.
    return episodes.reset_index(drop=True)


def clinical_recoding(frame, settings, config, audit):
    frame = frame.copy()
    if settings.get('use_x1000_counts'):
        for column in ['leucocitos', 'neutrofilos', 'linfocitos', 'plaquetas']:
            replacement = f'{column}_x1000'
            if replacement not in frame:
                raise ValueError(f'Missing configured VH blood-count column: {replacement}')
            frame[column] = frame.pop(replacement)
    for column in frame:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values) or (values.notna().any() and values.dropna().map(lambda v: isinstance(v, bool)).all()):
            frame[column] = values.astype('Int64')
    for column, mapping in [('sexo', settings['sex_map']),
                            ('somnolencia_estupor_coma', {'NORMAL': 0, 'SOMNOLENCIA': 1, 'ESTUPOR': 1, 'COMA': 1})]:
        if column in frame:
            text = frame[column].astype('string').str.strip().str.upper()
            values = text.map({**mapping, '0': 0, '1': 1, 'FALSE': 0, 'TRUE': 1})
            audit.setdefault('unknown_categorical_values', {})[column] = int((text.notna() & values.isna()).sum())
            frame[column] = values.astype('Int64')
    if 'control_foco' in frame:
        text = frame.control_foco.astype('string').str.strip().str.lower()
        frame['control_foco'] = text.ne('no tiene').astype('Int64').where(text.notna())
    for column in config.get('numeric_columns', []):
        if column not in frame:
            continue
        text = frame[column].astype('string').str.replace(',', '.', regex=False).str.strip()
        lod = pd.to_numeric(text.str.extract(r'^<\s*([0-9]+(?:\.[0-9]+)?)$', expand=False), errors='coerce')
        numeric = pd.to_numeric(text, errors='coerce').astype('Float64')
        numeric = numeric.mask(lod.notna(), lod / math.sqrt(2))
        audit.setdefault('numeric_coercions', {})[column] = int((text.notna() & numeric.isna()).sum())
        audit.setdefault('below_detection_recodings', {})[column] = int(lod.notna().sum())
        frame[column] = numeric
    return frame


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def load_config(path):
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or len(config.get('hospitals', {})) != 4:
        raise ValueError('Configure exactly the four hospital inputs')
    if config['clinical_conflict_policy'] not in {'first_nonmissing', 'error'}:
        raise ValueError('Invalid clinical_conflict_policy')
    if config['schema_policy'] not in {'reference', 'reference_plus_treatment_slots'}:
        raise ValueError('Invalid schema_policy')
    if config['reference_hospital'] not in config['hospitals']:
        raise ValueError('Missing reference hospital')
    if not isinstance(config['follow_up_days'], int) or config['follow_up_days'] < 1:
        raise ValueError('follow_up_days must be a positive integer')
    if not isinstance(config['apply_gap_rule_after_review'], bool):
        raise ValueError('apply_gap_rule_after_review must be a boolean')
    for settings in config['hospitals'].values():
        if settings['treatment_layout'] not in {'none', 'wide', 'long'}:
            raise ValueError('Invalid treatment_layout')
    return config


def load_hospital(raw_dir, settings, config):
    audit, inputs = {}, []
    path = raw_dir / settings['file']
    inputs.append(fingerprint(path))
    frame = read_sheet(path, settings['sheet'], config, audit.setdefault('episode_sheet', {}))
    frame['_source_row_id'] = range(len(frame))
    layout = settings['treatment_layout']
    treatment = read_sheet(path, settings['treatment_sheet'], config, audit.setdefault('treatment_sheet', {})) if layout != 'none' else None
    frame = attach_treatments(frame, treatment, layout, audit)
    review_path = raw_dir / settings['review_file']
    inputs.append(fingerprint(review_path))
    review = read_sheet(review_path, config['review_sheet'], config, audit.setdefault('review_sheet', {}))
    frame = select_reviewed(frame, review, config, audit)
    return frame, audit, inputs


def merge_cohorts(raw_dir, config):
    cohorts, audits, inputs = {}, {}, []
    for hospital, settings in config['hospitals'].items():
        frame, audit, source_inputs = load_hospital(raw_dir, settings, config)
        audits[hospital] = audit
        inputs.extend(source_inputs)
        frame = clinical_recoding(frame, settings, config, audit)
        cohorts[hospital] = group_episodes(frame, hospital, config, audit)
        audit['output_episodes'] = len(cohorts[hospital])
    columns = cohorts[config['reference_hospital']].columns.tolist()
    if config['schema_policy'] == 'reference_plus_treatment_slots':
        columns += sorted({c for f in cohorts.values() for c in f if TREATMENT_PATTERN.fullmatch(c)} - set(columns))
    for hospital, frame in cohorts.items():
        audits[hospital]['columns_added_as_missing'] = sorted(set(columns) - set(frame))
        audits[hospital]['columns_excluded_by_reference_schema'] = sorted(set(frame) - set(columns))
    merged = pd.concat([f.reindex(columns=columns).astype(object) for f in cohorts.values()], ignore_index=True).infer_objects(copy=False)
    require_keys(merged, ['hospital', *EPISODE_KEYS, 'episode_id'], 'merged cohort')
    if merged.episode_id.duplicated().any() or merged.duplicated(['hospital', *EPISODE_KEYS]).any():
        raise ValueError('Duplicate final episode IDs/keys')
    return merged, {'hospitals': audits, 'inputs': inputs, 'config': config,
                    'output_rows': len(merged), 'output_columns': len(merged.columns),
                    'unique_episode_ids': int(merged.episode_id.nunique()),
                    'policy': 'Identity-based reviews; culture-specific treatments; same-day aggregation; notebook adjacent-culture gap rule.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'ANALYSIS/01-PREPROCESS/00-data', help='Directory containing the four hospital XLSX exports and review files')
    parser.add_argument('--config-path', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-path', type=Path, default=ROOT / 'ANALYSIS/01-PREPROCESS/00-data/merged_cohort.csv')
    parser.add_argument('--check-inputs', action='store_true', help='Read-only validation of sheets, joins, reviews and per-hospital episode rules; no output written')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing CSV/audit outputs')
    args = parser.parse_args()
    config = load_config(args.config_path)
    if args.check_inputs:
        for hospital, settings in config['hospitals'].items():
            frame, audit, _ = load_hospital(args.raw_dir, settings, config)
            frame = clinical_recoding(frame, settings, config, audit)
            episodes = group_episodes(frame, hospital, config, audit)
            print(f"{hospital}: input joins and episode rules validated; {audit['review_kept_rows']} reviewed source rows, {len(episodes)} episodes", flush=True)
        print('Input validation complete; no cohort or audit files written.')
        return
    output = args.output_path
    audit_path = output.with_name(f'{output.stem}_merge_audit.json')
    if not args.overwrite and (output.exists() or audit_path.exists()):
        parser.error('Output already exists; choose a new --output-path or explicitly use --overwrite')
    merged, audit = merge_cohorts(args.raw_dir, config)
    audit['configuration_source'] = fingerprint(args.config_path)
    audit['script_source'] = fingerprint(Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w' if args.overwrite else 'x') as handle:
        merged.to_csv(handle, index=False, date_format='%Y-%m-%d')
    with audit_path.open('w' if args.overwrite else 'x') as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f'Merged {len(merged)} episodes from four hospitals: {output}')
    print(f'Merge audit: {audit_path}')


if __name__ == '__main__':
    main()
