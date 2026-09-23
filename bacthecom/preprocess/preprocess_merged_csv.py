from __future__ import annotations

import fnmatch
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def slug(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value).lower())
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.replace("l·l", "l").replace("l.l", "l")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def is_repeated_source(column: str) -> bool:
    """Match history/result slots without dropping names such as barthel_inf_90."""
    return bool(re.match(
        r"^(?:microorganism_episodio_previo|feno_resist_infec_prev|especimen_previo|cultivo_previo|"
        r"area_hosp_infecc_previa|fecha_episodio_previo|resultado_antibiograma_infec_previa_[rsi](?:_cmi)?|"
        r"antimicrobiano_previo|atc_antib_prev|dias_trat_antimicrobiano|via_administ_antib_prev|"
        r"fecha_(?:inicio|fin)_antib_previo|microorganismo_otros_cult|fenotipo_resistencia_otros|"
        r"especimen_otros|area_hosp_otros|id_otros_cultivos|resultado_antibiograma_otros_[rsi](?:_cmi)?)_\d+$", column))


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="ISO8601", errors="coerce")


def load_domain_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None and "domain" in config:
        return config["domain"]
    # Older run snapshots predate the consolidated domain section.
    path = Path(__file__).resolve().parent / "config/preprocess_bacthecom.yml"
    return yaml.safe_load(path.read_text())["domain"]


def child_pugh_class(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    parts = re.split(r"[;/,–-]+", str(value).strip().upper())
    matches = [re.fullmatch(r"([ABC])\s*(?:\d+)?", part.strip()) for part in parts]
    if not all(matches):
        return "unknown"
    classes = {match.group(1).lower() for match in matches}
    return classes.pop() if len(classes) == 1 else "unknown"


def recency_features(days, has_history, uncertain_dates, prefix, windows):
    """Preserve uncertain recency in the window features themselves.

    Positive dated evidence wins; an undated record prevents a confident negative
    window flag. Days remain missing when no valid date exists, never fake zero.
    """
    result = pd.DataFrame(index=days.index)
    unknown = has_history & (days.isna() | uncertain_dates)
    for window in windows:
        if not isinstance(window, int) or window <= 0:
            raise ValueError("History windows must be positive integer days")
        positive = has_history & days.between(1, window)
        values = pd.Series(0.0, index=days.index).where(has_history)
        values.loc[unknown & ~positive] = np.nan
        values.loc[positive] = 1
        result[f"{prefix}_{window}d"] = values
    return result


def admission_history_features(predictors, settings):
    """Derive only what reported month/year flags establish; do not invent dates."""
    options = settings.get("admission_history", {})
    empty = pd.Series(np.nan, index=predictors.index)
    month = predictors.get(options.get("month_column", "hospit_mes_previo"), empty)
    year = predictors.get(options.get("year_column", "hospit_ano_previo"), empty)
    for label, values in [("month", month), ("year", year)]:
        if not values.dropna().isin([0, 1]).all():
            raise ValueError(f"Previous admission {label} flag must be binary")
    conflict = month.eq(1) & year.eq(0)
    recent30 = month.mask(month.isna() & year.eq(0), 0).mask(conflict)
    recent90 = pd.Series(np.nan, index=predictors.index)
    recent90.loc[year.eq(0)] = 0
    recent90.loc[month.eq(1)] = 1
    recent90.loc[conflict] = np.nan
    result = pd.DataFrame({"hospit_previa_30d": recent30, "hospit_previa_90d": recent90}, index=predictors.index)
    return result, {"admission_history_policy": "30d uses the reported previous-month flag, or 0 if month is missing and year=0. 90d=1 if month=1; 90d=0 if year=0; otherwise unknown. Month=1/year=0 conflicts yield unknown windows.",
                    "admission_history_flag_conflicts": int(conflict.sum()),
                    "admission_history_90d_unknown_rows": int(recent90.isna().sum())}


def antibiotic_features(frame: pd.DataFrame, settings: dict[str, Any], domain: dict):
    """Summarize prior exposures; keep unavailable dates and measurements missing."""
    empty = pd.Series(np.nan, index=frame.index)
    index_date = parse_date(frame["fecha_hemocultivo"])
    aliases = settings.get("antibiotic_aliases", {})
    mapping = {}
    for source, family, english in domain["antimicrobial_groups"]:
        for name in [source if source not in {"A", "B"} else None, english]:
            if name:
                key = slug(name)
                mapping[aliases.get(key, key)] = family
    mapping.update(domain["antimicrobial_family_extensions"])
    groups = settings["antibiotic_family_groups"]
    result = pd.DataFrame(0, index=frame.index, columns=[f"antib_prev_{g}" for g in dict.fromkeys(groups.values())])
    count = pd.Series(0, index=frame.index)
    unknown = pd.Series(False, index=frame.index)
    unclassified = unknown.copy()
    timing_unknown = unknown.copy()
    recorded = unknown.copy()
    dates, durations, resolved, broad_resolved = {}, {}, {}, {}
    duration_missing = pd.Series(False, index=frame.index)
    excluded = 0
    slots = sorted({m.group(1) for c in frame for m in [re.match(
        r"^(?:antimicrobiano_previo_|atc_antib_prev_|dias_trat_antimicrobiano_|fecha_(?:inicio|fin)_antib_previo_|via_administ_antib_prev_)(\d+)$", c)] if m}, key=int)
    for slot in slots:
        names = frame.get(f"antimicrobiano_previo_{slot}", empty)
        codes = frame.get(f"atc_antib_prev_{slot}", empty)
        starts = parse_date(frame.get(f"fecha_inicio_antib_previo_{slot}", empty))
        ends = parse_date(frame.get(f"fecha_fin_antib_previo_{slot}", empty))
        raw_duration = pd.to_numeric(frame.get(f"dias_trat_antimicrobiano_{slot}", empty).astype(str).str.replace(",", ".", regex=False), errors="coerce").replace([np.inf, -np.inf], np.nan)
        exists = names.notna() | codes.notna() | starts.notna() | ends.notna() | raw_duration.notna()
        recorded |= exists
        # A future/same-day start is not prior; absent starts can use prior ends.
        eligible = exists & (starts.lt(index_date) | (starts.isna() & (ends.lt(index_date) | ends.isna())))
        excluded += int((exists & ~eligible).sum())
        timing_unknown |= eligible & starts.isna() & ends.isna()
        dates[slot] = ends.where(ends.lt(index_date) & (starts.isna() | ends.ge(starts))).fillna(starts.where(starts.lt(index_date))).where(eligible)
        # Never include eventual total duration of a course ending after prediction.
        durations[slot] = raw_duration.where(eligible & raw_duration.ge(0) & (ends.isna() | ends.lt(index_date)))
        duration_missing |= eligible & durations[slot].isna()
        for idx in frame.index[eligible]:
            drugs = str(names.at[idx]).split(";") if pd.notna(names.at[idx]) else []
            atcs = str(codes.at[idx]).split(";") if pd.notna(codes.at[idx]) else []
            n = max(len(drugs), len(atcs), 1)
            count.at[idx] += n
            for position in range(n):
                name = slug(re.sub(r"\(.*?\)", "", drugs[position])) if position < len(drugs) else ""
                # Ambiguous multi-name/code alignment must not attach the wrong ATC.
                code = atcs[position].strip().upper() if (not drugs or len(atcs) == len(drugs)) and position < len(atcs) else ""
                drug = settings.get("antibiotic_atc_names", {}).get(code, aliases.get(name, name))
                family = mapping.get(drug, "Otros_no_clasificados")
                broad = groups.get(family, "Otros_no_clasificados")
                resolved[drug or code or "missing_drug"] = family
                broad_resolved[drug or code or "missing_drug"] = broad
                if drug not in mapping or family not in groups:
                    unknown.at[idx] = True
                    unclassified.at[idx] = True
                result.at[idx, f"antib_prev_{broad}"] = 1
    family_columns = result.columns.tolist()
    result[family_columns] = result[family_columns].where(count.gt(0), axis=0)
    result["antib_previo_si_no"] = count.gt(0).astype(float).where(recorded)
    result["antib_previo_num_exposiciones"] = count
    result["antib_previo_num_familias"] = result[family_columns].sum(axis=1, min_count=1)
    duration_frame = pd.DataFrame(durations, index=frame.index)
    for stat in ["total", "max", "mean"]:
        values = duration_frame.sum(axis=1, min_count=1) if stat == "total" else getattr(duration_frame, stat)(axis=1)
        result[f"antib_previo_{stat}_dias"] = values
    latest = pd.DataFrame(dates, index=frame.index).max(axis=1) if dates else pd.Series(pd.NaT, index=frame.index)
    result["dias_desde_ultimo_antib"] = (index_date - latest).dt.days
    result = pd.concat([result, recency_features(result["dias_desde_ultimo_antib"], count.gt(0), timing_unknown,
        "antib_previo", settings.get("history_windows_days", [30, 90]))], axis=1)
    return result, {
        "antimicrobial_drug_family_map": resolved, "antimicrobial_drug_group_map": broad_resolved,
        "antimicrobial_family_source": domain["family_source"],
        "rows_with_antibiotic_history": int(recorded.sum()),
        "antibiotic_recency_states": {"no_recorded_prior_history": int(count.eq(0).sum()),
            "history_without_any_valid_prior_date": int((count.gt(0) & result["dias_desde_ultimo_antib"].isna()).sum()),
            "history_with_a_valid_prior_date": int((count.gt(0) & result["dias_desde_ultimo_antib"].notna()).sum()),
            "history_with_any_unknown_date": int(timing_unknown.sum())},
        "rows_with_unknown_antibiotic_family_mapping": frame.index[unknown].tolist(),
        "rows_with_unclassified_previous_antibiotic": frame.index[unclassified].tolist(),
        "rows_with_unknown_antibiotic_timing": int(timing_unknown.sum()),
        "rows_with_incomplete_antibiotic_duration": int(duration_missing.sum()),
        "antibiotic_entries_on_or_after_index": excluded,
    }


def evidence_flag(values: pd.Series) -> pd.Series:
    """Recorded positive/explicit negative/unknown, without treating blanks as no."""
    normalized = values.map(lambda value: slug(value) if pd.notna(value) else "")
    unknown = normalized.isin(["", "nan", "none", "unknown", "desconocido", "no_consta", "no_disponible", "pendiente"])
    negative = normalized.isin(["0", "no", "negativo", "negativa", "ninguno", "ninguna", "sin_crecimiento", "sin_resistencia", "sin_resistencias", "sensible", "sensible_a_todos"])
    return pd.Series(np.where(unknown, np.nan, np.where(negative, 0.0, 1.0)), index=values.index)


def microbiology_features(frame: pd.DataFrame, settings: dict[str, Any]):
    """Compact documented-evidence flags; zeros are not confirmed clinical negatives."""
    options = settings.get("microbiology", {})
    empty = pd.Series(np.nan, index=frame.index)
    index_date = parse_date(frame["fecha_hemocultivo"])
    features, definitions, timing, unmapped = {}, {}, {}, set()

    def register(name, values, sources, rule, availability):
        features[name] = values.astype(float)
        definitions[name] = {"sources": sources, "rule": rule,
                             "availability": availability,
                             "zero_meaning": "No matching evidence among observed eligible results; missing source results remain NaN."}

    contexts = {
        "infec": [("microorganismo", "fenotipo_resistencia", "resultado_antibiograma_hemocultivo", "", pd.Series(True, index=frame.index))],
        "infec_prev": [], "otros": [],
    }
    # Find slots from every source, including resistance-only historical records.
    slots = sorted({match.group(1) for column in frame for match in [re.match(
        r"^(?:microorganism_episodio_previo_|feno_resist_infec_prev_|resultado_antibiograma_infec_previa_[rsi]_)(\d+)$", column)] if match}, key=int)
    for slot in slots:
        date = parse_date(frame.get(f"fecha_episodio_previo_{slot}", empty))
        contexts["infec_prev"].append((f"microorganism_episodio_previo_{slot}", f"feno_resist_infec_prev_{slot}",
                                      "resultado_antibiograma_infec_previa", f"_{slot}", (date.isna() | date.lt(index_date))))
    slots = sorted({match.group(1) for column in frame for match in [re.match(
        r"^(?:microorganismo_otros_cult_|fenotipo_resistencia_otros_|resultado_antibiograma_otros_[rsi]_)(\d+)$", column)] if match}, key=int)
    for slot in slots:
        date = parse_date(frame.get(f"fecha_otros_cultivos_{slot}", frame.get(f"fecha_ otros_cultivos_{slot}", empty)))
        contexts["otros"].append((f"microorganismo_otros_cult_{slot}", f"fenotipo_resistencia_otros_{slot}",
                                 "resultado_antibiograma_otros", f"_{slot}", date.le(index_date)))

    for context, entries in contexts.items():
        availability = {"infec_prev": "historical", "infec": "current_hemoculture_results", "otros": "other_culture_results"}[context]
        suffix = {"infec": "", "infec_prev": "_prev", "otros": "_otros"}[context]
        organism_flags = {label: pd.Series(False, index=frame.index) for label in options["organism_patterns"]}
        resistance_flags = {label: pd.Series(False, index=frame.index) for label in options["phenotype_patterns"]}
        organism_other = pd.Series(False, index=frame.index)
        phenotype_other = pd.Series(False, index=frame.index)
        organism_known = pd.Series(False, index=frame.index)
        phenotype_known = pd.Series(False, index=frame.index)
        resistance_any = pd.Series(False, index=frame.index)
        positive_slots = pd.Series(0, index=frame.index)
        sources = []
        for organism_col, phenotype_col, antibiogram_prefix, slot, eligible in entries:
            antibiogram_cols = [f"{antibiogram_prefix}_{kind}{slot}" for kind in "rsi"]
            sources.extend([organism_col, phenotype_col, *antibiogram_cols])
            organism = frame.get(organism_col, empty)
            phenotype = frame.get(phenotype_col, empty)
            organism_evidence = evidence_flag(organism)
            phenotype_evidence = evidence_flag(phenotype)
            ast = pd.DataFrame({col: evidence_flag(frame.get(col, empty)) for col in antibiogram_cols})
            recorded = organism_evidence.notna() | phenotype_evidence.notna() | ast.notna().any(axis=1)
            timing[organism_col] = int((recorded & ~eligible).sum())
            organism_known |= organism_evidence.notna() & eligible
            phenotype_known |= phenotype_evidence.notna() & eligible
            positive_slots += (organism_evidence.eq(1) & eligible).astype(int)
            resistance_any |= (phenotype_evidence.eq(1) | ast[antibiogram_cols[0]].eq(1)) & eligible
            # Split organism lists before matching so a mixed known/other culture retains both.
            parts = organism.fillna("").astype(str).str.split(r"[;,|]+")
            for idx in frame.index[eligible & organism_evidence.eq(1)]:
                for part in parts.at[idx]:
                    text = slug(part)
                    if not text:
                        continue
                    labels = [label for label, pattern in options["organism_patterns"].items() if re.search(pattern, text)]
                    for label in labels:
                        organism_flags[label].at[idx] = True
                    if not labels:
                        organism_other.at[idx] = True
            normalized = phenotype.map(lambda value: slug(value) if pd.notna(value) else "")
            matched = pd.Series(False, index=frame.index)
            for label, pattern in options["phenotype_patterns"].items():
                hit = normalized.str.contains(pattern, regex=True) & phenotype_evidence.eq(1)
                matched |= hit
                resistance_flags[label] |= hit & eligible
            unknown_label = phenotype_evidence.eq(1) & ~matched & eligible
            phenotype_other |= unknown_label
            unmapped.update(phenotype[unknown_label].dropna().unique())
        sources = [col for col in sources if col in frame]
        for label, values in organism_flags.items():
            register(f"{context}_{label}", values.where(organism_known), sources, "Any matching organism across eligible slots; all historical slots merged by OR.", availability)
        register(f"{context}_Other", organism_other.where(organism_known), sources, "Any recorded organism outside the configured groups.", availability)
        register(f"{context}_positive_slots", positive_slots.where(organism_known), sources, "Number of eligible slots with an organism; not a count of unique infections.", availability)
        for label, values in resistance_flags.items():
            name = f"multiresistant{suffix}" if label == "MDR" else f"resistant{suffix}_{label}"
            register(name, values.where(phenotype_known), sources, "Any matching recorded phenotype across eligible slots; MDR uses explicit MDR/MR/XDR labels, not inferred susceptibility counts.", availability)
        register(f"resistant{suffix}_Other", phenotype_other.where(phenotype_known), sources, "Any unclassified recorded resistance phenotype.", availability)
        register(f"resistant{suffix}_any", resistance_any.where(phenotype_known | resistance_any), sources, "Any recorded resistant drug or resistance phenotype across eligible slots.", availability)
    full = pd.DataFrame(features, index=frame.index)
    include_current = options.get("include_current_results", False)
    selected = [name for name, definition in definitions.items() if definition["availability"] == "historical" or
                (include_current and definition["availability"] == "current_hemoculture_results")]
    for name, definition in definitions.items():
        definition["included_in_predictors"] = name in selected
    return full, full[selected], {
        "engineered_feature_definitions": definitions,
        "microbiology_excluded_by_sample_timing": timing,
        "unmapped_phenotype_values": sorted(unmapped),
        "microbiology_availability_assumption": "Current hemoculture organism and resistance results assumed available at prediction; other cultures remain audit-only." if include_current else "Current/other results are audit-only.",
        "microbiology_evidence_policy": "1=qualifying documented evidence, 0=no matching evidence among observed eligible results. Missing source results remain NaN; no coverage indicators are generated. Known historical dates must precede index culture; undated previous records are retained and audited. Release timestamps unavailable.",
        "engineered_feature_counts": {name: {str(value): int(count) for value, count in full[name].value_counts(dropna=False).items()} for name in full},
    }


def infection_history_features(frame, settings):
    empty = pd.Series(np.nan, index=frame.index)
    index_date = parse_date(frame.fecha_hemocultivo)
    options = settings["previous_sources"]
    result = pd.DataFrame(0, index=frame.index, columns=[f"foco_prev_{label}" for label in [*options, "otra"]])
    count = pd.Series(0, index=frame.index)
    unknown = pd.Series(False, index=frame.index)
    timing = unknown.copy()
    dates = {}
    source_known = pd.Series(False, index=frame.index)
    slots = sorted({m.group(1) for c in frame for m in [re.match(
        r"^(?:fecha_episodio_previo_|especimen_previo_|cultivo_previo_|microorganism_episodio_previo_|feno_resist_infec_prev_|area_hosp_infecc_previa_|resultado_antibiograma_infec_previa_[rsi](?:_cmi)?_)(\d+)$", c)] if m}, key=int)
    for slot in slots:
        date = parse_date(frame.get(f"fecha_episodio_previo_{slot}", empty))
        prefixes = ("fecha_episodio_previo_", "especimen_previo_", "cultivo_previo_",
                    "microorganism_episodio_previo_", "feno_resist_infec_prev_",
                    "area_hosp_infecc_previa_", "resultado_antibiograma_infec_previa_")
        cols = [c for c in frame if c.startswith(prefixes) and c.rsplit("_", 1)[-1] == slot]
        exists = frame[cols].notna().any(axis=1)
        eligible = exists & (date.isna() | date.lt(index_date))
        timing |= exists & date.isna()
        count += eligible.astype(int)
        dates[slot] = date.where(eligible)
        source = frame.get(f"especimen_previo_{slot}", empty).fillna("").map(slug)
        source_known |= eligible & source.ne("")
        matched = pd.Series(False, index=frame.index)
        for label, pattern in options.items():
            hit = source.str.contains(pattern, regex=True)
            result[f"foco_prev_{label}"] |= (hit & eligible).astype(int)
            matched |= hit
        result["foco_prev_otra"] |= (eligible & source.ne("") & ~matched).astype(int)
        organism = frame.get(f"microorganism_episodio_previo_{slot}", empty)
        phenotype = frame.get(f"feno_resist_infec_prev_{slot}", empty)
        org_match = organism.fillna("").map(slug).str.contains("|".join(settings["microbiology"]["organism_patterns"].values()), regex=True)
        res_match = phenotype.fillna("").map(slug).str.contains("|".join(settings["microbiology"]["phenotype_patterns"].values()), regex=True)
        unknown |= eligible & ((evidence_flag(organism).eq(1) & ~org_match) |
                               (evidence_flag(phenotype).eq(1) & ~res_match) |
                               (source.ne("") & ~matched) |
                               (~org_match & ~res_match & ~matched))
    result = result.where(source_known, axis=0)
    result["num_inf_previas"] = count
    result["infec_previa_si_no"] = count.gt(0).astype(int)
    latest = pd.DataFrame(dates, index=frame.index).max(axis=1) if dates else pd.Series(pd.NaT, index=frame.index)
    result["dias_desde_ultima_infec"] = (index_date - latest).dt.days
    result = pd.concat([result, recency_features(result["dias_desde_ultima_infec"], count.gt(0), timing,
        "infec_previa", settings.get("history_windows_days", [30, 90]))], axis=1)
    return result, {"infection_recency_states": {"no_recorded_prior_history": int(count.eq(0).sum()),
                        "history_without_any_valid_prior_date": int((count.gt(0) & result["dias_desde_ultima_infec"].isna()).sum()),
                        "history_with_a_valid_prior_date": int((count.gt(0) & result["dias_desde_ultima_infec"].notna()).sum()),
                        "history_with_any_unknown_date": int(timing.sum())},
                    "rows_with_unclassified_previous_infection": frame.index[unknown].tolist(),
                    "previous_infection_count_policy": "Count occupied eligible source slots, including source-only/resistance-only records; not necessarily unique clinical infections."}


def clinical_features(predictors, settings):
    options = settings["clinical_features"]
    result = {}
    for name, rule in options["binary_rules"].items():
        values = predictors.get(rule["column"], pd.Series(np.nan, index=predictors.index))
        positive = pd.Series(False, index=predictors.index)
        for op in ["gt", "ge", "lt", "le"]:
            if op in rule:
                positive |= getattr(values, op)(rule[op])
        result[name] = positive.astype(float).where(values.notna())
    devices = predictors.reindex(columns=options["device_columns"])
    # Partial missingness must not silently lower device burden.
    result["carga_dispositivos"] = devices.where(devices.isin([0, 1])).sum(axis=1, min_count=len(devices.columns))
    return pd.DataFrame(result, index=predictors.index)


def mortality_values(values: pd.Series) -> pd.Series:
    """Normalize numeric and boolean mortality flags without imputing unknowns."""
    normalized = values.astype("string").str.strip().str.lower()
    normalized = normalized.mask(normalized.eq(""))
    normalized = normalized.replace({"false": "0", "true": "1"})
    numeric = pd.to_numeric(normalized, errors="coerce").astype(float)
    invalid = normalized.notna() & (numeric.isna() | ~numeric.isin([0, 1]))
    if invalid.any():
        unexpected = sorted(normalized[invalid].unique().tolist())
        raise ValueError(f"mortalidad must contain 0, 1, False, True, or missing values; unexpected values: {unexpected}")
    return numeric


def mortality_targets(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    raw = mortality_values(frame["mortalidad"])
    death = parse_date(frame["fecha_mortalidad"])
    targets = pd.DataFrame(index=frame.index)
    targets["mortalidad_any"] = raw.mask(death.notna(), 1)
    settings = config["targets"]["early_mortality"]
    for reference, prefix in [("fecha_ingreso", "admission")]:
        days = (death - parse_date(frame[reference])).dt.days
        for window in settings.get("windows_days", [14, 30]):
            name = settings["columns"][f"{prefix}_{window}"]
            values = pd.Series(np.nan, index=frame.index)
            values.loc[targets.mortalidad_any.eq(0)] = 0
            known = targets.mortalidad_any.eq(1) & days.ge(0)
            values.loc[known] = days.loc[known].le(window).astype(int)
            targets[name] = values
    return targets


def preprocess_csv(input_path: Path, output_path: Path, config: dict[str, Any], config_path: Path):
    settings = config["merged_csv"]
    domain = load_domain_config(config)
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    # Discard obsolete date-referenced mortality labels supplied by older exports.
    frame = frame.drop(columns=frame.filter(regex=r"^mortalidad_(?:(?:14|30)_)?dias_desde_.+$").columns)
    frame = frame.apply(lambda values: values.str.strip())
    frame = frame.mask(frame.isin(settings["missing_tokens"]))
    # Accept earlier exports while using hospital consistently in new outputs.
    if "hospital" not in frame and "hospital_cohort" in frame:
        frame = frame.rename(columns={"hospital_cohort": "hospital"})
    keys = list(dict.fromkeys("hospital" if key == "hospital_cohort" else key
                             for key in settings["episode_keys"]))
    missing = sorted(set(keys + ["mortalidad", "fecha_mortalidad"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate episode keys: {keys}")
    if any(parse_date(frame[column]).isna().any() for column in ["fecha_ingreso", "fecha_hemocultivo"]):
        raise ValueError("Episode dates must be valid ISO dates")
    if "episode_id" not in frame:
        # JSON encoding is unambiguous even if individual keys contain delimiters.
        frame["episode_id"] = frame[keys].apply(lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)
    if frame.episode_id.isna().any() or frame.episode_id.duplicated().any():
        raise ValueError("episode_id must be complete and unique; resolve duplicate episodes upstream")
    targets = mortality_targets(frame, config)
    missing_targets = set(config["targets"]["required_final_targets"]) - set(targets.columns)
    if missing_targets:
        raise ValueError(f"Required targets were not generated: {sorted(missing_targets)}")
    predictor_drugs, audit = antibiotic_features(frame, settings, domain)
    audit_drugs = predictor_drugs
    family_audit = audit.copy()
    audit_microbiology, predictor_microbiology, microbiology_audit = microbiology_features(frame, settings)
    rename = {"infec_prev_" + short: "infec_prev_" + long for short, long in {
        "Ecoli": "Escherichia_coli", "Kpneumoniae": "Klebsiella_pneumoniae",
        "Paeruginosa": "Pseudomonas_aeruginosa", "Saureus": "Staphylococcus_aureus"}.items()}
    rename.update({c: c.replace("resistant_prev_", "res_prev_") for c in predictor_microbiology if c.startswith("resistant_prev_")})
    rename["multiresistant_prev"] = "res_prev_MDR"
    audit_microbiology = audit_microbiology.rename(columns=rename)
    predictor_microbiology = predictor_microbiology.rename(columns=rename)
    for key in ["engineered_feature_definitions", "engineered_feature_counts"]:
        microbiology_audit[key] = {rename.get(k, k): v for k, v in microbiology_audit[key].items()}
    audit.update(microbiology_audit)
    history, history_audit = infection_history_features(frame, settings)
    unknown_infection_rows = set(history_audit["rows_with_unclassified_previous_infection"])
    unknown_infection_rows.update(predictor_microbiology.index[
        predictor_microbiology["infec_prev_Other"].eq(1) | predictor_microbiology["res_prev_Other"].eq(1)].tolist())
    history_audit["rows_with_unclassified_previous_infection"] = sorted(unknown_infection_rows)
    audit.update(history_audit)
    audit.update({"input_rows": len(frame), "input_columns": len(frame.columns),
                  "cohort_counts": frame.hospital.value_counts().to_dict(),
                  "death_before_hemoculture": int((parse_date(frame.fecha_mortalidad) < parse_date(frame.fecha_hemocultivo)).sum()),
                  "mortality_flag_date_conflicts": int((mortality_values(frame.mortalidad).eq(0) & frame.fecha_mortalidad.notna()).sum())})
    audit["invalid_death_dates"] = int((frame.fecha_mortalidad.notna() & parse_date(frame.fecha_mortalidad).isna()).sum())
    if audit["invalid_death_dates"] or audit["death_before_hemoculture"]:
        raise ValueError("Invalid death dates or death before prediction time; correct source records before preprocessing")
    full = pd.concat([frame.drop(columns=targets.columns, errors="ignore"), targets, audit_drugs, audit_microbiology, history], axis=1)
    patterns = list(settings["exclude_predictors"])
    for name in ["drop_columns", "drop_patterns", "forbidden_predictor_columns"]:
        patterns += config["filtering"].get(name, [])
    if settings.get("drop_columns_file"):
        drop_file = config_path.parent / settings["drop_columns_file"]
        patterns += [line.strip() for line in drop_file.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    patterns += ["hospital*", "centro", "record_id", "episode_*", "fecha_*", "mortalidad*",
                 "microorganismo*", "fenotipo_resistencia*", "resultado_antibiograma*",
                 "hemo_positivo_si_no", "duracion_uci", "duracion_hospitalizacion", "control_foco"]
    # Defensive exclusion independent of configured lists: no repeated source slots.
    patterns += [c for c in frame if is_repeated_source(c)]
    excluded = [column for column in frame if any(fnmatch.fnmatchcase(column.lower(), pattern.lower()) for pattern in patterns)]
    features = frame.drop(columns=excluded + list(targets.columns), errors="ignore")
    conversions: dict[str, int] = {}
    categories: dict[str, list[str]] = {}
    encoded: dict[str, pd.Series] = {}
    for column in features:
        values = features[column]
        if column == "puntaje_child_pugh":
            normalized = values.map(child_pugh_class)
            categories[column] = [level for level in domain["child_pugh_levels"] if level != "unknown"]
            for label in categories[column]:
                encoded[f"{column}__{label}"] = normalized.eq(label).astype(float).where(normalized.isin(categories[column]))
            audit["child_pugh_class_counts"] = normalized.value_counts().to_dict()
            audit["child_pugh_unresolved_values"] = sorted(values[normalized.eq("unknown") & values.notna()].unique().tolist())
            continue
        if column in settings["categorical_columns"]:
            normalized = values.map(lambda value: slug(value) if pd.notna(value) else np.nan)
            normalized = normalized.mask(normalized.isin(["missing", "unknown", "desconocido", "no_consta", "no_disponible"]))
            labels = sorted(normalized.dropna().unique())
            categories[column] = labels
            if not labels:
                encoded[column] = pd.Series(np.nan, index=frame.index)
            for label in labels:
                encoded[f"{column}__{label}"] = normalized.eq(label).astype(float).where(normalized.isin(categories[column]))
            continue
        normalized = values.str.lower().replace({"si": "1", "sí": "1", "yes": "1", "true": "1", "no": "0", "false": "0"})
        numeric = pd.to_numeric(normalized.str.replace(",", ".", regex=False), errors="coerce")
        invalid = values.notna() & numeric.isna()
        conversions[column] = int(invalid.sum())
        if invalid.any() and column not in settings["numeric_columns"]:
            raise ValueError(f"Unconfigured nonnumeric predictor {column}: configure as categorical, numeric, or excluded")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if column in settings["nonnegative_columns"]:
            numeric = numeric.mask(numeric.lt(0))
        encoded[column] = numeric
    predictors = pd.concat([pd.DataFrame(encoded, index=frame.index), predictor_drugs, predictor_microbiology, history], axis=1)
    if "saturacion_po2" in predictors:
        saturation = predictors["saturacion_po2"]
        saturation = saturation.where(~saturation.between(0, 1), saturation * 100)
        predictors["saturacion_po2"] = saturation.where(saturation.between(0, 100))
    if "edad" in predictors:
        predictors["edad"] = predictors["edad"].where(predictors["edad"].between(0, 120))
    admission, admission_audit = admission_history_features(predictors, settings)
    audit.update(admission_audit)
    audit["missing_values_assumed_absent"] = {}
    clinical = clinical_features(predictors, settings)
    clinical = pd.concat([clinical, admission], axis=1)
    predictors = pd.concat([predictors, clinical], axis=1)
    full = pd.concat([full, clinical], axis=1)
    if predictors.columns.duplicated().any():
        raise ValueError("Encoded feature name collision")
    if set(predictors) & set(targets):
        raise ValueError("Predictor/target overlap")
    if any(is_repeated_source(c) for c in predictors):
        raise ValueError("Raw numbered source slot survived predictor filtering")
    filtered = pd.concat([predictors, targets], axis=1)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in filtered.dtypes):
        raise ValueError("Training dataset contains nonnumeric columns")
    # Retain episode identifiers for tracing cases, outside the predictor matrix.
    identifier_columns = ["hospital", "fecha_ingreso", "record_id", "fecha_hemocultivo"]
    filtered = pd.concat([frame[identifier_columns], filtered], axis=1)
    identifiers = frame[list(dict.fromkeys(["episode_id", *keys, *[c for c in ["centro", "hospital_id"] if c in frame]]))].copy()
    identifiers.insert(0, "row_number", np.arange(len(frame)))
    identifiers["patient_group"] = frame["hospital"] + ":" + frame["record_id"]
    generated = list(dict.fromkeys([*predictor_drugs, *predictor_microbiology, *history, *clinical,
                                   *[c for c in encoded if c not in frame]]))
    binary = [c for c in generated if not c.endswith(("_slots", "_dias")) and
              c not in {"num_inf_previas", "carga_dispositivos", "dias_desde_ultima_infec", "dias_desde_ultimo_antib", "antib_previo_num_exposiciones", "antib_previo_num_familias"}]
    audit.update({
        "generated_feature_count": len(generated), "source_columns_removed_count": len(excluded),
        "generated_binary_prevalence": {c: {"positive_rows": int(predictors[c].eq(1).sum()),
            "observed_rows": int(predictors[c].notna().sum()),
            "prevalence": float(predictors[c].mean()) if predictors[c].notna().any() else None} for c in binary},
        "unique_episode_ids": int(frame.episode_id.nunique()),
        "output_rows": len(filtered), "predictor_count": len(predictors.columns),
        "excluded_raw_columns": excluded,
        "numeric_values_coerced_to_missing": {column: count for column, count in conversions.items() if count},
        "all_missing_predictors": predictors.columns[predictors.isna().all()].tolist(),
        "constant_observed_predictors": predictors.columns[predictors.nunique() <= 1].tolist(),
        "target_counts": {column: {str(value): int(count) for value, count in targets[column].value_counts(dropna=False).items()} for column in targets},
        "antibiotic_policy": "Binary documented prior exposure: missing histories remain NaN. Undated previous records retained and audited; known same-day/future starts excluded. Duration excludes courses ending on/after culture.",
        "missing_value_policy": "Missing feature values remain NaN; no missingness indicators, unknown categories or missing-to-zero imputation. Categorical encodings propagate missingness. Imputation/scaling/selection belongs inside training folds.",
    })
    for key in ["rows_with_unknown_antibiotic_family_mapping", "rows_with_unclassified_previous_antibiotic", "rows_with_unclassified_previous_infection"]:
        audit[key + "_count"] = len(audit[key])
        audit[key + "_episode_ids"] = frame.loc[audit[key], "episode_id"].tolist()
    schema = {"predictor_columns": predictors.columns.tolist(), "target_columns": targets.columns.tolist(), "categorical_levels": categories,
              "identifier_columns": identifier_columns,
              "group_file": f"{output_path.stem}_identifiers.csv", "config": config, "domain_config": domain,
              "antimicrobial_drug_family_map": family_audit["antimicrobial_drug_family_map"],
              "antimicrobial_drug_group_map": family_audit["antimicrobial_drug_group_map"],
              "engineered_feature_definitions": microbiology_audit["engineered_feature_definitions"],
              "microbiology_availability_assumption": microbiology_audit["microbiology_availability_assumption"],
              "microbiology_evidence_policy": microbiology_audit["microbiology_evidence_policy"]}
    outcome_columns = [c for c in frame if c in {"microorganismo", "fenotipo_resistencia", "hemo_positivo_si_no"} or c.startswith("resultado_antibiograma_hemocultivo")]
    outcomes = pd.concat([identifiers[["episode_id"]], targets, frame[outcome_columns]], axis=1)
    schema.update({"metadata_columns": identifiers.columns.tolist(), "outcome_columns": outcomes.columns.drop("episode_id").tolist(),
                   "outcomes_file": f"{output_path.stem}_outcomes.csv", "generated_predictor_columns": generated,
                   "clinical_rules": settings.get("clinical_features", {}),
                   "history_policy": audit["antibiotic_policy"],
                   "recency_policy": "Days measure latest valid prior date and remain missing without a valid date. No missingness indicators are generated. Window positives use dated evidence; uncertain negatives remain missing.",
                   "admission_history_policy": audit["admission_history_policy"],
                   "missing_as_zero_columns": []})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(audit["generated_binary_prevalence"], orient="index").rename_axis("feature").to_csv(
        output_path.with_name(f"{output_path.stem}_binary_prevalence.csv"))
    outcomes.to_csv(output_path.with_name(f"{output_path.stem}_outcomes.csv"), index=False)
    full.to_csv(output_path, index=False)
    filtered.to_csv(output_path.with_name(f"{output_path.stem}_filtered.csv"), index=False)
    identifiers.to_csv(output_path.with_name(f"{output_path.stem}_identifiers.csv"), index=False)
    for name, contents in [("schema", schema), ("audit", audit)]:
        output_path.with_name(f"{output_path.stem}_{name}.json").write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8")
    return frame, full, filtered, audit
