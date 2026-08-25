from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_INVALID_DATE_RATIO = 0.1


@dataclass(frozen=True)
class CexExportSpec:
    provider_name: str
    required_columns: frozenset[str]
    optional_columns: tuple[str, ...] = ()
    fingerprint_columns: tuple[str, ...] = ()
    identity_column: str | None = None


def _csv_paths(input_csv: Path) -> list[Path]:
    if input_csv.is_file():
        return [input_csv]
    return sorted(path for path in input_csv.glob("*.csv") if path.is_file())


def load_cex_exports(*, input_csv: Path, spec: CexExportSpec) -> pd.DataFrame:
    csv_paths = _csv_paths(input_csv=input_csv)
    if not csv_paths:
        raise FileNotFoundError(
            f"No {spec.provider_name} transaction CSV files found in {input_csv}"
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = spec.required_columns.difference(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Unsupported {spec.provider_name} CSV schema in {csv_path.name}; "
                f"missing columns: {missing_text}."
            )
        for column in spec.optional_columns:
            if column not in frame.columns:
                frame[column] = ""
        normalized_columns = {
            *spec.required_columns,
            *spec.optional_columns,
            *spec.fingerprint_columns,
        }
        for column in normalized_columns:
            frame[column] = frame[column].map(lambda value: str(value).strip())
        frame["__source_file"] = csv_path.name
        frame["__source_row"] = range(len(frame))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if not spec.fingerprint_columns:
        return combined

    combined["__row_fingerprint"] = combined[list(spec.fingerprint_columns)].agg(
        "\x1f".join,
        axis=1,
    )
    combined["__identity_key"] = combined["__row_fingerprint"]
    if spec.identity_column:
        has_identity = combined[spec.identity_column].ne("")
        combined.loc[has_identity, "__identity_key"] = (
            combined.loc[has_identity, "__row_fingerprint"]
            + "\x1fidentity="
            + combined.loc[has_identity, spec.identity_column]
        )
    combined["__occurrence"] = combined.groupby(
        ["__source_file", "__identity_key"],
        sort=False,
    ).cumcount()
    combined = combined.sort_values(
        ["__source_file", "__source_row"],
        ascending=[True, True],
    )
    combined = combined.drop_duplicates(
        subset=["__identity_key", "__occurrence"],
        keep="first",
    )
    return combined.drop(
        columns=["__row_fingerprint", "__identity_key", "__occurrence"]
    ).reset_index(drop=True)


def prepare_cex_dates(
    *,
    frame: pd.DataFrame,
    parsed_dates: pd.Series,
    provider_name: str,
    source_column: str,
) -> pd.DataFrame:
    invalid_date_count = int(parsed_dates.isna().sum())
    total_rows = len(frame)
    if total_rows and (invalid_date_count / total_rows) > MAX_INVALID_DATE_RATIO:
        raise ValueError(
            f"Aborting {provider_name} snapshot generation: invalid dates="
            f"{invalid_date_count}/{total_rows} ({invalid_date_count / total_rows:.1%})."
        )
    if invalid_date_count:
        print(
            f"[{provider_name}] Dropping {invalid_date_count} rows with invalid "
            f"{source_column} values."
        )

    prepared = frame.copy()
    prepared["Date"] = parsed_dates
    prepared = prepared.dropna(subset=["Date"])
    return prepared.sort_values(
        ["Date", "__source_file", "__source_row"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
