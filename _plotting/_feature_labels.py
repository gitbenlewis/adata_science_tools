"""Internal display-only feature label resolution."""

import pandas as pd


def _resolve_feature_labels(
    table: pd.DataFrame,
    preferred_column: str | None,
    fallback_column: str | None,
) -> pd.Series | None:
    """Resolve labels without changing identities, order, or the source table.

    None leaves the caller's legacy label path untouched. If both values are
    unavailable, retain the original preferred value so the caller's existing
    missing-label policy still applies. A missing/disabled preferred column
    likewise retains the caller's existing index/error behavior.
    """
    if fallback_column is None:
        return None
    if fallback_column not in table.columns:
        raise ValueError(
            f"feature_label_fallback column {fallback_column!r} not found "
            "in the feature-label table."
        )
    if preferred_column is None or preferred_column not in table.columns:
        return None
    preferred = table[preferred_column]
    fallback = table[fallback_column]
    # Normalize only the missingness mask, never valid display text.
    preferred_text = preferred.astype("string").str.strip().str.casefold()
    fallback_text = fallback.astype("string").str.strip().str.casefold()
    missing = preferred.isna() | preferred_text.isin(["", "nan"])
    available = fallback.notna() & ~fallback_text.isin(["", "nan"])
    return preferred.astype(object).where(~(missing & available), fallback)
