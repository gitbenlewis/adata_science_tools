"""Helpers to plot GSEApy dotplots from saved result tables."""
# version 1 2026-01-14
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import gseapy as gp
import pandas as pd


def _add_gseapy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if "FDR q-val" not in df.columns and "fdr" in df.columns:
        df["FDR q-val"] = df["fdr"]
    if "Adjusted P-value" not in df.columns:
        if "FDR q-val" in df.columns:
            df["Adjusted P-value"] = df["FDR q-val"]
        elif "fdr" in df.columns:
            df["Adjusted P-value"] = df["fdr"]
    if "P-value" not in df.columns and "pval" in df.columns:
        df["P-value"] = df["pval"]
    if "NES" not in df.columns and "nes" in df.columns:
        df["NES"] = df["nes"]
    if "ES" not in df.columns and "es" in df.columns:
        df["ES"] = df["es"]
    if "Tag %" not in df.columns and "tag %" in df.columns:
        df["Tag %"] = df["tag %"]
    return df


def _apply_pvalue_zero_floor(
    df: pd.DataFrame,
    pvalue_col: str,
    n_perm_4_zero_pavlues: Optional[int],
) -> pd.DataFrame:
    if pvalue_col not in df.columns:
        raise ValueError(f"pvalue_col '{pvalue_col}' not found in table columns.")
    series = pd.to_numeric(df[pvalue_col], errors="coerce")
    zero_mask = series == 0
    if not zero_mask.any():
        return df
    if n_perm_4_zero_pavlues is not None:
        if n_perm_4_zero_pavlues <= 0:
            raise ValueError("n_perm_4_zero_pavlues must be > 0.")
        floor = 1.0 / float(n_perm_4_zero_pavlues)
    else:
        nonzero = series[series > 0]
        if nonzero.empty:
            raise ValueError(
                f"{pvalue_col} has only zeros; set n_perm_4_zero_pavlues to use 1/N."
            )
        floor = nonzero.min() / 2.0
    df = df.copy()
    df[pvalue_col] = series.mask(zero_mask, floor)
    return df


def load_gseapy_table(
    table_path: str | Path,
    sep: str = ",",
) -> pd.DataFrame:
    """Load a GSEApy output table for dotplotting."""
    return pd.read_csv(table_path, sep=sep)


def load_gseapy_tables(
    table_paths: Iterable[str | Path | pd.DataFrame],
    labels: Optional[Sequence[str]] = None,
    x_label: str = "Dataset",
    sep: str = ",",
) -> pd.DataFrame:
    """Load multiple GSEApy tables and annotate them with a dataset column."""
    tables = list(table_paths)
    paths = [Path(p) for p in tables if not isinstance(p, pd.DataFrame)]
    if labels is None:
        labels = []
        for idx, item in enumerate(tables):
            if isinstance(item, pd.DataFrame):
                labels.append(f"table_{idx}")
            else:
                labels.append(Path(item).stem)
    if len(labels) != len(tables):
        raise ValueError("labels length must match table_paths length")
    frames = []
    for item, label in zip(tables, labels):
        if isinstance(item, pd.DataFrame):
            df = item.copy()
        else:
            df = pd.read_csv(item, sep=sep)
        df = df.assign(**{x_label: label})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dotplot_from_table(
    table_path: str | Path | pd.DataFrame,
    sep: str = ",",
    term_label_char_limit: Optional[int] = None,
    term_label_col: Optional[str] = None,
    ordered_terms_list: Optional[Sequence[str]] = None,
    pvalue_col: str = "fdr",
    n_perm_4_zero_pavlues: Optional[int] = 10000,
    **dotplot_kwargs,
):
    """Plot a GSEApy dotplot from a single saved table or DataFrame."""
    if isinstance(table_path, pd.DataFrame):
        df = table_path.copy()
    else:
        df = load_gseapy_table(table_path, sep=sep)
    df = _add_gseapy_aliases(df)
    df = _apply_pvalue_zero_floor(df, pvalue_col, n_perm_4_zero_pavlues)
    if pvalue_col == "fdr":
        if "FDR q-val" in df.columns:
            df["FDR q-val"] = df["fdr"]
        if "Adjusted P-value" in df.columns:
            df["Adjusted P-value"] = df["fdr"]
    elif pvalue_col == "FDR q-val" and "Adjusted P-value" in df.columns:
        df["Adjusted P-value"] = df["FDR q-val"]
    y_col = term_label_col or dotplot_kwargs.get("y", "Term")
    ordered_terms = None
    if ordered_terms_list is not None:
        if y_col not in df.columns:
            raise ValueError(f"ordered_terms_list requires '{y_col}' column.")
        ordered_terms = [str(term) for term in ordered_terms_list]
        present_terms = set(df[y_col].astype(str))
        ordered_terms = [term for term in ordered_terms if term in present_terms]
        if not ordered_terms:
            raise ValueError("ordered_terms_list had no matches in the table.")
        df = df[df[y_col].astype(str).isin(ordered_terms)]
    if term_label_char_limit is not None and y_col in df.columns:
        df[y_col] = df[y_col].astype(str).str.slice(0, int(term_label_char_limit))
        if ordered_terms is not None:
            truncated = []
            seen = set()
            limit = int(term_label_char_limit)
            for term in ordered_terms:
                tval = str(term)[:limit]
                if tval not in seen:
                    truncated.append(tval)
                    seen.add(tval)
            ordered_terms = truncated
    if ordered_terms is not None:
        if "y_order" not in dotplot_kwargs:
            dotplot_kwargs["y_order"] = ordered_terms
        if "top_term" not in dotplot_kwargs:
            dotplot_kwargs["top_term"] = len(ordered_terms)
    if "column" not in dotplot_kwargs:
        dotplot_kwargs["column"] = "FDR q-val" if "FDR q-val" in df.columns else pvalue_col
    dotplot_kwargs.setdefault("top_term", 15)
    dotplot_kwargs.setdefault("cutoff", 1.0)
    return gp.dotplot(df, **dotplot_kwargs)


def dotplot_from_tables(
    table_paths: Iterable[str | Path],
    labels: Optional[Sequence[str]] = None,
    x_label: str = "Dataset",
    sep: str = ",",
    term_label_char_limit: Optional[int] = None,
    term_label_col: Optional[str] = None,
    ordered_terms_list: Optional[Sequence[str]] = None,
    pvalue_col: str = "fdr",
    n_perm_4_zero_pavlues: Optional[int] = 10000,
    **dotplot_kwargs,
):
    """Plot a GSEApy dotplot across multiple saved tables."""
    df = load_gseapy_tables(
        table_paths,
        labels=labels,
        x_label=x_label,
        sep=sep,
    )
    df = _add_gseapy_aliases(df)
    df = _apply_pvalue_zero_floor(df, pvalue_col, n_perm_4_zero_pavlues)
    if pvalue_col == "fdr":
        if "FDR q-val" in df.columns:
            df["FDR q-val"] = df["fdr"]
        if "Adjusted P-value" in df.columns:
            df["Adjusted P-value"] = df["fdr"]
    elif pvalue_col == "FDR q-val" and "Adjusted P-value" in df.columns:
        df["Adjusted P-value"] = df["FDR q-val"]
    y_col = term_label_col or dotplot_kwargs.get("y", "Term")
    ordered_terms = None
    if ordered_terms_list is not None:
        if y_col not in df.columns:
            raise ValueError(f"ordered_terms_list requires '{y_col}' column.")
        ordered_terms = [str(term) for term in ordered_terms_list]
        present_terms = set(df[y_col].astype(str))
        ordered_terms = [term for term in ordered_terms if term in present_terms]
        if not ordered_terms:
            raise ValueError("ordered_terms_list had no matches in the tables.")
        df = df[df[y_col].astype(str).isin(ordered_terms)]
    if term_label_char_limit is not None and y_col in df.columns:
        df[y_col] = df[y_col].astype(str).str.slice(0, int(term_label_char_limit))
        if ordered_terms is not None:
            truncated = []
            seen = set()
            limit = int(term_label_char_limit)
            for term in ordered_terms:
                tval = str(term)[:limit]
                if tval not in seen:
                    truncated.append(tval)
                    seen.add(tval)
            ordered_terms = truncated
    if ordered_terms is not None:
        if "y_order" not in dotplot_kwargs:
            dotplot_kwargs["y_order"] = ordered_terms
        if "top_term" not in dotplot_kwargs:
            dotplot_kwargs["top_term"] = len(ordered_terms)
    if "column" not in dotplot_kwargs:
        dotplot_kwargs["column"] = "FDR q-val" if "FDR q-val" in df.columns else pvalue_col
    dotplot_kwargs.setdefault("top_term", 15)
    dotplot_kwargs.setdefault("cutoff", 1.0)
    dotplot_kwargs = {"x": x_label, **dotplot_kwargs}
    return gp.dotplot(df, **dotplot_kwargs)


__all__ = [
    "load_gseapy_table",
    "load_gseapy_tables",
    "dotplot_from_table",
    "dotplot_from_tables",
]
