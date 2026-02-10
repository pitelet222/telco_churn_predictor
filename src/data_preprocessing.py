"""
Shared preprocessing for Telco Churn models.

This module mirrors the exact data cleaning and feature engineering steps
from notebooks/02_data_cleaning.ipynb so training and inference stay aligned.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


SERVICE_COLS = [
	"MultipleLines",
	"OnlineSecurity",
	"OnlineBackup",
	"DeviceProtection",
	"TechSupport",
	"StreamingTV",
	"StreamingMovies",
]

CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]


def _normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
	obj_cols = df.select_dtypes(include=["object", "string"]).columns
	if not obj_cols.empty:
		df = df.copy()
		df[obj_cols] = df[obj_cols].apply(lambda col: col.astype(str).str.strip().str.lower())
	return df


def _convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
	if "TotalCharges" in df.columns:
		df = df.copy()
		df["TotalCharges"] = pd.to_numeric(
			df["TotalCharges"].astype(str).str.strip(), errors="coerce"
		)
		df["TotalCharges"] = df["TotalCharges"].fillna(0)
	return df


def _encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
	mapping = {"yes": 1, "no": 0}
	df = df.copy()
	binary_cols = [
		col
		for col in df.columns
		if df[col].dtype in ["object", "string"]
		and set(df[col].dropna().unique()) == {"yes", "no"}
	]
	if binary_cols:
		df[binary_cols] = df[binary_cols].replace(mapping)
	return df


def _encode_services(df: pd.DataFrame) -> pd.DataFrame:
	cols = [col for col in SERVICE_COLS if col in df.columns]
	if not cols:
		return df
	return pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if "gender" in df.columns:
		df["gender"] = df["gender"].map({"female": 0, "male": 1})

	cols = [col for col in CATEGORICAL_COLS if col in df.columns]
	if cols:
		df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
	return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if "TotalCharges" in df.columns and "tenure" in df.columns:
		df["AvgMonthlyCharges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

	if "tenure" in df.columns:
		df["TenureGroup"] = pd.cut(
			df["tenure"],
			bins=[0, 12, 24, 48, 72],
			labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
		)
		df = pd.get_dummies(df, columns=["TenureGroup"], drop_first=True, dtype=int)

	service_features = [
		col
		for col in df.columns
		if any(
			token in col
			for token in [
				"OnlineSecurity",
				"OnlineBackup",
				"DeviceProtection",
				"TechSupport",
				"StreamingTV",
				"StreamingMovies",
				"MultipleLines",
			]
		)
	]
	yes_columns = [col for col in service_features if col.endswith("yes")]
	if yes_columns:
		df["TotalServices"] = df[yes_columns].sum(axis=1)
	return df


def preprocess_dataframe(
	df: pd.DataFrame,
	*,
	drop_customer_id: bool = True,
	expected_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
	"""Apply the full preprocessing pipeline used for training.

	Parameters
	----------
	df:
		Raw input DataFrame.
	drop_customer_id:
		Drop the `customerID` column if present.
	expected_columns:
		If provided, reindex to these columns (missing columns filled with 0).
	"""
	df = _convert_total_charges(df)
	df = _normalize_string_columns(df)
	df = _encode_binary_yes_no(df)
	df = _encode_services(df)
	df = _encode_categoricals(df)
	df = _engineer_features(df)

	if drop_customer_id and "customerID" in df.columns:
		df = df.drop(columns=["customerID"])

	if expected_columns is not None:
		df = df.reindex(columns=list(expected_columns), fill_value=0)

	return df


def preprocess_single_record(
	record: dict,
	*,
	expected_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
	"""Convenience wrapper for preprocessing a single input record."""
	df = pd.DataFrame([record])
	return preprocess_dataframe(df, expected_columns=expected_columns)
