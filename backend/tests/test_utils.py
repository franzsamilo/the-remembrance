"""Utility and unit tests."""
import pytest


def test_format_auc_roc_logic():
    """Format AUC-ROC logic: 3 decimals or N/A for null."""
    def format_auc_roc(value):
        if value is None:
            return "N/A"
        return f"{float(value):.3f}"
    assert format_auc_roc(0.95) == "0.950"
    assert format_auc_roc(None) == "N/A"
    assert format_auc_roc(0) == "0.000"
