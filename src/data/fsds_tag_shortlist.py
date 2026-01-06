"""
FSDS tag shortlist configuration for panel construction.
"""

from __future__ import annotations

# Provide the default FSDS tag shortlist for panel construction.
def get_default_fsds_tag_shortlist() -> list[str]:
    return [
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "Revenues",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "NetCashProvidedByUsedInOperatingActivities",
        "DebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "InterestExpense",
    ]


# Provide a set version of the default tag shortlist for fast membership checks.
def get_default_fsds_tag_shortlist_set() -> set[str]:
    return set(get_default_fsds_tag_shortlist())
