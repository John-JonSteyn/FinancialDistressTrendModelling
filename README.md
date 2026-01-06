# Financial Distress Trend Modelling

*A longitudinal, time-aware framework for predicting financial distress trajectories using SEC XBRL filings.*

<p align="center">
  <a href="https://github.com/John-JonSteyn/FinancialDistressTrendModelling/stargazers" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/stars/John-JonSteyn/FinancialDistressTrendModelling?style=for-the-badge&color=4C6FAF" alt="GitHub stars" />
  </a>
  <a href="https://github.com/John-JonSteyn/FinancialDistressTrendModelling" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/repo-size/John-JonSteyn/FinancialDistressTrendModelling?style=for-the-badge&color=4C6FAF" alt="Repo size" />
  </a>
  <a href="https://github.com/John-JonSteyn/FinancialDistressTrendModelling/commits/main" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/last-commit/John-JonSteyn/FinancialDistressTrendModelling?style=for-the-badge&color=4C6FAF" alt="Last commit" />
  </a>
  <a href="https://github.com/John-JonSteyn/FinancialDistressTrendModelling/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/license/John-JonSteyn/FinancialDistressTrendModelling?style=for-the-badge&color=4C6FAF" alt="License" />
  </a>
</p>

---

## Overview

This repository implements an **end-to-end financial distress trend modelling pipeline** using publicly available SEC XBRL filings.

Rather than predicting bankruptcy as a single binary event, the system focuses on **early-warning distress signals** by modelling **financial deterioration trajectories over time**.
The output is a probability-based assessment designed to support monitoring, triage, and intervention decisions rather than deterministic classification.

The project combines:

* Longitudinal financial statement feature engineering
* Time-aware model training and evaluation
* Live scoring from SEC company facts
* Automated, presentation-ready company reports

---

## Objectives

* Move beyond point-in-time credit risk assessment toward **temporal distress modelling**
* Engineer interpretable financial ratio, trend, and slope features from SEC filings
* Enforce strict **time-based evaluation** to avoid look-ahead bias
* Produce explainable, company-level reports suitable for professional review
* Demonstrate how academic models can be deployed against live regulatory data

---

## Example Outputs

Below are automatically generated company-level distress reports using the most recent available SEC filings.

### Stable example

<p align="center">
  <img src="docs/examples/company_report_stable.png" width="720"/>
</p>

### Recovery example

<p align="center">
  <img src="docs/examples/company_report_recovery.png" width="720"/>
</p>

### Elevated risk example

<p align="center">
  <img src="docs/examples/company_report_elevated_risk.png" width="720"/>
</p>

### High distress example

<p align="center">
  <img src="docs/examples/company_report_high_distress.png" width="720"/>
</p>

Each report includes:

* Recent distress probability history
* Filing-aware timestamps
* Feature coverage diagnostics
* Permutation-based feature importance
* Visual stability indicators

---

## Methodology Summary

### Data Source

* SEC XBRL *company facts* API
* Quarterly and annual filings (10-Q / 10-K)
* No proprietary or forward-looking information

### Feature Engineering

* Core financial statement values (assets, liabilities, income, cash flow, debt)
* Financial ratios (leverage, margins, liquidity, coverage)
* Temporal dynamics:

  * Quarter-on-quarter changes
  * Year-on-year changes
  * Rolling means
  * Short-window slopes

### Modelling

* Gradient boosting classifier
* Trained on historical firm trajectories
* Evaluated using **time-holdout splits** and **unseen-company splits**
* Outputs probabilistic distress scores rather than hard predictions

### Scoring and Reporting

* Live SEC integration for unseen firms
* Latest N-quarter scoring window
* Automated CSV, JSON, and PNG report generation

---

## Repository Structure

```
data/
├─ raw/                         # Raw extracted SEC data
├─ processed/                   # Modelling tables and derived datasets
outputs/
├─ models/                      # Trained model pipelines
├─ tables/                      # Score histories, summaries, coverage tables
└─ figures/                     # Generated plots and reports
src/
├─ modelling/                   # Feature engineering and training code
├─ reporting/                   # SEC scoring and report generation
└─ utils/
docs/
└─ examples/                    # Example company reports
```

---

## Reproducibility Notes

* All feature engineering mirrors the training pipeline exactly during scoring
* Time-based splits prevent information leakage
* Randomised demo selection is deterministic via seed control
* Reports can be regenerated entirely from SEC data and saved models

---

## Important Notes and Limitations

* This system does **not** predict bankruptcy events
* Outputs are **early-warning indicators**, not deterministic outcomes
* SEC filings may be incomplete, delayed, or amended
* Results depend on reporting frequency and accounting conventions
* This project is for **academic and research purposes only**

It is **not** financial, investment, or legal advice.

---

## Purpose

This project demonstrates how machine learning models can be responsibly applied to longitudinal financial data while respecting temporal causality, interpretability, and real-world deployment constraints.

It is intended as:

* An academic exploration of distress trend modelling
* A methodological reference for time-aware financial ML
* A foundation for further research into early-warning systems

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
You are genuinely at a strong stopping point now.