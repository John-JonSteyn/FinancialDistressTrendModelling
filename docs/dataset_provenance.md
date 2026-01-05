# Dataset Provenance

## Overview

This project uses the **SEC Financial Statement Data Sets (FSDS)** as its sole primary data source.  
The FSDS are publicly released, regulator-supplied datasets published by the United States Securities and Exchange Commission (SEC), containing standardised financial statement information extracted from corporate filings.

The dataset is used to construct a longitudinal, company-period panel for modelling financial condition deterioration over time.

---

## Data Source

**Provider**  
United States Securities and Exchange Commission (SEC)

**Dataset Name**  
Financial Statement Data Sets (FSDS)

**Source Page**  
SEC Financial Statement Data Sets (Quarterly)

**Original Data Origin**  
Public company filings submitted to the SEC via the EDGAR system, primarily:
- Form 10-K (annual reports)
- Form 10-Q (quarterly reports)

---

## Dataset Contents

Each quarterly FSDS release consists of multiple tab-delimited text files.  
The following files are used in this project:

- `sub.txt`  
  Filing-level metadata, including company identifiers, form type, filing dates, and fiscal period information.

- `num.txt`  
  Numeric financial statement values reported in filings, indexed by XBRL tags.

- `tag.txt`  
  Definitions and metadata for XBRL tags used in `num.txt`.

Other files present in the FSDS distribution are not used.

---

## Temporal Coverage

**Selected Horizon**  
To be finalised prior to modelling. The intended horizon is approximately **3 to 5 years of quarterly data**, subject to data availability and completeness.

**Granularity**  
Quarterly, based on fiscal period end dates rather than filing dates.

---

## Acquisition Details

**Download Method**  
Manual download of quarterly ZIP archives from the SEC FSDS webpage.

**Storage Location (Local)**  
- Raw ZIP files:  
  `data/raw/fsds_zip/`

**Download Dates**  
To be recorded at time of acquisition.

**Files Downloaded**  
To be enumerated once acquisition is complete, for example:

- `2021q1.zip`
- `2021q2.zip`
- `2021q3.zip`
- `2021q4.zip`
- `2022q1.zip`
- `...`

---

## Data Integrity and Reproducibility

- Raw FSDS ZIP archives are **not modified** after download.
- Raw data files are **not committed to version control** due to size constraints.
- All data transformations are performed via versioned scripts within the repository.
- Intermediate and processed datasets are derived deterministically from the recorded FSDS releases.

This provenance document, together with the ingestion scripts and transformation logs, enables full reconstruction of the modelling dataset from the original SEC source.

---

## Licensing and Usage

The SEC Financial Statement Data Sets are publicly available and may be used for research and analytical purposes.  
Users of this repository are responsible for ensuring compliance with SEC terms of use and any applicable downstream requirements.

---

## Notes and Limitations

- Financial statement values are reported as filed and may contain restatements, reporting inconsistencies, or firm-specific accounting choices.
- Amended filings (for example 10-K/A and 10-Q/A) are excluded during data processing unless explicitly stated otherwise.
- The dataset does not contain explicit bankruptcy or default labels. Any distress-related labels used in this project are constructed proxies defined separately.

Refer to `label_definition.md` for details on distress proxy construction.
