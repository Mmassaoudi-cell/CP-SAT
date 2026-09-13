# Data Audit

## Source-paper data availability

No author code, supplemental file, public workload trace, market trace, CEF trace, or named simulator was found in the supplied paper, the IEEE DOI record, or public code-oriented searches performed on 2026-09-01. Exact numerical reproduction is therefore impossible from released materials alone.

## Reproduction data decision

The source benchmark will use a frozen synthetic reconstruction matching the paper's stated protocol:

- three DCs;
- 96 quarter-hour periods;
- 10 workloads and 50 subtasks;
- dependency types limited to valid DAGs (pipeline, fork-join, tree, shared-predecessor);
- three regional price/CEF traces matching the qualitative ranges and opposing daily patterns described in the paper;
- fixed train/validation/test scenario seeds;
- no parameter chosen after test evaluation.

The reconstruction is explicitly labeled synthetic and must not be presented as author data.

## Local data inventory

`C:\Users\MMASSAOUDI\Desktop\Data` contains 1,331 files (approximately 71.3 GB). Most top-level collections are intrusion-detection/network-traffic datasets and are unrelated to the present scheduling problem.

The relevant collection is `Load Data`, especially:

- **GEFCom2014 price:** hourly zonal price with forecasted total/zonal load.
- **GEFCom2014 solar:** hourly normalized solar power for multiple zones.
- **GEFCom2014 wind:** hourly normalized wind power and 10 m/100 m wind components for 10 zones.
- **GEFCom2014 load:** hourly zonal load and weather features.
- **PJM load:** hourly system load.
- **ISO-NE load:** hourly load and temperature.
- **Pecan Street:** 15-minute residential power data.

## Selection

- **Source reproduction:** synthetic paper-like scenarios, because the source traces are unavailable.
- **External validation:** GEFCom2014 price, solar, and wind traces, selected before model testing because they jointly provide the market and renewable heterogeneity required by the problem.
- **Not selected:** cybersecurity datasets, residential-only Pecan Street data, and building-load forecasting code, because they do not provide dependency-aware DC workloads or coupled regional market signals.

## Leakage controls

- Scenario-level splits are used for the reconstructed simulator.
- GEFCom external validation will use chronological blocks, with preprocessing fitted on the training block only.
- Workload DAG templates and all scenario-generation distributions are frozen before final testing.
- Test scenario identifiers are listed in `DATA_SPLIT_MANIFEST.csv` but will not be evaluated during candidate screening.
- No exact or near-duplicate generated scenario can cross splits because each manifest row has a unique fixed seed and independent trace noise.

## Remaining limitations

- No real cloud-workflow DAG trace is available locally.
- GEFCom dates and zones differ across the price, solar, and wind tracks; external-validation alignment therefore requires a documented normalization/sampling layer rather than pretending the tracks are contemporaneous observations.
- CEFs are not present and must be derived from renewable share using a declared emissions-factor model or obtained from another authoritative source.
