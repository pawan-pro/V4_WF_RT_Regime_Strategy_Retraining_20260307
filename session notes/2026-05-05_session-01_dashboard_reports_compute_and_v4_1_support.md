# 2026-05-05 Session 01 — Dashboard PDF Recovery, Avik Access, V4.1 Support Package

## Session objective

Start the 2026-05-05 session by confirming the V4 production/dashboard state, fixing the daily PDF report failures, enabling Avik to access dashboard outputs safely, and generating the V4.1 presentation support package.

Production context remained the V4 Databricks stack under:

```text
pawan_quant_research_1.v4
```

Safety posture throughout the session:

- Do not change Saxo live execution.
- Do not change Saxo token refresh.
- Do not change model inference or monthly retraining.
- Do not modify live trading schedules or RT feature/signal generation.
- Keep Databricks Git state, Databricks Repo metadata, and actual workspace notebook state separate.

---

## 1. Starting agenda

The working agenda was:

1. Give Avik dashboard access so he can run/trigger reports himself.
2. Ensure `04_Saxo_Institutional_Dashboard` can generate PDFs with all charts/dataframes.
3. Keep automated 06:00 SGT daily report generation alive and prepare for SharePoint storage once Avik/Pawan provide folder/auth details.
4. Update presentation numbers using the V4.1/B-C2 support package.
5. Share supporting files to close the V4 package.

Initial status at session start:

| Workstream | Status |
|---|---|
| Dashboard PDF crash | Open |
| PDF formatting | Needs review |
| Avik job-trigger PDF access | Needs implementation |
| Shared dashboard compute | Not yet created |
| V4.1 support CSVs | Needed |
| Deck update | Waiting for support files / deck file |
| SharePoint upload | Waiting for SharePoint folder/auth |

---

## 2. Dashboard / PDF failure diagnosis

The current dashboard notebook was:

```text
02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb
```

Actual workspace path used by the jobs:

```text
/Workspace/Users/pawan.jatale@mubama.group/V4_WF_RT_Regime_Strategy_Retraining_github/02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard
```

Reported Databricks error in upcoming-events section:

```text
ValueError: can not infer schema from empty dataset
```

Root cause:

- The notebook created an empty pandas DataFrame for the no-upcoming-events case.
- Databricks `display()` internally attempted `spark.createDataFrame(empty_pandas_df)`.
- Spark could not infer schema from an empty dataset.

Safe fix:

- Replace empty pandas display path with a deterministic non-empty placeholder row or a message panel.
- Preserve report-only/read-only behavior.

---

## 3. MUB-20 — Dashboard execution failures and Avik access

A Linear issue was created and completed:

```text
MUB-20 — Fix 04_Saxo_Institutional_Dashboard execution failures and Avik access
```

Reported completion details:

```text
Branch: fix/mub-20-dashboard-empty-events-and-avik-access
Commit: 34094c8f6d954ba6968e9c6396aad4b04b0e4b56
```

Files changed:

```text
02_Analysis/02_Dashboard/04_Saxo_Institutional_Dashboard.ipynb
AGENTS.md
05_Context/04_Codex_Project_Memory.md
```

Workspace notebook hash:

```text
before: 971d09c727dec1863b923713dd029c0c22dfc1ec5220dc396fbede63263ee40f
after:  cf20877cbb95323605c7131509b07f9f21c060ba10ccfeda2b20bcbfb0236075
```

Validation runs:

```text
Interactive submit run: 1050681692173429 — SUCCESS
Daily dashboard job 516523889858332, run 35479693038800, task 1048455091836773 — SUCCESS
Circulation job 559657131456932, run 141917130861700, task 740411663023702 — SUCCESS
```

PDF outputs:

```text
/Volumes/pawan_quant_research_1/v4/market_data/reports/institutional_dashboard/saxo_institutional_dashboard_2026-05-05_SGT.pdf
/Volumes/pawan_quant_research_1/v4/market_data/reports/institutional_dashboard/circulation_report/saxo_institutional_circulation_2026-05-05_SGT.pdf
```

Both PDFs were reported readable PDF 1.4 files, 13 pages each.

Avik access decision:

- Implemented Option A: Avik has job-level `CAN_MANAGE_RUN` on both dashboard PDF jobs.
- Pawan's single-user production cluster `0311-005159-pm2x3pce` was not changed.
- Avik should trigger jobs, not run directly on Pawan's single-user cluster.

Sync note:

- Local Git branch had the MUB-20 fix.
- Databricks Repo metadata still showed `feat/live-macro-phase1-pce-gdp-overrides` at commit `f8613656039839e01cdf7e8947def9c0ccc7823e`.
- Actual workspace notebook was intentionally imported directly and updated for the PDF jobs.

---

## 4. MUB-21 — V4.1 presentation support CSV package

A Linear issue was created and reported functionally complete:

```text
MUB-21 — Generate V4.1 presentation support CSVs and update deck numbers
```

Output folder:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/presentation_support_2026-05-05/
```

Primary input:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/performance_methodology_reconciliation_2026-04-30/performance/best_match_hourly_equity_curve.csv
```

Input hash:

```text
2ba607bc0f118fdc1d606198b85f7f2170cb8c3034fc145580547b32d59c3582
```

Generated files:

```text
daily_performance.csv
weekly_performance.csv
monthly_performance.csv
quarterly_performance.csv
half_yearly_performance.csv
yearly_performance.csv
trades.csv
summary_metrics.csv
presentation_numbers.csv
metric_reconciliation.csv
source_artifact_inventory.csv
signal_files/full_signal_export.csv
signal_files/teacher_pseudo_labels.csv
signal_files/walkforward_student_signals.csv
signal_files/walkforward_signals.csv
```

Metric reconciliation:

| Metric | Value | Target | Status |
|---|---:|---:|---|
| CAGR | 20.840487% | 20.840487% | pass |
| Sharpe | 1.002359 | 1.002359 | pass |
| Sortino | 1.372906 | 1.372906 | pass |
| Max DD | -25.538514% | -25.538514% | pass |
| Final equity | $12,883,454.99 | $12,883,454.99 | pass |
| Switches | 296 | 296 | pass |

Deck-ready values:

- B/C2 baseline: CAGR 20.8%, Sharpe 1.00, Sortino 1.37, Max DD -25.5%, Final Equity $12.88M
- SPY benchmark: CAGR 12.8%, Sharpe 0.77, Max DD -25.2%, Final Equity $9.11M
- C0 current: CAGR 14.6%, Sharpe 0.80, Max DD -24.1%, Final Equity $9.89M
- D strict PIT teacher-student: CAGR 8.5%, Sharpe 0.51, Max DD -26.3%, Final Equity $7.51M

Support script reported:

```text
03_Research/03_Support/01_V4_1_Presentation_Support_CSVs.py
```

Deck status:

- No PPT/deck source file was available in repo during this workstream.
- Presentation numbers were ready in `presentation_numbers.csv`.
- Production V4 trading/model/workflow objects were not changed.

---

## 5. MUB-22 — PDF formatting polish

A new issue was created after reviewing the generated PDFs:

```text
MUB-22 — Polish Saxo institutional dashboard PDF formatting after MUB-20
```

Observed formatting issues included:

- Page 1 KPI cards were crowded.
- Upcoming-events placeholder was too wide.
- Macro inputs source/timing table had overlapping source/availability text.
- Current feature state page was dense.
- Source timestamp table names wrapped badly.

Reported completion details:

```text
Branch: fix/mub-22-dashboard-pdf-formatting
Commit: fe370c51eb4d3c66fad3805125757ccf52fc4a12
```

Validation:

```text
Overall PDF job 516523889858332, run 185346942484721, task 634731992074577 — SUCCESS
Circulation PDF job 559657131456932, run 609825653847638, task 1065890489371261 — SUCCESS
```

PDF outputs:

```text
/Volumes/pawan_quant_research_1/v4/market_data/reports/institutional_dashboard/saxo_institutional_dashboard_2026-05-05_SGT.pdf
/Volumes/pawan_quant_research_1/v4/market_data/reports/institutional_dashboard/circulation_report/saxo_institutional_circulation_2026-05-05_SGT.pdf
```

Visual QA reported:

- Page 1 KPI grid reduced.
- No-events section became a clean message panel.
- Empty catalyst-detail page suppressed in circulation/no-event case.
- Macro input pages split into readable model feature / source detail pages.
- Source timestamp page used aliases such as `Dashboard feed`, `RT silver`, `Live macro events`, `Model registry`.

---

## 6. MUB-23 — Shared dashboard compute path

A new issue was created:

```text
MUB-23 — Create dashboard-only compute path for Avik and Pawan
```

Problem:

- `0311-005159-pm2x3pce` is a single-user cluster assigned to `pawan.jatale@mubama.group`.
- Avik cannot run notebooks on it.

Implemented solution:

```text
v4-dashboard-shared-compute
Cluster id: 0505-040158-nnjdzqxa
Access mode: USER_ISOLATION
Runtime: 18.0.x-scala2.13
Node: single-node i3.xlarge
Autotermination: 120 minutes
```

Permissions:

```text
Avik: CAN_RESTART
Pawan: CAN_MANAGE
```

Validation:

- Dashboard notebook ran successfully on the new shared cluster.
- Read validation succeeded for key dashboard/model tables.
- Notebook-scoped installs for `joblib` and `fredapi` worked.

Production safety:

- Production cluster `0311-005159-pm2x3pce` remained unchanged.
- Existing jobs continued to use existing compute.

---

## 7. Open items at end of 2026-05-05 session

1. SharePoint upload remained blocked on folder/auth route.
2. Deck update remained dependent on PPT file availability.
3. PDF formatting had improved but required continued final stakeholder polish in subsequent sessions.
4. Branch pushes for later local commits should be verified, because some reported local SHAs were not immediately visible through the GitHub connector.

## Final state

| Workstream | End-of-session status |
|---|---|
| Dashboard PDF crash | Done |
| Daily dashboard PDF | Running |
| Circulation PDF | Running |
| Avik PDF job access | Done |
| Shared dashboard compute | Created and validated |
| V4.1 support CSVs | Done |
| Deck update | Waiting for deck file |
| SharePoint upload | Waiting for SharePoint folder/auth |

## Production safety confirmation

No Saxo live execution, token refresh, model inference, monthly retraining, live trading schedules, RT feature/signal generation, or production job/cluster definitions were changed during the reported session work.
