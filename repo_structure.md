/Workspace/Users/pawan.jatale@mubama.group/V4_Regime_Strategy/
├── 01_Backtesting/                    (existing notebooks 01-10)
├── 02_RealTime/
│   ├── RT_01_Live_Market_Ingestion.py       # Alpaca WebSocket → Bronze RT
│   ├── RT_02_Live_Rates_VIX_Ingestion.py    # Databento/Polygon → Rates + VIX
│   ├── RT_03_Silver_RT_Enrichment.py        # Replicate Silver Grid on live bar close
│   ├── RT_04_Model_Inference_Engine.py      # Load joblib → predict regime → emit signal
│   ├── RT_05_Paper_Trade_Execution.py       # Alpaca Paper → place/cancel orders
│   ├── RT_06_Signal_Audit_Log.py            # Append every signal + portfolio state to Delta
│   └── RT_07_Dashboard_Feed.py             # Write Power BI-ready Delta tables
├── 03_Dashboard/
│   └── PowerBI_Template.pbix               # (manual step — see Phase 4)
└── 04_Orchestration/
    └── Workflow_Definition.json            # Databricks Workflow DAG