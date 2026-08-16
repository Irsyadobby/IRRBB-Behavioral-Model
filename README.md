# IRRBB Behavioral Model - Early Redemption & Prepayment Rate Forecasting

## Overview
This project implements a **quantitative behavioral modeling framework** for Interest Rate Risk in the Banking Book (IRRBB), estimating early redemption and 
loan prepayment rate of interest-sensitive products using time-series econometric models with macroeconomic predictors.

Behavioral assumptions (e.g Term Deposits Early Redemption, Loan Prepayments) are one of the most material drivers of IRRBB exposure. Under-estimating these rates
leads to **under-hedging** (residual PV01/EVE exposure), while over-estimating leads to **over-hedging** (excess notional in payer/receiver swaps), both of which distorts
ΔEVE, ΔNII, and hedge accounting effectiveness. This framework aims to produce statistically robust, macro-driven forecast of behavioral rates to support the bank's IRRBB measurement, 
hedging strategy, and ALM / Treasury cash flow optimization process.

## Target Variables (Y)
Five behavioral rate series are modeled independently:

|No|Target Variable|Product Category|
|---|---|---|
| 1 | Term Deposit Early Redemption — IDR | Liabilities (behavioural optionality, deposit side) |
| 2 | Term Deposit Early Redemption — USD | Liabilities (behavioural optionality, deposit side) |
| 3 | Personal Loan (Prepayment Rate) | Assets (financial/statistical prepayment) |
| 4 | Mortgage (Prepayment Rate) | Assets (financial/statistical prepayment) |
| 5 | Joint Finance (Prepayment Rate) | Assets (financial/statistical prepayment) |

Each series is bounded in **(0, 1)** as a rate, which is handled explicitly via a 
**logit transformation** in the modeling pipeline (see Methodology).

## Macroeconomic Variables (X)
Macroeconomic indicators used as exogenous regressors, including but not limited to:
- Benchmark interest rates (BI Rate / Fed Rate)
- Inflation (CPI)
- Exchange rate (USD/IDR)
- Equity/market index
- Commodity prices
- Capital expenditure indicators
- GDP growth

An **anomaly/outlier dummy variable (XAO)** is optionally included to control for 
structural shocks in the observation window (e.g., COVID-19 period), preventing such 
events from biasing the estimated macro-sensitivity coefficients.

## End-to-End Workflow
```
mermaid
---
config:
  theme: default
---
flowchart TD
    A[("Raw Data: Y Behavioral Rate Series, X Macro Indicator Pool")]:::data --> FE

    subgraph FE["Feature Screening"]
        direction TB
        B1[Lag-Correlation Screening]:::screen
        B2[VIF Multicollinearity Check]:::screen
        B3[EMA Smoothing]:::screen
        B4[PCA - optional]:::screen
    end

    FE --> C[Logit / Log Transformation on Target Y]:::transform

    C --> D1
    C --> D2

    subgraph ARIMAX["Primary Model: ARIMAX"]
        direction TB
        D1[Combinatorial Grid Search: X-subset x p,d,q order]:::arimax
        D3[Diagnostic Tests: Ljung-Box / Heteroskedasticity / Jarque-Bera]:::arimax
        D1 --> D3
    end

    subgraph ML["Benchmark Model: HGBR"]
        direction TB
        D2[Time-Series CV Hyperparameter Tuning]:::ml
        D4[Permutation Feature Importance]:::ml
        D2 --> D4
    end

    D3 --> E
    D4 --> E

    E[["Final Behavioral Rate Model, selected by OOS MAPE + Diagnostic Validity"]]:::final
    E --> F[("Behaviouralised Cash Flow, mapped to IRRBB Metrics: Delta EVE / Delta NII / PV01")]:::output

    classDef data fill:#1f6feb,stroke:#0d419d,color:#fff
    classDef screen fill:#8250df,stroke:#5a32a3,color:#fff
    classDef transform fill:#bf8700,stroke:#9a6700,color:#fff
    classDef arimax fill:#1a7f37,stroke:#116329,color:#fff
    classDef ml fill:#cf222e,stroke:#a40e26,color:#fff
    classDef final fill:#0969da,stroke:#0550ae,color:#fff
    classDef output fill:#57606a,stroke:#32383f,color:#fff
```
