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
