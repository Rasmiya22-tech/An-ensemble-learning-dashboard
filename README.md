# Ensemble Learning Dashboard : Malaria Case Trend Prediction in Sri Lanka

Forecasting next month's malaria cases per district with **Bagging, Boosting, Stacking** and **hybrid** models (RF + ANN, SARIMA + Random Forest), plus an interactive Streamlit dashboard.
 

## Files

| File | Purpose |
|---|---|
| `malaria_ensemble_learning.ipynb` |  already executed: data checks, EDA, models, comparison, interpretation |
| `srilanka_malaria_monthly_2000_2024.csv` | District-month dataset: 25 districts × 300 months (Jan 2000 to Dec 2024), 7,500 rows |
| `malaria_models.py` | Feature engineering, all models and evaluation. Run it directly for a printed comparison |
| `app.py` | Streamlit dashboard (story, model comparison, district forecasts) |
| `results.csv`, `feature_importances.csv` | Saved outputs from the default run |
| `model_comparison.png`, `national_forecast.png` | Key charts |
| `requirements.txt` | Python dependencies |

## Setup and run

```bash
pip install -r requirements.txt
python malaria_models.py          
streamlit run app.py              
jupyter notebook malaria_ensemble_learning.ipynb
```



## Data

Columns: year, month, district, province, population, climate zone, rainfall, temperature, humidity, blood slides examined, indigenous and imported cases, *P. falciparum* and *P. vivax* cases, deaths, and API per 1,000.

**Important data note.** The dataset follows Sri Lanka's real elimination trajectory: a steep fall from 2000, a rebound in 2008–2010 and zero indigenous cases from 2013. **But it does not match official Anti-Malaria Campaign (AMC) figures.** For example, it has 254,277 cases in 2000 vs 210,039 officially, and 59,010 in 2002 vs 41,411. District populations are round numbers growing at a fixed rate. It is therefore treated as a **reconstructed dataset**, and the notebook includes a comparison with the official annual series (source: AMC table in Wickremasinghe, *The elimination of malaria*, University of Kelaniya / IAEA). The method transfers directly to real AMC district data.

## Method

- **Unit:** district × month, endemic era 2000–2012. After 2012 there are no indigenous cases to forecast.
- **Target:** next month's indigenous cases as `log1p`. Models learn the **month-on-month change** in log cases, which is added back to the current level. Tree models can't predict values outside their training range, and 2000–2007 contained only decline, while 2008 rebounded.
- **Features:** lagged log cases (1, 2, 3, 12), rolling means, 3-month trend, rainfall and its lags / 3-month total, temperature, humidity, blood slides examined, slide positivity, Pf share, population, month (sin/cos) and climate zone.
- **Split:** time-ordered. Train on target months up to Dec 2007, test on Jan 2008 – Dec 2012 (1,500 district-months). Robustness split: train to Dec 2008.
- **Back-transform:** Duan smearing correction (from out-of-fold training residuals), then district forecasts are summed to national totals, so all models can be compared with SARIMA on one scale.

| Technique | Models |
|---|---|
| Baselines | Naive (next = this month), seasonal naive |
| Bagging | Random Forest, BaggingRegressor (decision trees) |
| Boosting | XGBoost, LightGBM, CatBoost, sklearn GradientBoosting |
| Stacking | RF + XGBoost + LightGBM, then a Ridge meta-model (5-fold OOF) |
| Hybrid | RF + ANN (MLP 64-32, averaged); SARIMA(1,1,1)(1,0,1,12) + RF on residuals |

## Results (default split, test 2008–2012)

| Model | Technique | National RMSE (cases/month) | District RMSE (log) | District R² (log) |
|---|---|---|---|---|
| SARIMA | Time series | **14.9** | – | – |
| Naive | Baseline | 15.7 | 0.608 | 0.256 |
| SARIMA + RF | Hybrid | 16.7 | – | – |
| Random Forest | Bagging | 17.8 | **0.512** | **0.471** |
| Bagging (trees) | Bagging | 18.0 | 0.517 | 0.462 |
| RF + ANN | Hybrid | 19.5 | 0.521 | 0.452 |
| Stacking | Stacking | 19.9 | 0.524 | 0.446 |
| CatBoost | Boosting | 20.2 | 0.530 | 0.434 |
| LightGBM | Boosting | 20.5 | 0.532 | 0.430 |
| Gradient Boosting | Boosting | 20.6 | 0.541 | 0.410 |
| XGBoost | Boosting | 21.1 | 0.538 | 0.416 |
| Seasonal naive | Baseline | 33.8 | 0.767 | −0.186 |

With the robustness split (train to Dec 2008), all the ensembles also beat naive nationally, SARIMA + RF narrowly beats SARIMA, and stacking narrowly beats RF at district level.

## Interpretation

- **District level:** every ensemble beats the baselines (R² about 0.41–0.47 vs 0.26 for naive). They pool 25 districts and use lagged rainfall, season and surveillance signals.
- **Bagging did best on this noisy count data.** Averaging deep trees reduces variance. Boosting reduces bias by chasing residuals, which on a noisy, regime-shifting series partly means fitting noise.
- **Stacking** didn't beat its best base model with the default split. RF, XGBoost and LightGBM make similar errors, and stacking needs diverse base models.
- **Top features** are recent case history (current level, 3- and 12-month means). Transmission is self-reinforcing. Rainfall lags and season add smaller amounts.
- **National level:** totals are smooth and autocorrelated, so **SARIMA** wins. The SARIMA + RF hybrid helps only when there is enough training data for residual patterns (see the robustness split).
- The order of the top models changes with the split. The robust conclusions are that ensembles beat the baselines at district level and that SARIMA is strong for national totals.

## Limitations

- reconstructed data.
- Single time-ordered split plus one robustness split, with no walk-forward re-training.
- Near-default hyperparameters (adjustable in the dashboard).
- Very small case counts at the end of the test period inflate percentage errors.

## Possible extensions

- Re-run on real AMC district data (Anti-Malaria Campaign annual reports)
- Walk-forward validation with `TimeSeriesSplit`
- A post-elimination **resurgence risk score** combining climate suitability with imported-case exposure
- Poisson / Tweedie objectives in LightGBM and XGBoost for count data
