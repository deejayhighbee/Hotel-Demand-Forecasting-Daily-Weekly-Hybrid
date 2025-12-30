This notebook delivers a production-ready demand forecasting pipeline for the Kaggle **Hotel Booking Demand** dataset (`/kaggle/input/hotel-booking-demand/hotel_bookings.csv`). The objective is to generate **reliable daily and weekly forecasts** across multiple operational and commercial demand signals while keeping the workflow robust to data issues, minimizing noise, and producing exportable artifacts that can be reused in downstream analytics or deployment contexts.

### Forecasting objectives

Hotels face strong seasonality (weekday/weekend, annual cycles), short-term volatility, and structural differences across segments. This notebook is designed to support practical planning use cases by producing forecasts at two levels:

- **Daily forecasts**: short-horizon operational planning (staffing, inventory, rate adjustments, workload planning).
- **Weekly forecasts**: medium-horizon planning (capacity planning, budgeting, procurement, strategic operations).

Each hotel type is modeled as a separate series (e.g., **City Hotel** and **Resort Hotel**) to avoid mixing series with different seasonal and volatility behaviors.

### Targets forecasted

From booking-level records, the notebook constructs forecastable time series at daily and weekly frequency, such as:

- **bookings**: total arrivals per day/week.
- **canceled_bookings**: cancellations per day/week.
- **net_bookings**: bookings adjusted for cancellations (as defined in the notebook).
- **guests_total / adults_total**: aggregated guest volume measures.
- **room_nights**: room-night demand implied by stays.
- **revenue**: a revenue proxy constructed from aggregated ADR and volume (as defined in the notebook).
- **adr_mean**: average daily rate signal.
- **cancel_rate**: cancellation ratio.

All targets are represented in a consistent **panel time series** format:

- `unique_id`: series identifier (hotel type)
- `ds`: timestamp
- `y`: target value

### Data preparation and noise reduction principles

Because the dataset is transactional and must be aggregated into time series, the notebook prioritizes data steps that preserve the true demand signal:

1. **Robust date construction**  
   A valid `arrival_date` is created from year/month/day fields using a safe month mapping. This becomes the canonical time index used for aggregation.

2. **Deterministic missing value handling**  
   - `children`: imputed with 0, clipped at 0, and converted to integer (count variable).
   - `country`: imputed with `"UNK"`.
   - `agent` / `company`: imputed with `-1` to preserve categorical identity safely.

3. **Duplicate handling aligned with demand aggregation**  
   Duplicate-looking rows are not automatically removed. Because the dataset represents booking transactions, repeated identical rows can be legitimate and contribute to true demand. The workflow preserves them unless a targeted investigation confirms they are erroneous duplicates.

4. **Stable aggregation rules**  
   Daily and weekly series are computed consistently per hotel type. Weekly series are anchored to a consistent definition (e.g., **W-MON**) so that each week is comparable for modeling and evaluation.

### Modeling approach: hybrid forecasting

To maximize accuracy and stability, the notebook uses a hybrid approach combining statistical time series models and machine learning:

#### Statistical models (StatsForecast)

- **ETS / AutoETS (or ETSModel depending on library version)**: trend + seasonality baselines.
- **AutoARIMA**: autocorrelation and seasonal structure.
- **Seasonal Naive** (when included): strong benchmark, frequently competitive for stable seasonal demand.

#### Machine learning model (MLForecast + LightGBM)

- **LightGBM regressor** trained on:
  - lag features (e.g., 1, 7, 14, 28, etc.)
  - calendar features (day-of-week, week-of-year/week index, month, year depending on frequency)
- Designed to capture nonlinearities and interactions that classical models may not represent explicitly.

#### Variance stabilization with log1p (ML side)

For spiky and heteroscedastic targets (counts and revenue-like series), the ML model can be trained on a stabilized scale:

- `y_ml = log1p(clip(y, 0))`
- predictions are back-transformed using `expm1` and clipped at 0

This reduces sensitivity to outliers and typically improves generalization while maintaining non-negativity constraints.

#### Hybrid blending

A **Hybrid** prediction is computed as a weighted blend:

- `Hybrid = alpha * base_stat + (1 - alpha) * LGBM`

Where:
- `alpha` is chosen via backtesting.
- `base_stat` is the selected statistical baseline (ETS or AutoARIMA), depending on what best supports each target/series.

### Validation and model selection (rolling backtests)

Forecast performance is assessed using **rolling-origin cross-validation** (walk-forward validation), which simulates real forecasting:

- Each window trains on historical data and predicts a fixed horizon.
- Multiple windows are evaluated to reduce dependence on a single cutoff.
- Best model selection is performed per **frequency × target × hotel series** using robust error metrics.

Evaluation focuses on accuracy and operational relevance:

- **MAE / RMSE**: magnitude of errors.
- **MAPE / sMAPE**: relative accuracy (with safe handling).
- **WAPE**: weighted error metric commonly preferred for demand.
- **Bias / Bias%**: systematic over/under forecasting.
- **Over/under forecasting rates**: directional diagnostics.
- **Actual vs forecast totals**: volume-level realism.

### Deliverables produced

At the end of the workflow, the notebook generates production-ready artifacts:

1. **Backtest outputs**
   - historical predictions for each model and “best model” selection
   - backtest CSVs for daily and weekly

2. **Future forecasts (2 years)**
   - all-model forecasts saved as CSV
   - best-model forecasts saved as CSV

3. **Visual artifacts**
   For each target and series:
   - actual + forecast plots (daily and weekly)
   - last-window backtest overlays
   - error distribution histograms

4. **Forecast statistics report**
   A detailed CSV summarizing performance and diagnostics suitable for reporting, monitoring, and stakeholder communication.

### Forecast Performance Narrative (Backtest)

**Overall (D)**
- Weighted WAPE: **28.68%**
- Weighted Bias%: **-7.32%**
- Weighted sMAPE: **32.43%**

**Overall (W-MON)**
- Weighted WAPE: **22.05%**
- Weighted Bias%: **-15.18%**
- Weighted sMAPE: **24.07%**

**Strongest performance (D)**
- adr_mean | City Hotel: WAPE **8.39%**, Bias% **-3.47%**, MAE **9.42**
- adr_mean | Resort Hotel: WAPE **13.47%**, Bias% **-6.01%**, MAE **12.98**
- net_bookings | City Hotel: WAPE **25.82%**, Bias% **3.14%**, MAE **16.13**
- room_nights | City Hotel: WAPE **26.55%**, Bias% **-4.71%**, MAE **89.88**
- cancel_rate | City Hotel: WAPE **26.86%**, Bias% **0.32%**, MAE **0.11**

**Most challenging segments (D)**
- canceled_bookings | City Hotel: WAPE **46.52%**, Bias% **-5.04%**, MAE **21.33**
- canceled_bookings | Resort Hotel: WAPE **44.88%**, Bias% **-17.16%**, MAE **6.88**
- room_nights | Resort Hotel: WAPE **33.71%**, Bias% **-0.39%**, MAE **76.73**
- cancel_rate | Resort Hotel: WAPE **33.22%**, Bias% **-14.72%**, MAE **0.09**
- revenue | Resort Hotel: WAPE **31.35%**, Bias% **-6.94%**, MAE **7689.83**

**Strongest performance (W-MON)**
- adr_mean | City Hotel: WAPE **8.19%**, Bias% **-5.82%**, MAE **9.02**
- net_bookings | City Hotel: WAPE **11.77%**, Bias% **-5.75%**, MAE **54.40**
- bookings | Resort Hotel: WAPE **12.76%**, Bias% **-4.52%**, MAE **47.78**
- cancel_rate | City Hotel: WAPE **13.77%**, Bias% **0.44%**, MAE **0.06**
- net_bookings | Resort Hotel: WAPE **14.11%**, Bias% **-0.47%**, MAE **37.62**

**Most challenging segments (W-MON)**
- canceled_bookings | Resort Hotel: WAPE **28.00%**, Bias% **-14.53%**, MAE **30.21**
- canceled_bookings | City Hotel: WAPE **27.30%**, Bias% **-6.83%**, MAE **89.80**
- revenue | Resort Hotel: WAPE **26.14%**, Bias% **-19.57%**, MAE **45064.37**
- cancel_rate | Resort Hotel: WAPE **22.72%**, Bias% **-9.83%**, MAE **0.06**
- revenue | City Hotel: WAPE **19.69%**, Bias% **-12.69%**, MAE **53135.03**

**Bias and volume behavior**
- Positive Bias% indicates systematic over-forecasting; negative Bias% indicates under-forecasting.
- Volume_Error_pct indicates whether total forecasted volume over the evaluation period is above or below actual volume.

### What the workflow achieves

By design, the pipeline delivers:

- Clean, validated daily and weekly time series built from booking-level records.
- Multi-model forecasting using both:
  - statistical models (ETS/ARIMA/Seasonal Naive), and
  - machine learning (LightGBM via MLForecast).
- Hybrid blending to combine seasonal stability with nonlinear predictive power.
- Rolling backtests that realistically represent how the model will perform in production.
- Two-year forward forecasts exported in structured CSV format.
- Full transparency through saved predictions, saved metrics, and saved diagnostic plots.

### Why the workflow is production-ready

This workflow is structured for operational reuse:

1. **Reproducibility**  
   The entire process—from raw data ingestion to forecasting outputs—is deterministic and produces consistent artifacts (CSVs and plots) that can be versioned.

2. **Frequency-specific forecasting**  
   Daily and weekly forecasts are treated as separate products with consistent time indexing and aggregation. Weekly series alignment (e.g., W-MON) prevents misinterpretation and improves comparability.

3. **Model governance**  
   Best-model selection is performed per target and hotel type using repeatable backtest criteria. This supports segment-level monitoring, targeted improvements, and explainable model decisions.

4. **Diagnostics that support trust**  
   Accuracy is complemented with bias and distribution diagnostics:
   - error histograms show skew and tail behavior,
   - bias measures show systematic under/over forecasting,
   - backtest overlays help validate tracking against real demand patterns.

5. **Operational constraints**  
   Forecasts are constrained where appropriate (e.g., non-negativity), which ensures outputs remain realistic for demand and revenue-like metrics.

### How to use the final outputs

The notebook produces artifacts designed for both analytics and operational planning:

- **Backtest CSVs**: performance reporting and validation, audit-ready evaluation.
- **Future forecast CSVs**: planning dashboards, BI pipelines, automated workflows.
- **Forecast plots**: communication-ready visual summaries for stakeholders.
- **Detailed forecast stats CSV**: monitoring and alerting, KPI-based governance.

### Recommended operational next steps

For ongoing usage beyond Kaggle:

- Retrain periodically (e.g., monthly or quarterly) to incorporate new patterns.
- Track WAPE and Bias% over time to detect drift and degradation.
- Add external drivers (holidays, promotions, events) if available for further uplift.
- Consider reconciliation policies for strict consistency among linked targets (e.g., bookings vs cancellations vs net bookings).

This notebook provides a robust forecasting foundation: **validated data → rolling backtests → best-model selection → future forecasting → exportable artifacts**, ready for integration into decision-making and production analytics environments.
