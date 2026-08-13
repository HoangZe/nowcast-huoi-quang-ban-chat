# Analysis of LSTM Inflow Nowcasting Results — Bản Chát Reservoir

**Dataset:** `ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv`
**Model:** LSTM multi-horizon inflow nowcasting (t+1 to t+6 hours)
**Date of analysis:** 2026-08-13

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total records | 92,472 |
| Time range | 2016-01-01 07:00 → 2026-07-20 06:00 |
| Time span | ~3,853 days (~10.6 years) |
| Temporal resolution | 1 hour (uniform, no gaps) |
| Reservoir | Bản Chát (lake_info_id = 45) |
| Missing values | 0 (inflow & precipitation fully populated) |

The dataset is a **single-reservoir, hourly time series** spanning roughly 10.6 years. All 16 columns are present with no missing values. The two primary physical features are **inflow (m³/s)** and **precipitation (mm)**; the remaining columns (e.g., `precip_sum_3h/6h/12h/24h`, `precip_rolling_mean_6h`, `day_of_year_sin/cos`) are engineered derivatives of these two.

---

## 2. Inflow (inflow_m3s) Analysis

### 2.1 Descriptive Statistics

| Statistic | Value (m³/s) |
|---|---|
| Count | 92,472 |
| Mean | 119.13 |
| Median (P50) | 43.80 |
| Std | 197.51 |
| Min | 2.00 |
| Max | 4,995.40 |
| Range | 4,993.40 |
| P01 | 7.00 |
| P05 | 10.55 |
| P10 | 11.50 |
| P25 (Q1) | 19.00 |
| P75 (Q3) | 151.70 |
| P90 | 303.90 |
| P95 | 434.00 |
| P99 | 887.43 |
| IQR | 132.70 |

### 2.2 Distribution Shape

| Metric | Value |
|---|---|
| Skewness | 6.36 (strongly right-skewed) |
| Kurtosis | 86.32 (extremely heavy-tailed) |
| Mean / Median ratio | 2.72 |
| Coefficient of variation (CV) | 1.66 |
| Outliers (beyond 1.5×IQR fences) | 6,877 (7.44%) |

**Key observations:**

- The inflow distribution is **extremely right-skewed** (skewness = 6.36) with **heavy tails** (kurtosis = 86.32). The mean (119.13) is nearly **2.7× the median** (43.80), confirming a strongly asymmetric distribution dominated by a long upper tail of flood events.
- The **coefficient of variation is 1.66**, meaning the standard deviation is 66% larger than the mean — extremely high variability.
- **7.44% of records are statistical outliers** (above the 1.5×IQR upper fence of 350.75 m³/s). These are the flood/storm events that dominate the tail.
- **No zero or near-zero values** — the minimum inflow is 2.0 m³/s, and only 0.30% of values fall below 5 m³/s. This is a perennial river with a persistent baseflow.
- The **max inflow (4,995.4 m³/s) is ~42× the median** and ~114× the P25 value, illustrating the extreme dynamic range the model must handle.

### 2.3 Percentile Interpretation

The distribution is heavily concentrated at low values:

- **50% of all hours** have inflow ≤ 43.8 m³/s
- **75% of all hours** have inflow ≤ 151.7 m³/s
- **90% of all hours** have inflow ≤ 303.9 m³/s
- Only **1% of hours** exceed 887.4 m³/s

This means the model operates mostly in a low-flow regime, but must occasionally predict extreme flood peaks that are 10–100× the typical value. This asymmetry is critical for interpreting the error metrics.

---

## 3. Precipitation (precipitation_mm) Analysis

### 3.1 Descriptive Statistics

| Statistic | Value (mm) |
|---|---|
| Count | 92,472 |
| Mean | 0.235 |
| Median | 0.000 |
| Std | 0.855 |
| Max | 27.146 |
| P50 | 0.000 |
| P75 | 0.069 |
| P90 | 0.540 |
| P95 | 1.319 |
| P99 | 4.168 |
| P99.9 | 10.082 |

### 3.2 Distribution Shape

| Metric | Value |
|---|---|
| Skewness | 7.83 (extremely right-skewed) |
| Kurtosis | 97.80 (extremely heavy-tailed) |
| Dry hours (0 mm) | 67,018 (72.47%) |
| Wet hours (>0 mm) | 25,454 (27.53%) |

### 3.3 Intensity Classification (hourly)

| Intensity | Range (mm) | Count | % of all hours |
|---|---|---|---|
| Dry | 0 | 67,018 | 72.47% |
| Light | 0 – 2.5 | 23,256 | 25.15% |
| Moderate | 2.5 – 7.6 | 1,981 | 2.14% |
| Heavy | 7.6 – 50 | 217 | 0.23% |
| Violent | > 50 | 0 | 0.00% |

**Key observations:**

- Precipitation is **extremely sparse**: **72.47% of all hours are completely dry** (0 mm). The median is 0.
- The distribution is **even more skewed than inflow** (skewness = 7.83, kurtosis = 97.80).
- **97.6% of all hours** have precipitation ≤ 2.5 mm (dry or light). Only **0.23% of hours** are heavy (>7.6 mm), and there are **no violent (>50 mm) hourly events** in the record.
- The **maximum hourly precipitation is 27.15 mm** — a heavy but not extreme event.
- The top-5 precipitation events range from 18.4 to 27.1 mm, all occurring in the wet season (June–August).

---

## 4. Seasonal / Temporal Analysis

### 4.1 Monthly Inflow Statistics

| Month | Mean (m³/s) | Median (m³/s) | Std (m³/s) | Max (m³/s) |
|---|---|---|---|---|
| Jan | 32.02 | 19.70 | 42.17 | 634.0 |
| Feb | 23.97 | 19.00 | 26.35 | 858.0 |
| Mar | 22.82 | 16.00 | 26.00 | 492.3 |
| Apr | 35.35 | 19.90 | 53.73 | 916.7 |
| **May** | **87.15** | 40.20 | 123.52 | 2,437.0 |
| **Jun** | **284.55** | 219.80 | 306.87 | 4,995.4 |
| **Jul** | **314.35** | 226.20 | 324.59 | 4,971.9 |
| **Aug** | **298.88** | 237.78 | 234.60 | 2,404.4 |
| **Sep** | **180.52** | 128.00 | 168.66 | 1,970.0 |
| **Oct** | **85.48** | 61.00 | 82.36 | 1,362.7 |
| Nov | 44.28 | 33.00 | 49.06 | 642.9 |
| Dec | 24.04 | 19.70 | 20.70 | 370.6 |

### 4.2 Monthly Precipitation Statistics

| Month | Mean (mm) | Sum (mm) | Max (mm) |
|---|---|---|---|
| Jan | 0.070 | 571.7 | 12.9 |
| Feb | 0.062 | 462.1 | 13.6 |
| Mar | 0.097 | 796.7 | 14.7 |
| Apr | 0.187 | 1,482.8 | 13.7 |
| May | 0.303 | 2,477.5 | 13.2 |
| **Jun** | **0.489** | **3,871.1** | 14.9 |
| **Jul** | **0.505** | **3,993.2** | 25.8 |
| **Aug** | **0.493** | **3,667.2** | 27.1 |
| Sep | 0.332 | 2,389.0 | 18.4 |
| Oct | 0.159 | 1,179.7 | 10.6 |
| Nov | 0.081 | 585.2 | 7.5 |
| Dec | 0.033 | 248.1 | 10.6 |

### 4.3 Wet vs Dry Season (Vietnam monsoon)

| Season | Inflow Mean (m³/s) | Inflow Median (m³/s) | Inflow Std (m³/s) | Precip Mean (mm) | Precip Sum (mm) |
|---|---|---|---|---|---|
| **Wet (May–Oct)** | 208.53 | 145.00 | 246.56 | 0.381 | 17,577.6 |
| **Dry (Nov–Apr)** | 30.29 | 19.70 | 39.21 | 0.089 | 4,146.6 |

**Key observations:**

- There is a **very strong seasonal signal**. Wet-season inflow is **~6.9× higher** than dry-season inflow on average (208.5 vs 30.3 m³/s).
- The **peak inflow months are June–August** (mean 284–314 m³/s), coinciding with peak precipitation (0.49–0.51 mm/h mean).
- **July has the highest mean inflow (314.4 m³/s)** and the highest precipitation total (3,993 mm).
- The **dry season (Nov–Apr)** is remarkably stable: mean inflow stays between 22.8 and 35.4 m³/s, with low variability (std 20–54).
- The **wet season accounts for ~81% of total precipitation** (17,578 of 21,724 mm) and the vast majority of flood peaks.

---

## 5. Inflow–Precipitation Relationship

### 5.1 Correlations

| Measure | Value |
|---|---|
| Pearson correlation (contemporaneous) | 0.292 |
| Spearman correlation (contemporaneous) | 0.425 |

### 5.2 Lagged Correlation (precipitation at t−k vs inflow at t)

| Lag (hours) | Correlation |
|---|---|
| 0 | 0.292 |
| 1 | 0.322 |
| 2 | 0.350 |
| 3 | 0.374 |
| **6** | **0.404 (peak)** |
| 12 | 0.353 |
| 24 | 0.283 |
| 48 | 0.220 |
| 72 | 0.173 |

**Key observations:**

- The **contemporaneous correlation is weak-to-moderate** (Pearson 0.29, Spearman 0.43). The higher Spearman value indicates a **monotonic but non-linear** relationship.
- The **peak lagged correlation occurs at 6 hours** (0.404), meaning precipitation leads inflow by ~6 hours — the catchment response time. This is physically sensible for a mountainous reservoir catchment.
- The correlation **decays slowly** beyond 6h (0.35 at 12h, 0.28 at 24h, 0.17 at 72h), indicating that rainfall has a **multi-day influence** on inflow, consistent with soil saturation and groundwater contributions.
- The relatively modest correlation (max 0.40) suggests that **precipitation alone explains only ~16% of inflow variance** — other factors (antecedent soil moisture, snowmelt, reservoir operations) play a significant role.

---

## 6. Inflow Autocorrelation

| Lag (hours) | Autocorrelation |
|---|---|
| 1 | 0.971 |
| 2 | 0.948 |
| 3 | 0.926 |
| 6 | 0.870 |
| 12 | 0.784 |
| 24 | 0.702 |
| 48 | 0.581 |
| 72 | 0.505 |
| 168 (1 week) | 0.410 |

**Key observations:**

- Inflow is **extremely persistent**: lag-1 autocorrelation is 0.971, meaning today's inflow is an excellent predictor of tomorrow's.
- The autocorrelation **decays slowly** — even at 72 hours it remains 0.505, and at 1 week it is still 0.410.
- This high persistence explains why the model performs best at short horizons (t+1) and degrades as the horizon extends: the model can rely heavily on the persistence of inflow itself, but this advantage fades with time.

---

## 7. Test Results Analysis — How Many % Off From the True Value?

### 7.1 Test Metrics (as provided)

| Horizon | MAE (m³/s) | RMSE (m³/s) | sMAPE (%) | MASE | NSE |
|---|---|---|---|---|---|
| **Overall** | 27.50 | 70.04 | **27.90** | 0.522 | 0.867 |
| t+1 | 18.00 | 46.59 | **19.14** | 0.341 | 0.941 |
| t+2 | 23.84 | 56.89 | **25.11** | 0.452 | 0.913 |
| t+3 | 27.64 | 66.91 | **28.48** | 0.524 | 0.879 |
| t+4 | 29.95 | 74.35 | **29.80** | 0.568 | 0.851 |
| t+5 | 31.91 | 80.87 | **31.40** | 0.605 | 0.823 |
| t+6 | 33.67 | 86.49 | **33.49** | 0.639 | 0.798 |

### 7.2 Percentage Off From True Value

The **sMAPE** is the direct measure of percentage deviation from the true value. It is a symmetric percentage error: a sMAPE of X% means the average symmetric absolute error is X% of the average of |true| and |predicted|.

| Horizon | sMAPE (%) | MAE / Mean Inflow (%) | RMSE / Mean Inflow (%) | MAE / Std (%) | RMSE / Std (%) |
|---|---|---|---|---|---|
| **Overall** | **27.90** | 23.08 | 58.79 | 13.92 | 35.46 |
| t+1 | **19.14** | 15.11 | 39.11 | 9.11 | 23.59 |
| t+2 | **25.11** | 20.01 | 47.75 | 12.07 | 28.80 |
| t+3 | **28.48** | 23.20 | 56.17 | 13.99 | 33.88 |
| t+4 | **29.80** | 25.14 | 62.41 | 15.16 | 37.64 |
| t+5 | **31.40** | 26.79 | 67.89 | 16.16 | 40.95 |
| t+6 | **33.49** | 28.26 | 72.61 | 17.05 | 43.79 |

*Reference values: mean inflow = 119.13 m³/s, median = 43.80 m³/s, std = 197.51 m³/s.*

### 7.3 Direct Answer: % Off From True Value

**The predictions are, on average, off from the true value by:**

| Horizon | % Off (sMAPE) |
|---|---|
| **Overall** | **~27.9%** |
| t+1 | **~19.1%** |
| t+2 | **~25.1%** |
| t+3 | **~28.5%** |
| t+4 | **~29.8%** |
| t+5 | **~31.4%** |
| t+6 | **~33.5%** |

In absolute terms, the average error (MAE) ranges from **18.0 m³/s at t+1** to **33.7 m³/s at t+6**, which corresponds to **15.1% to 28.3% of the mean inflow** (119.13 m³/s).

### 7.4 Interpretation in Context of the Dataset Distribution

The percentage error must be interpreted carefully given the **extreme right-skew** of the inflow distribution:

1. **sMAPE is the most honest % error measure.** Because it normalizes by the average of |true| and |pred|, it is robust to the skew. The overall sMAPE of **27.9%** means the model is, on average, ~28% off from the true value.

2. **The model is excellent at short horizons.** At t+1, the sMAPE is **19.1%** and the NSE is **0.941** — the model captures 94.1% of the variance. This is expected given the very high lag-1 autocorrelation (0.971) of inflow.

3. **Performance degrades monotonically with horizon.** From t+1 to t+6:
   - sMAPE rises from 19.1% → 33.5% (+14.4 percentage points)
   - MAE rises from 18.0 → 33.7 m³/s (+87%)
   - RMSE rises from 46.6 → 86.5 m³/s (+86%)
   - NSE falls from 0.941 → 0.798
   - MASE rises from 0.341 → 0.639

4. **The model is always better than the naive baseline.** MASE < 1.0 at all horizons (0.34–0.64) means the model outperforms a naive persistence forecast by 36–66%. This is a strong result.

5. **The RMSE is much larger than the MAE** (ratio ~2.5×), indicating the presence of **large errors on extreme flood events**. Because the inflow distribution is heavy-tailed (max = 4,995 m³/s vs median = 43.8 m³/s), the model's largest absolute errors occur on rare flood peaks, inflating RMSE disproportionately.

6. **The % error is scale-dependent.** A 28% error on a typical low-flow value (e.g., 44 m³/s) is ~12 m³/s, but a 28% error on a flood peak (e.g., 1,000 m³/s) is ~280 m³/s. The sMAPE averages these across the distribution, so the overall 27.9% reflects a mix of small absolute errors on low flows and larger absolute errors on high flows.

### 7.5 Why the % Error Grows with Horizon

The degradation from t+1 to t+6 is explained by the dataset's temporal structure:

- **Inflow autocorrelation decays** from 0.971 (lag 1) to 0.870 (lag 6). The model's ability to "persist" the current state weakens as the horizon extends.
- **Precipitation's predictive power peaks at a 6-hour lead** (correlation 0.404). Beyond t+6, the model must rely on increasingly stale precipitation information.
- **The seasonal cycle** (wet vs dry) means the model must correctly transition between regimes, which is harder at longer horizons.

### 7.6 Overall Assessment

| Metric | Value | Assessment |
|---|---|---|
| Overall sMAPE | 27.9% | Good for a highly skewed hydrological series |
| t+1 sMAPE | 19.1% | Excellent |
| t+6 sMAPE | 33.5% | Acceptable, but degrades |
| Overall NSE | 0.867 | Very good (captures 86.7% of variance) |
| Overall MASE | 0.522 | Beats naive baseline by ~48% |
| t+1 NSE | 0.941 | Excellent |
| t+6 NSE | 0.798 | Good |

**Bottom line:** The model is **~19% off** from the true value at 1-hour lead, **~28% off** on average across all horizons, and **~33% off** at the 6-hour lead. Given the extreme skew of the inflow distribution (CV = 1.66, max/median = 114×), these are strong results. The model captures 86.7% of the variance overall and beats a naive persistence baseline at every horizon.

---

## 8. Summary of Key Findings

1. **Dataset:** 92,472 hourly records over 10.6 years (2016–2026) for the Bản Chát reservoir, with no missing values.

2. **Inflow is extremely right-skewed** (skew = 6.36, kurtosis = 86.32): mean 119.1 m³/s, median 43.8 m³/s, max 4,995.4 m³/s. 7.44% of values are statistical outliers (flood events).

3. **Precipitation is sparse and skewed** (skew = 7.83): 72.5% of hours are dry, 97.6% are ≤2.5 mm, max 27.1 mm.

4. **Strong seasonality:** Wet season (May–Oct) inflow is ~6.9× the dry season (208.5 vs 30.3 m³/s). Peak months are June–August.

5. **Precipitation leads inflow by ~6 hours** (peak lagged correlation 0.404), with a slow multi-day decay.

6. **Inflow is highly persistent** (lag-1 autocorrelation 0.971), which the model exploits at short horizons.

