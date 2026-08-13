import pandas as pd
import numpy as np
import json

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv('dataset/ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv')

# Parse time
df['time_update'] = pd.to_datetime(df['time_update'])
df = df.sort_values('time_update').reset_index(drop=True)

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"Time range: {df['time_update'].min()}  ->  {df['time_update'].max()}")
print(f"Time span: {(df['time_update'].max() - df['time_update'].min())}")
print(f"Unique lakes: {df['lake_name'].unique()}")
print(f"Lake info IDs: {df['lake_info_id'].unique()}")

# Check for duplicates / gaps
time_diffs = df['time_update'].diff().dropna()
print(f"\nMedian time step: {time_diffs.median()}")
print(f"Unique time steps: {time_diffs.unique()[:10]}")

# ============================================================
# 2. INFLOW ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("INFLOW (inflow_m3s) ANALYSIS")
print("=" * 80)

inflow = df['inflow_m3s'].dropna()
print(f"\nCount (non-null): {inflow.count()} / {len(df)}")
print(f"Missing: {df['inflow_m3s'].isna().sum()}")

print("\n--- Descriptive Statistics ---")
stats = inflow.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(stats.to_string())

print("\n--- Distribution Shape ---")
mean = inflow.mean()
median = inflow.median()
std = inflow.std()
skew = inflow.skew()
kurt = inflow.kurtosis()
print(f"Mean: {mean:.4f}")
print(f"Median: {median:.4f}")
print(f"Std: {std:.4f}")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurt:.4f}")
print(f"Mean/Median ratio: {mean/median:.4f}")
print(f"CV (std/mean): {std/mean:.4f}")

# Percentiles
print("\n--- Key Percentiles ---")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:02d}: {np.percentile(inflow, p):.4f} m3/s")

# IQR
q1, q3 = np.percentile(inflow, 25), np.percentile(inflow, 75)
iqr = q3 - q1
print(f"\nIQR: {iqr:.4f}")
print(f"Lower fence (Q1 - 1.5*IQR): {q1 - 1.5*iqr:.4f}")
print(f"Upper fence (Q3 + 1.5*IQR): {q3 + 1.5*iqr:.4f}")
outlier_mask = (inflow < (q1 - 1.5 * iqr)) | (inflow > (q3 + 1.5 * iqr))
print(f"Outliers (beyond fences): {outlier_mask.sum()} ({outlier_mask.mean()*100:.2f}%)")

# Range
print(f"\nMin: {inflow.min():.4f}")
print(f"Max: {inflow.max():.4f}")
print(f"Range: {inflow.max() - inflow.min():.4f}")

# Zero / near-zero values
print(f"\nZero values: {(inflow == 0).sum()} ({(inflow == 0).mean()*100:.2f}%)")
print(f"Values < 1: {(inflow < 1).sum()} ({(inflow < 1).mean()*100:.2f}%)")
print(f"Values < 5: {(inflow < 5).sum()} ({(inflow < 5).mean()*100:.2f}%)")

# ============================================================
# 3. PRECIPITATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PRECIPITATION (precipitation_mm) ANALYSIS")
print("=" * 80)

precip = df['precipitation_mm'].dropna()
print(f"\nCount (non-null): {precip.count()} / {len(df)}")
print(f"Missing: {df['precipitation_mm'].isna().sum()}")

print("\n--- Descriptive Statistics ---")
print(precip.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).to_string())

print("\n--- Distribution Shape ---")
pmean = precip.mean()
pmedian = precip.median()
pstd = precip.std()
print(f"Mean: {pmean:.4f}")
print(f"Median: {pmedian:.4f}")
print(f"Std: {pstd:.4f}")
print(f"Skewness: {precip.skew():.4f}")
print(f"Kurtosis: {precip.kurtosis():.4f}")

# Wet / dry analysis
dry = (precip == 0).sum()
wet = (precip > 0).sum()
print(f"\nDry hours (0 mm): {dry} ({dry/len(precip)*100:.2f}%)")
print(f"Wet hours (>0 mm): {wet} ({wet/len(precip)*100:.2f}%)")

# Intensity classification
light = ((precip > 0) & (precip <= 2.5)).sum()
moderate = ((precip > 2.5) & (precip <= 7.6)).sum()
heavy = ((precip > 7.6) & (precip <= 50)).sum()
violent = (precip > 50).sum()
print(f"\nIntensity classification (hourly):")
print(f"  Light (0-2.5 mm): {light} ({light/len(precip)*100:.2f}%)")
print(f"  Moderate (2.5-7.6 mm): {moderate} ({moderate/len(precip)*100:.2f}%)")
print(f"  Heavy (7.6-50 mm): {heavy} ({heavy/len(precip)*100:.2f}%)")
print(f"  Violent (>50 mm): {violent} ({violent/len(precip)*100:.2f}%)")

# Max precipitation
print(f"\nMax hourly precip: {precip.max():.4f} mm")
print(f"Top 5 precip events:")
print(precip.nlargest(5).to_string())

# ============================================================
# 4. SEASONAL / TEMPORAL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("TEMPORAL / SEASONAL ANALYSIS")
print("=" * 80)

df['month'] = df['time_update'].dt.month
df['year'] = df['time_update'].dt.year

print("\n--- Monthly Inflow Statistics ---")
monthly_inflow = df.groupby('month')['inflow_m3s'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
print(monthly_inflow.to_string())

print("\n--- Monthly Precipitation Statistics ---")
monthly_precip = df.groupby('month')['precipitation_mm'].agg(['mean', 'sum', 'max', 'count'])
print(monthly_precip.to_string())

# Wet season vs dry season (Vietnam: wet ~ May-Oct, dry ~ Nov-Apr)
wet_months = [5, 6, 7, 8, 9, 10]
dry_months = [11, 12, 1, 2, 3, 4]
wet_mask = df['month'].isin(wet_months)
dry_mask = df['month'].isin(dry_months)

print("\n--- Wet Season (May-Oct) vs Dry Season (Nov-Apr) ---")
print(f"Wet season inflow: mean={df.loc[wet_mask, 'inflow_m3s'].mean():.4f}, median={df.loc[wet_mask, 'inflow_m3s'].median():.4f}, std={df.loc[wet_mask, 'inflow_m3s'].std():.4f}")
print(f"Dry season inflow: mean={df.loc[dry_mask, 'inflow_m3s'].mean():.4f}, median={df.loc[dry_mask, 'inflow_m3s'].median():.4f}, std={df.loc[dry_mask, 'inflow_m3s'].std():.4f}")
print(f"Wet season precip: mean={df.loc[wet_mask, 'precipitation_mm'].mean():.4f}, sum={df.loc[wet_mask, 'precipitation_mm'].sum():.4f}")
print(f"Dry season precip: mean={df.loc[dry_mask, 'precipitation_mm'].mean():.4f}, sum={df.loc[dry_mask, 'precipitation_mm'].sum():.4f}")

# ============================================================
# 5. INFLOW-PRECIPITATION RELATIONSHIP
# ============================================================
print("\n" + "=" * 80)
print("INFLOW-PRECIPITATION RELATIONSHIP")
print("=" * 80)

# Correlation
corr_pearson = df['inflow_m3s'].corr(df['precipitation_mm'])
corr_spearman = df['inflow_m3s'].corr(df['precipitation_mm'], method='spearman')
print(f"Pearson correlation (inflow, precip): {corr_pearson:.4f}")
print(f"Spearman correlation (inflow, precip): {corr_spearman:.4f}")

# Lagged correlations (precip leads inflow)
print("\n--- Lagged correlation (precip at t-k vs inflow at t) ---")
for lag in [0, 1, 2, 3, 6, 12, 24, 48, 72]:
    corr = df['precipitation_mm'].shift(lag).corr(df['inflow_m3s'])
    print(f"  Lag {lag:3d}h: {corr:.4f}")

# ============================================================
# 6. AUTOCORRELATION OF INFLOW
# ============================================================
print("\n" + "=" * 80)
print("INFLOW AUTOCORRELATION")
print("=" * 80)
for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
    ac = inflow.autocorr(lag=lag)
    print(f"  Lag {lag:3d}h: {ac:.4f}")

# ============================================================
# 7. TEST METRICS ANALYSIS - % OFF FROM TRUE VALUE
# ============================================================
print("\n" + "=" * 80)
print("TEST METRICS ANALYSIS - % OFF FROM TRUE VALUE")
print("=" * 80)

# Test metrics provided
metrics = {
    'overall': {'MAE': 27.4996, 'RMSE': 70.0393, 'sMAPE': 27.9042, 'MASE': 0.5216, 'NSE': 0.8674},
    't+1': {'MAE': 17.9992, 'RMSE': 46.5883, 'sMAPE': 19.1438, 'MASE': 0.3414, 'NSE': 0.9413},
    't+2': {'MAE': 23.8429, 'RMSE': 56.8860, 'sMAPE': 25.1102, 'MASE': 0.4522, 'NSE': 0.9125},
    't+3': {'MAE': 27.6354, 'RMSE': 66.9130, 'sMAPE': 28.4784, 'MASE': 0.5242, 'NSE': 0.8790},
    't+4': {'MAE': 29.9451, 'RMSE': 74.3468, 'sMAPE': 29.7985, 'MASE': 0.5680, 'NSE': 0.8506},
    't+5': {'MAE': 31.9084, 'RMSE': 80.8725, 'sMAPE': 31.4002, 'MASE': 0.6052, 'NSE': 0.8233},
    't+6': {'MAE': 33.6666, 'RMSE': 86.4947, 'sMAPE': 33.4942, 'MASE': 0.6385, 'NSE': 0.7979},
}

# Compute % off from true value using sMAPE
# sMAPE = (100/n) * sum(2*|y_true - y_pred| / (|y_true| + |y_pred|))
# The sMAPE value IS the average % error (symmetric). 
# But we can also express MAE as % of mean inflow, and RMSE as % of mean inflow.

inflow_mean = inflow.mean()
inflow_median = inflow.median()
inflow_std = inflow.std()

print(f"\nDataset inflow reference values:")
print(f"  Mean inflow: {inflow_mean:.4f} m3/s")
print(f"  Median inflow: {inflow_median:.4f} m3/s")
print(f"  Std inflow: {inflow_std:.4f} m3/s")

print("\n--- % Off From True Value (using sMAPE as the direct % error) ---")
print(f"{'Horizon':<10} {'sMAPE (%)':<12} {'MAE/Mean (%)':<14} {'RMSE/Mean (%)':<14} {'MAE/Std (%)':<12} {'RMSE/Std (%)':<12}")
print("-" * 80)
for horizon, m in metrics.items():
    mae_pct_mean = m['MAE'] / inflow_mean * 100
    rmse_pct_mean = m['RMSE'] / inflow_mean * 100
    mae_pct_std = m['MAE'] / inflow_std * 100
    rmse_pct_std = m['RMSE'] / inflow_std * 100
    print(f"{horizon:<10} {m['sMAPE']:<12.4f} {mae_pct_mean:<14.4f} {rmse_pct_mean:<14.4f} {mae_pct_std:<12.4f} {rmse_pct_std:<12.4f}")

# Interpretation of sMAPE
print("\n--- Interpretation of sMAPE ---")
print("sMAPE is a symmetric percentage error. A sMAPE of X% means the average")
print("symmetric absolute error is X% of the average of |true| and |pred|.")
print("")
print("For the overall model: sMAPE = 27.90%")
print("  - This means on average, predictions deviate ~27.9% from the true value")
print("    (symmetric measure).")
print("")
print("For t+1: sMAPE = 19.14%")
print("  - 1-hour-ahead predictions are off by ~19.1% on average.")
print("")
print("For t+6: sMAPE = 33.49%")
print("  - 6-hour-ahead predictions are off by ~33.5% on average.")

# MAE as % of mean
print("\n--- MAE as % of Mean Inflow ---")
print("MAE represents the average absolute error in m3/s.")
print(f"Overall MAE = {metrics['overall']['MAE']:.4f} m3/s = {metrics['overall']['MAE']/inflow_mean*100:.2f}% of mean inflow")
print(f"t+1 MAE = {metrics['t+1']['MAE']:.4f} m3/s = {metrics['t+1']['MAE']/inflow_mean*100:.2f}% of mean inflow")
print(f"t+6 MAE = {metrics['t+6']['MAE']:.4f} m3/s = {metrics['t+6']['MAE']/inflow_mean*100:.2f}% of mean inflow")

# RMSE as % of mean
print("\n--- RMSE as % of Mean Inflow ---")
print(f"Overall RMSE = {metrics['overall']['RMSE']:.4f} m3/s = {metrics['overall']['RMSE']/inflow_mean*100:.2f}% of mean inflow")
print(f"t+1 RMSE = {metrics['t+1']['RMSE']:.4f} m3/s = {metrics['t+1']['RMSE']/inflow_mean*100:.2f}% of mean inflow")
print(f"t+6 RMSE = {metrics['t+6']['RMSE']:.4f} m3/s = {metrics['t+6']['RMSE']/inflow_mean*100:.2f}% of mean inflow")

# ============================================================
# 8. PEAK FLOOD EVENT ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PEAK FLOOD EVENT ANALYSIS")
print("=" * 80)

# Detect local maxima (peaks) in inflow. A peak is a point higher than its
# neighbors within a window. Use scipy if available, else a simple rolling
# comparison. We require a minimum prominence to avoid noise.
try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

inflow_values = df['inflow_m3s'].to_numpy(dtype=float)

if _HAS_SCIPY:
    # Find peaks with a minimum height of 850 (smallest flood category) and
    # a minimum horizontal distance of 24 hours so distinct events are not
    # double-counted on the same rising/falling limb.
    peak_indices, peak_props = find_peaks(
        inflow_values,
        height=850,
        distance=24,
        prominence=50,
    )
else:
    # Fallback: simple local maxima above 850 with a 12-hour separation.
    peak_indices = []
    for i in range(1, len(inflow_values) - 1):
        if inflow_values[i] >= 850 and inflow_values[i] >= inflow_values[i - 1] and inflow_values[i] >= inflow_values[i + 1]:
            if not peak_indices or (i - peak_indices[-1]) >= 12:
                peak_indices.append(i)
    peak_indices = np.asarray(peak_indices, dtype=int)

print(f"\nTotal detected peak events (inflow >= 850 m3/s): {len(peak_indices)}")
print(f"Peak detection method: {'scipy.signal.find_peaks' if _HAS_SCIPY else 'simple local maxima'}")

# Classify peaks into the 3 flood categories
def classify_peak(peak_value: float) -> str:
    if peak_value >= 3000:
        return "major_and_large"
    elif peak_value >= 2100:
        return "big"
    elif peak_value >= 850:
        return "small"
    return "below_threshold"

peak_records = []
for idx in peak_indices:
    peak_value = float(inflow_values[idx])
    peak_records.append({
        'index': int(idx),
        'timestamp': df['time_update'].iloc[idx],
        'peak_value': peak_value,
        'category': classify_peak(peak_value),
    })

peak_df = pd.DataFrame(peak_records)

print("\n--- Peak Event Counts by Category ---")
category_counts = peak_df['category'].value_counts()
for cat in ['small', 'big', 'major_and_large']:
    count = int(category_counts.get(cat, 0))
    print(f"  {cat:<18}: {count} events")

print("\n--- Peak Events Detail ---")
if len(peak_df) > 0:
    for _, row in peak_df.iterrows():
        print(f"  {row['timestamp']}  peak={row['peak_value']:>8.2f} m3/s  [{row['category']}]")

# Determine which split each peak falls into (using the same 70/85 split as training)
train_boundary = int(len(df) * 0.70)
validation_boundary = int(len(df) * 0.85)
print("\n--- Peak Events by Split (70/15/15) ---")
for cat in ['small', 'big', 'major_and_large']:
    cat_df = peak_df[peak_df['category'] == cat]
    if len(cat_df) == 0:
        print(f"  {cat:<18}: no events")
        continue
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    for idx in cat_df['index']:
        if idx < train_boundary:
            split_counts['train'] += 1
        elif idx < validation_boundary:
            split_counts['validation'] += 1
        else:
            split_counts['test'] += 1
    print(f"  {cat:<18}: train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}")

# ============================================================
# 8b. DATA LEAKAGE CHECK - WHICH PEAKS WERE IN THE TRAINING SET?
# ============================================================
print("\n" + "=" * 80)
print("DATA LEAKAGE CHECK - PEAK EVENTS vs TRAIN/VAL/TEST SPLIT")
print("=" * 80)

# Read the checkpoint's actual split boundaries (authoritative source)
import torch
checkpoint_path = 'src/models/lstm_inflow.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    split_boundaries = checkpoint.get('split_boundaries', {})
    print(f"\nCheckpoint split boundaries:")
    print(f"  train_end_row_exclusive: {split_boundaries.get('train_end_row_exclusive')}")
    print(f"  validation_end_row_exclusive: {split_boundaries.get('validation_end_row_exclusive')}")
    print(f"  test_end_row_exclusive: {split_boundaries.get('test_end_row_exclusive')}")
    print(f"  train_end_timestamp: {split_boundaries.get('train_end_timestamp')}")
    print(f"  validation_end_timestamp: {split_boundaries.get('validation_end_timestamp')}")
    ckpt_train_end = int(split_boundaries.get('train_end_row_exclusive', train_boundary))
    ckpt_val_end = int(split_boundaries.get('validation_end_row_exclusive', validation_boundary))
    print(f"\n  => Rows [0, {ckpt_train_end}) = TRAIN")
    print(f"  => Rows [{ckpt_train_end}, {ckpt_val_end}) = VALIDATION")
    print(f"  => Rows [{ckpt_val_end}, {len(df)}) = TEST")
except Exception as e:
    print(f"\nCould not read checkpoint: {e}")
    ckpt_train_end = train_boundary
    ckpt_val_end = validation_boundary

# Classify each peak into train/val/test using the checkpoint boundaries
def split_of(idx: int) -> str:
    if idx < ckpt_train_end:
        return "train"
    elif idx < ckpt_val_end:
        return "validation"
    return "test"

peak_df['split'] = peak_df['index'].apply(split_of)

print("\n--- Peak Events in the TRAINING set (LEAKED into model weights) ---")
train_peaks = peak_df[peak_df['split'] == 'train']
print(f"Total training-set peaks: {len(train_peaks)}")
for cat in ['small', 'big', 'major_and_large']:
    cat_train = train_peaks[train_peaks['category'] == cat]
    if len(cat_train) == 0:
        continue
    print(f"\n  [{cat}] ({len(cat_train)} events):")
    for _, row in cat_train.iterrows():
        print(f"    {row['timestamp']}  peak={row['peak_value']:>8.2f} m3/s")

print("\n--- Peak Events in the HELD-OUT TEST set (leakage-free) ---")
test_peaks = peak_df[peak_df['split'] == 'test']
print(f"Total test-set peaks: {len(test_peaks)}")
for cat in ['small', 'big', 'major_and_large']:
    cat_test = test_peaks[test_peaks['category'] == cat]
    if len(cat_test) == 0:
        print(f"\n  [{cat}]: no test events")
        continue
    print(f"\n  [{cat}] ({len(cat_test)} events):")
    for _, row in cat_test.iterrows():
        print(f"    {row['timestamp']}  peak={row['peak_value']:>8.2f} m3/s")

print("\n--- Peak Events in the VALIDATION set ---")
val_peaks = peak_df[peak_df['split'] == 'validation']
print(f"Total validation-set peaks: {len(val_peaks)}")
for cat in ['small', 'big', 'major_and_large']:
    cat_val = val_peaks[val_peaks['category'] == cat]
    if len(cat_val) == 0:
        continue
    print(f"  [{cat}] ({len(cat_val)} events): {', '.join(str(r['timestamp']) for _, r in cat_val.iterrows())}")

# ============================================================
# 9. SAVE SUMMARY TO JSON FOR MD GENERATION
# ============================================================
summary = {
    'dataset': {
        'rows': len(df),
        'time_range': [str(df['time_update'].min()), str(df['time_update'].max())],
        'time_span_days': (df['time_update'].max() - df['time_update'].min()).days,
    },
    'inflow': {
        'mean': float(inflow_mean),
        'median': float(inflow_median),
        'std': float(inflow_std),
        'min': float(inflow.min()),
        'max': float(inflow.max()),
        'skew': float(skew),
        'kurtosis': float(kurt),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'p01': float(np.percentile(inflow, 1)),
        'p05': float(np.percentile(inflow, 5)),
        'p10': float(np.percentile(inflow, 10)),
        'p25': float(np.percentile(inflow, 25)),
        'p50': float(np.percentile(inflow, 50)),
        'p75': float(np.percentile(inflow, 75)),
        'p90': float(np.percentile(inflow, 90)),
        'p95': float(np.percentile(inflow, 95)),
        'p99': float(np.percentile(inflow, 99)),
        'zero_pct': float((inflow == 0).mean() * 100),
        'lt1_pct': float((inflow < 1).mean() * 100),
        'lt5_pct': float((inflow < 5).mean() * 100),
        'outlier_pct': float(((inflow < q1-1.5*iqr) | (inflow > q3+1.5*iqr)).mean() * 100),
    },
    'precip': {
        'mean': float(pmean),
        'median': float(pmedian),
        'std': float(pstd),
        'max': float(precip.max()),
        'dry_pct': float(dry / len(precip) * 100),
        'wet_pct': float(wet / len(precip) * 100),
        'light_pct': float(light / len(precip) * 100),
        'moderate_pct': float(moderate / len(precip) * 100),
        'heavy_pct': float(heavy / len(precip) * 100),
        'violent_pct': float(violent / len(precip) * 100),
    },
    'correlations': {
        'pearson': float(corr_pearson),
        'spearman': float(corr_spearman),
    },
    'metrics': metrics,
    'inflow_ref': {
        'mean': float(inflow_mean),
        'median': float(inflow_median),
        'std': float(inflow_std),
    }
}

with open('scripts/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n\nSummary saved to scripts/analysis_summary.json")
print("=" * 80)