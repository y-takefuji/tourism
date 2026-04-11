import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr

# ============================================================
# 1. Load and Prepare Data
# ============================================================
df = pd.read_csv('Tourism_Hospitality_Industry_Analysis.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

target_col = 'Tourist_Spending_USD'

print("\nDistribution of Target Variable:")
print(df[target_col].describe())
print("\nValue Counts (top 20):")
print(df[target_col].value_counts().head(20))

all_cols    = df.columns.tolist()
string_cols = [all_cols[0], all_cols[1], all_cols[5], all_cols[10]]
print("\nString (Categorical) Columns:", string_cols)

df_encoded = df.copy()
le = LabelEncoder()
for col in string_cols:
    if col != target_col:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

print("\nFeature Columns:", X.columns.tolist())
print("Number of Features:", X.shape[1])

feature_names = X.columns.tolist()
X_array       = X.values
y_array       = y.values.astype(float)

# ============================================================
# 2. Feature Selection Functions
#    Each function returns (top_n features, highest_feature)
#    highest_feature = the single #1 ranked feature by that
#    algorithm's own criterion from the full dataset
# ============================================================

def select_rf_features(X, y, feat_names, top_n):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    importances    = rf.feature_importances_
    indices_sorted = np.argsort(importances)[::-1]
    selected = [feat_names[i] for i in indices_sorted[:top_n]]
    highest  = feat_names[indices_sorted[0]]
    return selected, highest


def select_xgb_features(X, y, feat_names, top_n):
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X, y)
    importances    = xgb.feature_importances_
    indices_sorted = np.argsort(importances)[::-1]
    selected = [feat_names[i] for i in indices_sorted[:top_n]]
    highest  = feat_names[indices_sorted[0]]
    return selected, highest


def select_lasso_features(X, y, feat_names, top_n):
    cv       = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas   = np.logspace(-4, 4, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=cv,
                       max_iter=100000, random_state=42)
    lasso_cv.fit(X, y)
    importances    = np.abs(lasso_cv.coef_)
    indices_sorted = np.argsort(importances)[::-1]
    selected = [feat_names[i] for i in indices_sorted[:top_n]]
    highest  = feat_names[indices_sorted[0]]
    return selected, highest, lasso_cv.alpha_


def select_fa_features(X, feat_names, top_n):
    n_clusters = min(X.shape[1], max(top_n * 3, 20))
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)
    labels        = fa.labels_
    raw_variances = np.var(X, axis=0)

    feature_info = [(i, labels[i], raw_variances[i])
                    for i in range(X.shape[1])]
    feature_info_sorted = sorted(feature_info,
                                  key=lambda x: x[2], reverse=True)
    seen_clusters = set()
    selected = []
    for feat_idx, clust_id, var in feature_info_sorted:
        if len(selected) >= top_n:
            break
        if clust_id not in seen_clusters:
            selected.append(feat_names[feat_idx])
            seen_clusters.add(clust_id)

    if len(selected) < top_n:
        for feat_idx, clust_id, var in feature_info_sorted:
            if len(selected) >= top_n:
                break
            fname = feat_names[feat_idx]
            if fname not in selected:
                selected.append(fname)

    # highest = #1 ranked by variance from full set
    highest = feat_names[np.argsort(raw_variances)[::-1][0]]
    return selected[:top_n], highest


def select_hvgs_features(X, feat_names, top_n):
    variances      = np.var(X, axis=0)
    indices_sorted = np.argsort(variances)[::-1]
    selected = [feat_names[i] for i in indices_sorted[:top_n]]
    highest  = feat_names[indices_sorted[0]]
    return selected, highest


def select_spearman_features(X, y, feat_names, top_n):
    correlations = []
    for i in range(X.shape[1]):
        corr, _ = spearmanr(X[:, i], y)
        correlations.append(abs(corr))
    correlations   = np.array(correlations)
    indices_sorted = np.argsort(correlations)[::-1]
    selected = [feat_names[i] for i in indices_sorted[:top_n]]
    highest  = feat_names[indices_sorted[0]]
    return selected, highest

# ============================================================
# 3. Cross-Validation (R² clipped to 0)
# ============================================================
def cross_validate_features(X_data, y, feat_names,
                             selected_features, method):
    indices = [feat_names.index(f) for f in selected_features]
    X_sel   = X_data[:, indices]

    if method in ['FA', 'HVGS', 'Spearman', 'RF']:
        model = RandomForestRegressor(random_state=42)
    elif method == 'XGB':
        model = XGBRegressor(random_state=42)
    elif method == 'LASSO':
        model = LassoCV(
            alphas=np.logspace(-4, 4, 100),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            max_iter=100000,
            random_state=42
        )

    scores         = cross_val_score(model, X_sel, y, cv=5, scoring='r2')
    scores_clipped = np.clip(scores, a_min=0.0, a_max=None)

    print(f"  [{method}] Raw fold R²  : {np.round(scores, 4)}")
    print(f"  [{method}] Clipped R²   : {np.round(scores_clipped, 4)}")

    return round(scores_clipped.mean(), 4)

# ============================================================
# 4. Top 6 — FULL dataset, each algorithm independently
# ============================================================
print("\n" + "="*60)
print("FEATURE SELECTION FROM FULL DATASET (Top 6)")
print("="*60)

top6_rf,   highest_rf  = select_rf_features(
    X_array, y_array, feature_names, top_n=6)

top6_xgb,  highest_xgb = select_xgb_features(
    X_array, y_array, feature_names, top_n=6)

top6_lasso, highest_lasso, best_alpha_full = select_lasso_features(
    X_array, y_array, feature_names, top_n=6)

top6_fa,   highest_fa  = select_fa_features(
    X_array, feature_names, top_n=6)

top6_hvgs, highest_hvgs = select_hvgs_features(
    X_array, feature_names, top_n=6)

top6_spearman, highest_spearman = select_spearman_features(
    X_array, y_array, feature_names, top_n=6)

print(f"RF       Top 6 : {top6_rf}")
print(f"XGB      Top 6 : {top6_xgb}")
print(f"LASSO    Top 6 : {top6_lasso}  (alpha={best_alpha_full:.6f})")
print(f"FA       Top 6 : {top6_fa}")
print(f"HVGS     Top 6 : {top6_hvgs}")
print(f"Spearman Top 6 : {top6_spearman}")

print(f"\nHighest feature per algorithm:")
print(f"  RF       #1 : {highest_rf}")
print(f"  XGB      #1 : {highest_xgb}")
print(f"  LASSO    #1 : {highest_lasso}")
print(f"  FA       #1 : {highest_fa}")
print(f"  HVGS     #1 : {highest_hvgs}")
print(f"  Spearman #1 : {highest_spearman}")

# ============================================================
# 5. CV with Top 6
# ============================================================
print("\n" + "="*60)
print("5-FOLD CV — TOP 6 FEATURES (R²)")
print("="*60)

cv6_rf       = cross_validate_features(
    X_array, y_array, feature_names, top6_rf,       'RF')
cv6_xgb      = cross_validate_features(
    X_array, y_array, feature_names, top6_xgb,      'XGB')
cv6_lasso    = cross_validate_features(
    X_array, y_array, feature_names, top6_lasso,    'LASSO')
cv6_fa       = cross_validate_features(
    X_array, y_array, feature_names, top6_fa,       'FA')
cv6_hvgs     = cross_validate_features(
    X_array, y_array, feature_names, top6_hvgs,     'HVGS')
cv6_spearman = cross_validate_features(
    X_array, y_array, feature_names, top6_spearman, 'Spearman')

print(f"\nRF       CV6 R²: {cv6_rf}")
print(f"XGB      CV6 R²: {cv6_xgb}")
print(f"LASSO    CV6 R²: {cv6_lasso}")
print(f"FA       CV6 R²: {cv6_fa}")
print(f"HVGS     CV6 R²: {cv6_hvgs}")
print(f"Spearman CV6 R²: {cv6_spearman}")

# ============================================================
# 6. Each algorithm builds its OWN reduced dataset
#    by removing its OWN highest feature from the full dataset
#    Then selects top 5 from that reduced dataset
# ============================================================
print("\n" + "="*60)
print("BUILDING REDUCED DATASETS — one per algorithm")
print("Each removes its own #1 feature from full dataset")
print("="*60)

def build_reduced(X_df, highest):
    X_red     = X_df.drop(columns=[highest])
    red_names = X_red.columns.tolist()
    red_arr   = X_red.values
    print(f"  Removed '{highest}' -> {X_df.shape[1]-1} cols remain")
    assert highest not in red_names, \
        f"FATAL: '{highest}' still present after drop!"
    return red_arr, red_names

X_red_rf,       names_red_rf       = build_reduced(X, highest_rf)
X_red_xgb,      names_red_xgb      = build_reduced(X, highest_xgb)
X_red_lasso,    names_red_lasso    = build_reduced(X, highest_lasso)
X_red_fa,       names_red_fa       = build_reduced(X, highest_fa)
X_red_hvgs,     names_red_hvgs     = build_reduced(X, highest_hvgs)
X_red_spearman, names_red_spearman = build_reduced(X, highest_spearman)

# ============================================================
# 7. Top 5 — each algorithm selects from its OWN reduced dataset
# ============================================================
print("\n" + "="*60)
print("FEATURE SELECTION FROM REDUCED DATASET (Top 5)")
print("="*60)

top5_rf,    _ = select_rf_features(
    X_red_rf, y_array, names_red_rf, top_n=5)

top5_xgb,   _ = select_xgb_features(
    X_red_xgb, y_array, names_red_xgb, top_n=5)

top5_lasso, _, best_alpha_red = select_lasso_features(
    X_red_lasso, y_array, names_red_lasso, top_n=5)

top5_fa,    _ = select_fa_features(
    X_red_fa, names_red_fa, top_n=5)

top5_hvgs,  _ = select_hvgs_features(
    X_red_hvgs, names_red_hvgs, top_n=5)

top5_spearman, _ = select_spearman_features(
    X_red_spearman, y_array, names_red_spearman, top_n=5)

print(f"RF       Top 5 : {top5_rf}")
print(f"XGB      Top 5 : {top5_xgb}")
print(f"LASSO    Top 5 : {top5_lasso}  (alpha={best_alpha_red:.6f})")
print(f"FA       Top 5 : {top5_fa}")
print(f"HVGS     Top 5 : {top5_hvgs}")
print(f"Spearman Top 5 : {top5_spearman}")

# ============================================================
# 8. Verification — each algorithm's highest must not appear
#    in its own top 5
# ============================================================
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

checks = [
    ('RF',       highest_rf,       top5_rf),
    ('XGB',      highest_xgb,      top5_xgb),
    ('LASSO',    highest_lasso,    top5_lasso),
    ('FA',       highest_fa,       top5_fa),
    ('HVGS',     highest_hvgs,     top5_hvgs),
    ('Spearman', highest_spearman, top5_spearman),
]

all_clear = True
for method_name, hf, t5 in checks:
    if hf in t5:
        print(f"  FAIL [{method_name}]: '{hf}' still in top5 -> {t5}")
        all_clear = False
    else:
        print(f"  PASS [{method_name}]: '{hf}' absent from top5 -> {t5}")

if all_clear:
    print("\nAll checks passed.")
else:
    raise RuntimeError("Highest feature still appears in top5!")

# ============================================================
# 9. Summary Table -> result.csv
# ============================================================
summary = {
    'Method'        : ['RF', 'XGB', 'LASSO', 'FA', 'HVGS', 'Spearman'],
    'CV6 R2'        : [cv6_rf, cv6_xgb, cv6_lasso,
                       cv6_fa, cv6_hvgs, cv6_spearman],
    'Top 6 Features': [
        ', '.join(top6_rf),
        ', '.join(top6_xgb),
        ', '.join(top6_lasso),
        ', '.join(top6_fa),
        ', '.join(top6_hvgs),
        ', '.join(top6_spearman)
    ],
    'Top 5 Features': [
        ', '.join(top5_rf),
        ', '.join(top5_xgb),
        ', '.join(top5_lasso),
        ', '.join(top5_fa),
        ', '.join(top5_hvgs),
        ', '.join(top5_spearman)
    ]
}

result_df = pd.DataFrame(summary)
result_df.to_csv('result.csv', index=False)

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(result_df.to_string(index=False))
print("\nSaved as result.csv")
