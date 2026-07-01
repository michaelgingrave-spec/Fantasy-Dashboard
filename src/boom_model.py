"""
boom_model.py
Train a weekly boom-week classifier per position.
A "boom" is defined as hitting the 90th percentile FP threshold for that position:
  WR >= 18.1,  TE >= 12.3,  RB >= 20.2

For each player in the 2025 season, predict the number of expected boom weeks
and a per-week boom probability profile.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

from features import WEEKLY_FEATURES, BOOM_THRESHOLDS, build_weekly_features


def _prep(df: pd.DataFrame, pos: str):
    sub = df[df['POS'] == pos].copy()
    cols = [c for c in WEEKLY_FEATURES if c in sub.columns]
    X = sub[cols].fillna(0)
    y = sub['boom']
    return X, y, sub


def train_boom_models(weekly_df: pd.DataFrame, season_stats: pd.DataFrame) -> dict:
    """
    Train boom classifiers for WR, TE, RB using all years' weekly data.
    Returns dict of {pos: {'model', 'feature_names', 'cv_results'}}.
    """
    feat_df = build_weekly_features(weekly_df, season_stats)
    results = {}

    for pos in ['WR', 'TE', 'RB']:
        X, y, sub = _prep(feat_df, pos)
        years = sorted(sub['Year'].unique())

        cv_results = []
        for holdout_year in years:
            mask_train = sub['Year'] != holdout_year
            mask_test  = sub['Year'] == holdout_year

            X_train, y_train = X[mask_train], y[mask_train]
            X_test,  y_test  = X[mask_test],  y[mask_test]

            if y_test.sum() < 5:
                continue

            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                reg_alpha=0.1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
            )
            clf.fit(X_train, y_train, verbose=False)
            probs = clf.predict_proba(X_test)[:, 1]

            auc  = roc_auc_score(y_test, probs)
            ap   = average_precision_score(y_test, probs)
            boom_rate = y_test.mean()

            cv_results.append({
                'holdout_year': holdout_year,
                'n_test':       len(y_test),
                'boom_rate':    round(boom_rate, 3),
                'AUC':          round(auc, 3),
                'AvgPrecision': round(ap, 3),
            })

        # Final model on all data, with probability calibration
        base_clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            reg_alpha=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
        # Calibrate using isotonic regression for better-calibrated probabilities
        final_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
        final_clf.fit(X, y)

        feat_names = [c for c in WEEKLY_FEATURES if c in feat_df.columns]
        results[pos] = {
            'model':         final_clf,
            'feature_names': feat_names,
            'cv_results':    cv_results,
            'boom_threshold': BOOM_THRESHOLDS[pos],
        }

    return results


def project_boom_weeks(weekly_df: pd.DataFrame, season_stats: pd.DataFrame,
                       boom_models: dict, target_year: int = 2025) -> pd.DataFrame:
    """
    For each player's weeks in `target_year`, predict boom probability
    using the trained boom models on that week's features.

    Returns a DataFrame with one row per (player, week) including:
      - boom_prob: probability of boom that week
      - actual_boom: whether they actually boomed (1/0)
      - actual_FP: actual FP that week

    Also returns a summary DataFrame per player with:
      - Expected boom weeks (sum of probabilities)
      - Predicted boom weeks (count of weeks with prob > 0.5)
      - Actual boom weeks
    """
    feat_df = build_weekly_features(weekly_df, season_stats)
    target_df = feat_df[feat_df['Year'] == target_year].copy()

    weekly_results = []

    for pos in ['WR', 'TE', 'RB']:
        sub = target_df[target_df['POS'] == pos].copy()
        result = boom_models[pos]
        feat_names = result['feature_names']

        X = sub[feat_names].fillna(0)
        probs = result['model'].predict_proba(X)[:, 1]

        sub = sub.copy()
        sub['boom_prob']   = probs.round(3)
        sub['actual_boom'] = sub['boom']
        sub['actual_FP']   = sub['FP']

        weekly_results.append(sub[['Name', 'Team', 'POS', 'Year', 'WEEK',
                                   'boom_prob', 'actual_boom', 'actual_FP']])

    weekly_out = pd.concat(weekly_results, ignore_index=True)

    # Player summary
    summary = weekly_out.groupby(['Name', 'Team', 'POS']).agg(
        Weeks_Played       = ('WEEK',        'count'),
        Exp_Boom_Weeks     = ('boom_prob',   'sum'),
        Pred_Boom_Weeks    = ('boom_prob',   lambda x: (x >= 0.35).sum()),
        Actual_Boom_Weeks  = ('actual_boom', 'sum'),
        Avg_Boom_Prob      = ('boom_prob',   'mean'),
        Max_Boom_Prob      = ('boom_prob',   'max'),
    ).reset_index()

    summary['Exp_Boom_Weeks']  = summary['Exp_Boom_Weeks'].round(1)
    summary['Avg_Boom_Prob']   = summary['Avg_Boom_Prob'].round(3)
    summary['Max_Boom_Prob']   = summary['Max_Boom_Prob'].round(3)

    return weekly_out, summary
