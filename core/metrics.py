# -*- coding: utf-8 -*-
"""
核心指标库 —— 严格按 m2/m3 知识库口径实现
- KS: m2 §2.1 (max|cum_bad - cum_good|)
- AUC: sklearn roc_auc_score
- PSI: Σ(Ai-Ei)*ln(Ai/Ei)，模型分/特征通用
- IV: 特征分箱后 Σ(good%-bad%)*ln(good%/bad%)
- 坏账率/通过率/坏账捕获率 Top10%
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def ks_score(y_true, y_pred):
    """m2 §2.1 知识库原版实现"""
    df = pd.DataFrame({'y': np.asarray(y_true), 'pred': np.asarray(y_pred)})
    df = df.sort_values('pred', ascending=False)
    df['cum_bad'] = df['y'].cumsum() / df['y'].sum()
    df['cum_good'] = (1 - df['y']).cumsum() / (1 - df['y']).sum()
    df['ks'] = (df['cum_bad'] - df['cum_good']).abs()
    return float(df['ks'].max())


def auc_score(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))


def psi(expected, actual, bins=10):
    """PSI = Σ(Ai-Ei)*ln(Ai/Ei)。expected=基线分布, actual=当前分布。"""
    e = pd.Series(np.asarray(expected, dtype=float))
    a = pd.Series(np.asarray(actual, dtype=float))
    qs = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    e_pct = np.clip(pd.cut(e, qs).value_counts(normalize=True).sort_index().values, EPS, None)
    a_pct = np.clip(pd.cut(a, qs).value_counts(normalize=True).sort_index().values, EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_categorical(expected, actual):
    """类特征 PSI：按类别占比算，数值型低基数特征也可走这个"""
    e = pd.Series(expected)
    a = pd.Series(actual)
    cats = sorted(set(e.unique()) | set(a.unique()))
    v = 0.0
    for c in cats:
        ep = np.clip((e == c).mean(), EPS, None)
        ap = np.clip((a == c).mean(), EPS, None)
        v += (ap - ep) * np.log(ap / ep)
    return float(v)


def iv_score(feature, y, bins=10):
    """IV = Σ(good% - bad%) * WOE，WOE = ln(good%/bad%)"""
    f = pd.Series(np.asarray(feature, dtype=float))
    y = pd.Series(np.asarray(y))
    qs = np.unique(np.quantile(f.dropna(), np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    grp = pd.cut(f, qs)
    t = pd.DataFrame({'g': grp, 'y': y}).groupby('g', observed=True)['y'].agg(['count', 'sum'])
    good = np.clip((t['count'] - t['sum']) / max((y == 0).sum(), 1), EPS, None)
    bad = np.clip(t['sum'] / max((y == 1).sum(), 1), EPS, None)
    woe = np.log(good / bad)
    return float(np.sum((good - bad) * woe))


def bad_rate(y):
    return float(np.mean(y))


def pass_rate(score, threshold=0.5):
    """通过率：score < threshold 判好放行（本项目 score 为坏概率）"""
    return float((np.asarray(score) < threshold).mean())


def capture_rate_topk(y, score, k=0.10):
    """坏账捕获率：最高分前 k 中的坏样本占比"""
    n = len(y)
    top_n = max(int(round(n * k)), 1)
    idx = np.argsort(-np.asarray(score))[:top_n]
    return float(np.mean(np.asarray(y)[idx]))


def predict_bad_rate(score, threshold=0.5):
    """m5 护栏口径：固定阈值下预测为坏（可疑）的占比"""
    return float((np.asarray(score) >= threshold).mean())


def full_metric_pack(y_true, score, base_score=None, threshold=0.5):
    """m5 §二 6 项核心指标打包"""
    out = {
        'KS': ks_score(y_true, score),
        'AUC': auc_score(y_true, score),
        'bad_rate': bad_rate(y_true),
        'pass_rate': pass_rate(score, threshold),
        'capture_top10': capture_rate_topk(y_true, score, 0.10),
        'predict_bad_rate': predict_bad_rate(score, threshold),
    }
    if base_score is not None:
        out['score_psi_vs_base'] = psi(base_score, score)
    return out


# ---------- 显著性检验：配对 AUC 的 DeLong（结构协方差 z 统计量） ----------
def _delong_components(pos, neg):
    """返回 V10(每正样本)、V01(每负样本) 结构成分。pos=正样本分数, neg=负样本分数。

    AUC 的 φ 结构：每个正样本对所有负样本的"胜场比例"、每个负样本对所有正样本的
    "负场比例"。对 ties（结值）用 (left+right)/2 平分。连续分数 ties 极少，与精确解一致。
    """
    m, n = len(pos), len(neg)
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn, pos, "left") + np.searchsorted(sn, pos, "right")) / 2.0 / n
    v01 = 1.0 - (np.searchsorted(sp, neg, "left") + np.searchsorted(sp, neg, "right")) / 2.0 / m
    return v10, v01


def delong_auc_test(y_true, score1, score2):
    """配对样本的 AUC 差异 DeLong 检验（结构协方差法）。

    返回 (auc1, auc2, z, p)，z>0 表示 score1 的 AUC 更高，p 为双侧。
    对两个分类器在相同样本上的 V10/V01 结构成分求 2×2 协方差；
    ΔAUC 的方差 = Var(ΔV10)/m + Var(ΔV01)/n，z = ΔAUC / sqrt(var) ~ N(0,1)。
    相比 bootstrap 近似：给出解析方差与标准 p 值，是为"严格化"的合格替代。
    """
    y = np.asarray(y_true)
    s1 = np.asarray(score1, float)
    s2 = np.asarray(score2, float)
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    if m == 0 or n == 0:
        return auc_score(y, s1), auc_score(y, s2), 0.0, 1.0
    pos = y == 1
    v10a, v01a = _delong_components(s1[pos], s1[~pos])
    v10b, v01b = _delong_components(s2[pos], s2[~pos])
    auc1, auc2 = float(v10a.mean()), float(v10b.mean())
    s10 = np.cov(np.vstack([v10a, v10b]))          # 正样本侧 2×2 协方差
    s01 = np.cov(np.vstack([v01a, v01b]))          # 负样本侧 2×2 协方差
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    var = max(float(var), 1e-12)
    z = (auc1 - auc2) / np.sqrt(var)
    import math
    p = math.erfc(abs(z) / math.sqrt(2.0))          # 双侧 p
    return auc1, auc2, float(z), float(p)


def gain_topk(artifact, k=10, X=None, y=None):
    """模型 top-k 特征贡献度（归一化）。

    树模型取内建 gain/importance；LR 等无线性内建重要性时，若提供 X/y 则用
    排列重要性（permutation，AUC 下降幅度）近似，否则退化为 |coef| 归一化。
    返回 [{feature, gain}] 按 gain 降序，长度 ≤k。
    """
    algo = artifact["algo"]
    m = artifact["model"]
    feats = list(artifact["features"])
    imp = None
    if algo in ("xgb", "lgb") and hasattr(m, "feature_importances_"):
        imp = np.asarray(m.feature_importances_, dtype=float)
    elif algo == "lr":
        try:
            imp = np.abs(m.named_steps["logisticregression"].coef_[0]).astype(float)
        except Exception:
            imp = None
    if imp is None and X is not None and y is not None:
        if algo == "lr" and hasattr(m, "predict_proba"):
            base = auc_score(y, m.predict_proba(X)[:, 1])
            imp = np.zeros(len(feats))
            rng = np.random.default_rng(42)
            Xa = np.asarray(X, dtype=float)
            for j in range(len(feats)):
                saved = Xa[:, j].copy()
                rng.shuffle(Xa[:, j])
                imp[j] = max(0.0, base - auc_score(y, m.predict_proba(Xa)[:, 1]))
                Xa[:, j] = saved
    if imp is None:
        imp = np.ones(len(feats))
    tot = imp.sum()
    imp = imp / tot if tot > 0 else np.ones(len(feats)) / len(feats)
    pairs = sorted(zip(feats, imp), key=lambda kv: -kv[1])[:k]
    return [{"feature": f, "gain": float(g)} for f, g in pairs]
