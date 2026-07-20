# -*- coding: utf-8 -*-
"""
V8 — Étape 4/4 : les cinq graphiques de l'article.

Entrées : v8_features.csv, v8_model_lgbm.txt   (étapes 2 et 3)
Sorties : v8_g1_importance.png ... v8_g5_case.png

Progression pédagogique :
  1. IMPORTANCE   — quels signaux comptent
  2. DIRECTION    — dans quel sens ils jouent
  3. GAINS        — ce que ça donne en pratique (auditer X %, capter Y %)
  4. CALIBRATION  — peut-on lire les scores comme des probabilités
  5. CAS RÉEL     — une société décomposée facteur par facteur

Charte reprise de la série « Prévoir l'électricité » : fond marine profond, titre
capitales, badge du chiffre clé, phrase de conclusion en pied — et un accent
différent par graphique (jaune, cyan, vert, magenta) pour rythmer la série.

Dépendances : pandas numpy lightgbm shap matplotlib
"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AUTEUR = "Thomas Le Joubioux"
try:
    import lightgbm as lgb, shap
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError as e:
    sys.exit(f"Dépendance manquante ({e.name}) :  pip install lightgbm shap matplotlib")

# ────────────────────────────────────────────────────────────── charte
NAVY    = "#0C1230"   # fond de figure
PANEL   = "#161C42"   # zone de tracé
GRID    = "#252C55"   # grille
WHITE   = "#FFFFFF"
LIGHT   = "#B8C0D9"   # sous-titres, pieds
MUTED   = "#8C95B5"   # graduations
SANS    = {"family": "sans-serif"}

# un accent par graphique — rythme visuel de la série
YELLOW  = "#FFE500"
CYAN    = "#4FC3E8"
MAGENTA = "#FF3D8B"
GREEN   = "#3DED97"

SOURCE = "Données publiques Nasdaq & NYSE 1999-2018  ·  Modèle et graphique : " + AUTEUR


def canevas(titre, sous_titre, badge, pied, h=7.0, accent=YELLOW):
    """Figure éditoriale sombre : titre capitales, badge coloré, pied de page."""
    fig = plt.figure(figsize=(14, h), dpi=170)
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0.255, 0.175, 0.715, 0.545])
    ax.set_facecolor(PANEL)

    fig.text(0.048, 0.955, titre.upper(), color=WHITE, fontsize=25, fontweight="bold",
             va="top", **SANS)
    fig.text(0.048, 0.882, sous_titre, color=LIGHT, fontsize=11.5, va="top", **SANS)

    if badge:
        fig.text(0.048, 0.812, f"  {badge}  ", color=NAVY, fontsize=11.5,
                 fontweight="bold", va="center", **SANS,
                 bbox=dict(boxstyle="round,pad=0.42", facecolor=accent, edgecolor="none"))

    fig.text(0.048, 0.052, pied, color=LIGHT, fontsize=10.5, style="italic", **SANS)
    fig.text(0.968, 0.052, SOURCE, color=MUTED, fontsize=9, ha="right", **SANS)

    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.grid(color=GRID, lw=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    return fig, ax


def enregistrer(fig, nom):
    p = SCRIPT_DIR / nom
    fig.savefig(p, facecolor=NAVY, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p.name)


# ────────────────────────────────────────────────── libellés en clair
LIB = {
 "altman_z":"Altman Z-score", "re_to_assets":"Retained earnings / assets",
 "retained_earnings":"Retained earnings", "market_to_book":"Market-to-book",
 "market_value":"Market capitalisation", "market_to_sales":"Market cap / sales",
 "debt_to_assets":"Total debt / assets", "lt_debt_to_assets":"Long-term debt / assets",
 "debt_to_equity":"Debt / equity", "equity_ratio":"Equity / assets",
 "current_ratio":"Current ratio", "quick_ratio":"Quick ratio",
 "working_capital_ratio":"Working capital / assets", "working_capital":"Working capital",
 "roa":"Return on assets", "roe":"Return on equity", "net_margin":"Net margin",
 "ebit_margin":"EBIT margin", "ebitda_margin":"EBITDA margin", "gross_margin":"Gross margin",
 "asset_turnover":"Asset turnover", "inventory_turnover":"Inventory turnover",
 "receivables_turnover":"Receivables turnover", "opex_ratio":"Operating costs / sales",
 "cogs_ratio":"Cost of goods / sales", "da_ratio":"Depreciation / assets",
 "total_assets":"Total assets", "total_liabilities":"Total liabilities",
 "total_long_term_debt":"Long-term debt", "net_sales":"Net sales", "ebit":"EBIT",
 "ebitda":"EBITDA", "net_income":"Net income", "gross_profit":"Gross profit",
 "inventory":"Inventory", "current_assets":"Current assets", "equity":"Equity",
 "total_current_liabilities":"Current liabilities", "total_receivables":"Receivables",
 "cost_of_goods_sold":"Cost of goods sold", "total_revenue":"Total revenue",
 "total_operating_expenses":"Operating expenses", "depreciation_amortization":"Depreciation",
 "anciennete":"Company age", "fyear":"Fiscal year",
 "majorgroup":"Industry (minor)", "division":"Industry (major)",
 "baisses_resultat_3ans":"Consecutive earnings declines",
 "croissance_ca":"Sales growth", "croissance_actif":"Asset growth",
 "croissance_ebitda":"EBITDA growth", "croissance_resultat":"Earnings growth",
 "flag_equity_negatif":"Negative equity", "flag_fdr_negatif":"Negative working capital",
 "flag_ebitda_negatif":"Negative EBITDA", "flag_resultat_negatif":"Loss-making",
 "flag_report_negatif":"Accumulated deficit",
}
SUF = {"_d1":" (1-yr change)", "_d2":" (2-yr change)", "_d3":" (3-yr change)",
       "_vol4":" (4-yr volatility)"}


def joli(c):
    for s, t in SUF.items():
        if c.endswith(s):
            b = c[:-len(s)]
            return LIB.get(b, b.replace("_", " ").capitalize()) + t
    return LIB.get(c, c.replace("_", " ").capitalize())


def charger():
    spec = importlib.util.spec_from_file_location("m3", SCRIPT_DIR / "03_models.py")
    m3 = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(m3)
    df, parts, feat = m3.charger(False)
    X, Xs, Y = m3.preparer(parts, feat)
    model = lgb.Booster(model_file=str(SCRIPT_DIR / "v8_model_lgbm.txt"))
    if model.num_feature() != len(feat):
        df, parts, feat = m3.charger(True)
        X, Xs, Y = m3.preparer(parts, feat)
    return m3, parts, feat, X, Y, model


def main():
    m3, parts, feat, X, Y, model = charger()
    Xte, yte, te = X["test"], Y["test"], parts["test"]
    p = model.predict(Xte)
    auc = m3.auc_mw(yte, p)
    n_soc = te.company_name.nunique()
    base = 100 * yte.mean()
    print(f"Test 2015-2018 : {n_soc:,} sociétés · {int(yte.sum())} défauts · AUC {auc:.3f}\n")

    print("Calcul des valeurs SHAP...")
    sv = shap.TreeExplainer(model).shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    S = pd.DataFrame(sv, columns=feat)
    imp = S.abs().mean().sort_values(ascending=False)
    TOP = 12
    top = imp.head(TOP).index[::-1]
    labs = [joli(c) for c in top]
    yy = np.arange(TOP)
    print("\nGraphiques :")

    # ───────────────────────────────────────── 1 · IMPORTANCE
    fig, ax = canevas(
        "Quels signaux annoncent la défaillance ?",
        f"Poids de chaque indicateur dans les prédictions du modèle — {n_soc:,} sociétés testées sur 2015-2018 (données publiques Nasdaq & NYSE)",
        f"AUC 0,{str(round(auc,3)).split('.')[1]}  ·  {int(yte.sum())} défaillances observées  ·  {len(feat)} indicateurs analysés",
        "L'endettement et les réserves accumulées dominent — la lecture des comptes reste le meilleur signal.", h=7.6, accent=YELLOW)
    v = imp[top].values
    alphas = np.linspace(0.42, 1.0, TOP)
    for i, (w, al) in enumerate(zip(v, alphas)):
        ax.barh(i, w, color=YELLOW, alpha=al, height=0.62, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(labs, color=WHITE, fontsize=11)
    ax.set_xlim(0, v.max() * 1.15); ax.set_ylim(-0.7, TOP - 0.3)
    for i, w in zip(yy, v):
        ax.text(w * 1.02, i, f"{w:.3f}", va="center", fontsize=9.5, color=LIGHT, **SANS)
    ax.set_xlabel("influence moyenne sur le risque estimé", color=MUTED, fontsize=10, labelpad=9, **SANS)
    ax.grid(axis="y", lw=0)
    enregistrer(fig, "v8_g1_importance.png")

    # ───────────────────────────────────────── 2 · DIRECTION
    hi, lo = [], []
    for c in top:
        x = Xte[:, feat.index(c)]; s = S[c].values; b = np.nanmean(s)
        qh, ql = np.nanquantile(x, .75), np.nanquantile(x, .25)
        hi.append(np.nanmean(s[x >= qh]) - b if (x >= qh).any() else 0)
        lo.append(np.nanmean(s[x <= ql]) - b if (x <= ql).any() else 0)
    hi, lo = np.array(hi), np.array(lo)
    j = int(np.argmax(np.abs(hi - lo)))
    fig, ax = canevas(
        "Dans quel sens joue chaque signal ?",
        "Effet moyen sur le risque estimé selon que l'indicateur est élevé (jaune) ou faible (bleu)",
        f"Signal le plus tranché : {joli(top[j])}",
        "Un indicateur peut peser fort et jouer dans les deux sens — c'est son niveau qui décide.", h=7.6, accent=CYAN)
    ax.barh(yy + 0.19, hi, color=YELLOW, height=0.31, zorder=3, label="valeur élevée")
    ax.barh(yy - 0.19, lo, color=CYAN,   height=0.31, zorder=3, label="valeur faible")
    ax.set_yticks(yy); ax.set_yticklabels(labs, color=WHITE, fontsize=11)
    m = max(np.abs(hi).max(), np.abs(lo).max()) * 1.18
    ax.set_xlim(-m, m); ax.set_ylim(-0.75, TOP - 0.25)
    ax.axvline(0, color=LIGHT, lw=1.2, zorder=4)
    ax.set_xlabel("← diminue le risque          augmente le risque →", color=MUTED,
                  fontsize=10, labelpad=9, **SANS)
    ax.grid(axis="y", lw=0)
    lg = ax.legend(loc="lower right", frameon=True, fontsize=10)
    lg.get_frame().set_facecolor(NAVY); lg.get_frame().set_edgecolor(GRID)
    for t in lg.get_texts(): t.set_color(WHITE)
    enregistrer(fig, "v8_g2_direction.png")

    # ───────────────────────────────────────── 3 · GAINS
    o = np.argsort(-p); yo = yte[o]
    frac = np.arange(1, len(yo) + 1) / len(yo) * 100
    capt = np.cumsum(yo) / max(yo.sum(), 1) * 100
    r20 = capt[int(len(yo) * .2) - 1]
    fig, ax = canevas(
        "Combien de défaillances pour combien d'efforts ?",
        "Part des défaillances réelles captées en n'examinant que les dossiers les plus risqués",
        f"{r20:.0f} % des défaillances captées en examinant 20 % des dossiers — {r20/20:.1f}× mieux que le hasard",
        "Un dossier sur douze audité part en défaut, contre un sur quarante-huit au hasard.", h=7.2, accent=GREEN)
    ax.fill_between(frac, capt, color=GREEN, alpha=0.16, zorder=2)
    ax.plot(frac, capt, color=GREEN, lw=3, zorder=4, label="avec le modèle")
    ax.plot([0, 100], [0, 100], color=WHITE, lw=1.6, ls="--", zorder=3, label="sans modèle")
    for q in (10, 20, 30):
        r = capt[int(len(yo) * q / 100) - 1]
        big = q == 20
        ax.scatter([q], [r], s=90 if big else 55, color=GREEN if big else WHITE,
                   zorder=6, edgecolor=NAVY, lw=1.6)
        ax.annotate(f"{r:.0f} %", (q, r), textcoords="offset points", xytext=(10, -16),
                    fontsize=12 if big else 10, color=GREEN if big else LIGHT,
                    fontweight="bold" if big else "normal", **SANS)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel("% de dossiers examinés", color=MUTED, fontsize=10, labelpad=9, **SANS)
    ax.set_ylabel("% de défaillances captées", color=MUTED, fontsize=10, labelpad=9, **SANS)
    lg = ax.legend(loc="lower right", frameon=True, fontsize=10)
    lg.get_frame().set_facecolor(NAVY); lg.get_frame().set_edgecolor(GRID)
    for t in lg.get_texts(): t.set_color(WHITE)
    enregistrer(fig, "v8_g3_gains.png")

    # ───────────────────────────────────────── 4 · CALIBRATION
    f = SCRIPT_DIR / "v8_predictions_test.csv"
    pc = pd.read_csv(f)["proba_calibree"].values if f.exists() else p
    q = pd.qcut(pd.Series(pc).rank(method="first"), 10, labels=False)
    obs = pd.Series(yte).groupby(q).mean().values * 100
    pred = pd.Series(pc).groupby(q).mean().values * 100
    fig, ax = canevas(
        "Peut-on lire ces scores comme des probabilités ?",
        "Probabilité annoncée par le modèle face au taux de défaillance réellement observé, par tranche de risque",
        f"Tranche la plus risquée : {obs[-1]:.0f} % de défaillances réelles  ·  tranche la plus sûre : {obs[0]:.1f} %",
        "Les deux barres coïncident : le modèle annonce ce qui se produit — ses probabilités sont exploitables.", h=7.2, accent=MAGENTA)
    xx = np.arange(10); w = 0.38
    ax.bar(xx - w/2, pred, width=w, color=MAGENTA, zorder=3, label="probabilité annoncée")
    ax.bar(xx + w/2, obs,  width=w, color=WHITE,  zorder=3, label="défaillances observées")
    ax.set_xticks(xx); ax.set_xticklabels([f"T{i+1}" for i in xx], color=LIGHT, fontsize=10)
    ax.set_xlabel("tranche de risque  (T1 = les plus sûres, T10 = les plus risquées)",
                  color=MUTED, fontsize=10, labelpad=9, **SANS)
    ax.set_ylabel("taux de défaillance, %", color=MUTED, fontsize=10, labelpad=9, **SANS)
    ax.grid(axis="x", lw=0)
    lg = ax.legend(loc="upper left", frameon=True, fontsize=10)
    lg.get_frame().set_facecolor(NAVY); lg.get_frame().set_edgecolor(GRID)
    for t in lg.get_texts(): t.set_color(WHITE)
    enregistrer(fig, "v8_g4_calibration.png")

    # ───────────────────────────────────────── 5 · CAS RÉEL
    cand = np.where((yte == 1) & (p > np.quantile(p, 0.97)))[0]
    i = int(cand[np.argmax(p[cand])]) if len(cand) else int(np.argmax(p))
    row = S.iloc[i]
    ordre = row.abs().sort_values(ascending=False).head(9).index[::-1]
    vals = row[ordre].values
    fig, ax = canevas(
        "Une société, décomposée",
        f"Pourquoi le modèle a classé ce dossier en risque élevé — exercice {int(te.fyear.iloc[i])}, "
        f"défaillance survenue dans les trois ans",
        f"Risque estimé : {100*pc[i]:.0f} %  ·  taux de base du portefeuille : {base:.1f} %",
        "Chaque barre est un fait comptable : c'est ce qui rend la décision auditable.", h=7.2, accent=YELLOW)
    y2 = np.arange(len(ordre))
    ax.barh(y2, np.where(vals > 0, vals, 0), color=YELLOW, height=0.6, zorder=3)
    ax.barh(y2, np.where(vals < 0, vals, 0), color=CYAN,   height=0.6, zorder=3)
    ax.set_yticks(y2); ax.set_yticklabels([joli(c) for c in ordre], color=WHITE, fontsize=11)
    mm = np.abs(vals).max() * 1.3
    ax.set_xlim(-mm, mm); ax.set_ylim(-0.7, len(ordre) - 0.3)
    ax.axvline(0, color=LIGHT, lw=1.2, zorder=4)
    for k, w2 in zip(y2, vals):
        ax.text(w2 + (0.03 * mm if w2 > 0 else -0.03 * mm), k, f"{w2:+.2f}", va="center",
                ha="left" if w2 > 0 else "right", fontsize=9.5, color=LIGHT, **SANS)
    ax.set_xlabel("← a réduit le risque          a augmenté le risque →", color=MUTED,
                  fontsize=10, labelpad=9, **SANS)
    ax.grid(axis="y", lw=0)
    enregistrer(fig, "v8_g5_case.png")

    print(f"\nCinq graphiques écrits dans {SCRIPT_DIR.name}")


if __name__ == "__main__":
    main()
