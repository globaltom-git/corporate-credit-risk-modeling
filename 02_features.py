# -*- coding: utf-8 -*-
"""
V8 — Étape 2/4 : ré-étiquetage à horizon, ratios financiers, score d'Altman,
variables de trajectoire.

Entrée  : v8_base.csv   (produit par 01_build_dataset.py)
Sortie  : v8_features.csv

Trois apports, tous hérités de V7 mais appliqués pour la première fois à des
colonnes correctement identifiées :

1. ÉTIQUETAGE À HORIZON. Le fichier public marque « failed » TOUS les exercices
   d'une société défaillante, y compris ceux où elle était saine. Le README du
   dataset annonce pourtant l'intention inverse (« we label the fiscal year before
   the chapter filling »). On restreint donc le label aux HORIZON derniers
   exercices : la question devient « défaut sous HORIZON ans ? ».

2. RATIOS + SCORE D'ALTMAN. Calculables pour la première fois : les cinq
   composantes d'Altman (fonds de roulement, report à nouveau, EBIT, capitalisation
   sur dettes, ventes) sont désormais toutes justes et à la bonne unité.

3. TRAJECTOIRE. Variations 1/2/3 ans, volatilité 4 ans, dégradations consécutives.
   Calculées pour TOUTES les sociétés du panel, saines comme défaillantes.

Usage :
    python 02_features.py                 # horizon 3 (défaut)
    python 02_features.py --horizon 2
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CLIP = 20.0          # bornes des ratios, pour neutraliser les divisions quasi-nulles


def div(a, b):
    """Division protégée : dénominateur nul ou absent -> NaN, résultat borné."""
    r = a / b.replace(0, np.nan)
    return r.replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)


def etiqueter(df, horizon):
    df = df.sort_values(["company_name", "fyear"]).reset_index(drop=True)
    df["last_fyear"] = df.groupby("company_name")["fyear"].transform("max")
    df["societe_defaillante"] = (df["status_label"] == "failed").astype(int)
    df["y"] = ((df.societe_defaillante == 1) & (df.fyear > df.last_fyear - horizon)).astype(int)
    return df


def ratios(df):
    """Ratios financiers classiques + score d'Altman."""
    eq = df.total_assets - df.total_liabilities          # capitaux propres comptables
    wc = df.current_assets - df.total_current_liabilities  # fonds de roulement
    df["equity"] = eq
    df["working_capital"] = wc

    # liquidité
    df["current_ratio"]        = div(df.current_assets, df.total_current_liabilities)
    df["quick_ratio"]          = div(df.current_assets - df.inventory, df.total_current_liabilities)
    df["working_capital_ratio"] = div(wc, df.total_assets)

    # rentabilité
    df["gross_margin"]  = div(df.gross_profit, df.net_sales)
    df["ebitda_margin"] = div(df.ebitda,       df.net_sales)
    df["ebit_margin"]   = div(df.ebit,         df.net_sales)
    df["net_margin"]    = div(df.net_income,   df.net_sales)
    df["roa"]           = div(df.net_income,   df.total_assets)
    df["roe"]           = div(df.net_income,   eq)
    df["re_to_assets"]  = div(df.retained_earnings, df.total_assets)

    # structure financière
    df["debt_to_assets"]    = div(df.total_liabilities,     df.total_assets)
    df["lt_debt_to_assets"] = div(df.total_long_term_debt,  df.total_assets)
    df["debt_to_equity"]    = div(df.total_liabilities,     eq)
    df["equity_ratio"]      = div(eq,                        df.total_assets)

    # exploitation
    df["asset_turnover"]       = div(df.net_sales,               df.total_assets)
    df["inventory_turnover"]   = div(df.cost_of_goods_sold,      df.inventory)
    df["receivables_turnover"] = div(df.net_sales,               df.total_receivables)
    df["opex_ratio"]           = div(df.total_operating_expenses, df.net_sales)
    df["cogs_ratio"]           = div(df.cost_of_goods_sold,      df.net_sales)
    df["da_ratio"]             = div(df.depreciation_amortization, df.total_assets)

    # marché
    df["market_to_book"]  = div(df.market_value, eq)
    df["market_to_sales"] = div(df.market_value, df.net_sales)

    # score d'Altman (version sociétés cotées)
    df["altman_z"] = (1.2 * div(wc, df.total_assets)
                      + 1.4 * div(df.retained_earnings, df.total_assets)
                      + 3.3 * div(df.ebit, df.total_assets)
                      + 0.6 * div(df.market_value, df.total_liabilities)
                      + 1.0 * div(df.net_sales, df.total_assets))

    # signaux d'alerte binaires
    df["flag_equity_negatif"]   = (eq < 0).astype(int)
    df["flag_fdr_negatif"]      = (wc < 0).astype(int)
    df["flag_ebitda_negatif"]   = (df.ebitda < 0).astype(int)
    df["flag_resultat_negatif"] = (df.net_income < 0).astype(int)
    df["flag_report_negatif"]   = (df.retained_earnings < 0).astype(int)
    return df


def croissances(df):
    g = df.groupby("company_name")
    for c, nom in [("net_sales", "croissance_ca"), ("total_assets", "croissance_actif"),
                   ("ebitda", "croissance_ebitda"), ("net_income", "croissance_resultat")]:
        prec = g[c].shift(1)
        df[nom] = ((df[c] - prec) / prec.abs().replace(0, np.nan)) \
                    .replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)
    return df


BASE_TRAJ = ["roa", "net_margin", "debt_to_assets", "current_ratio", "equity_ratio",
             "ebitda_margin", "asset_turnover", "working_capital_ratio",
             "re_to_assets", "altman_z"]


def trajectoire(df):
    """Variations pluriannuelles — calculées pour TOUTES les sociétés."""
    g = df.groupby("company_name")
    for c in BASE_TRAJ:
        l1, l2, l3 = g[c].shift(1), g[c].shift(2), g[c].shift(3)
        df[f"{c}_d1"] = df[c] - l1
        df[f"{c}_d2"] = df[c] - l2
        df[f"{c}_d3"] = df[c] - l3
        df[f"{c}_vol4"] = np.nanstd(np.vstack([df[c].values, l1.values, l2.values, l3.values]), axis=0)
    # dégradations consécutives du résultat sur 3 ans
    baisse = (g["net_income"].diff() < 0).astype(float)
    bg = baisse.groupby(df.company_name)
    df["baisses_resultat_3ans"] = baisse.fillna(0) + bg.shift(1).fillna(0) + bg.shift(2).fillna(0)
    df["anciennete"] = g.cumcount() + 1
    return df


def periodes(df):
    """Découpage recommandé par les auteurs du dataset."""
    df["periode"] = np.select(
        [df.fyear <= 2011, df.fyear.between(2012, 2014), df.fyear >= 2015],
        ["train", "validation", "test"], default="?")
    return df


def controles(df, horizon):
    ok = True
    print("\n" + "=" * 66)
    print("CONTRÔLES")
    print("=" * 66)

    # 1. étiquetage — trois propriétés qui doivent être vraies par construction
    defaillantes = df[df.societe_defaillante == 1]
    n_soc_def = defaillantes.company_name.nunique()

    fuite = int(df.loc[df.societe_defaillante == 0, "y"].sum())
    etat = "OK" if fuite == 0 else "ÉCHEC"
    if fuite: ok = False
    print(f"  [{etat:5s}] aucune société saine étiquetée en défaut : {fuite} anomalie(s)")

    par_soc = defaillantes.groupby("company_name")["y"].sum()
    trop = int((par_soc > horizon).sum()); sans = int((par_soc == 0).sum())
    etat = "OK" if (trop == 0 and sans == 0) else "ÉCHEC"
    if trop or sans: ok = False
    print(f"  [{etat:5s}] chaque défaillante a 1 à {horizon} exercices marqués "
          f"({trop} au-dessus, {sans} sans aucun)")

    # les exercices marqués doivent être les DERNIERS de la société
    d1 = defaillantes[defaillantes.y == 1].groupby("company_name").fyear.min()
    d0 = defaillantes[defaillantes.y == 0].groupby("company_name").fyear.max()
    comm = d1.index.intersection(d0.index)
    desordre = int((d0[comm] >= d1[comm]).sum())
    etat = "OK" if desordre == 0 else "ÉCHEC"
    if desordre: ok = False
    print(f"  [{etat:5s}] les exercices marqués sont bien les plus récents "
          f"({desordre} société(s) en désordre)")
    print(f"  [INFO ] {int(df.y.sum()):,} lignes y=1 pour {n_soc_def:,} sociétés défaillantes")
    print(f"  [INFO ] taux de défaut : {100*df.y.mean():.2f} % des lignes "
          f"(contre {100*df.societe_defaillante.mean():.2f} % avant ré-étiquetage)")

    # 2. plausibilité des ratios
    for nom, med, lo, hi in [("current_ratio", df.current_ratio.median(), 0.8, 4.0),
                             ("debt_to_assets", df.debt_to_assets.median(), 0.2, 0.9),
                             ("asset_turnover", df.asset_turnover.median(), 0.3, 2.0),
                             ("gross_margin", df.gross_margin.median(), 0.1, 0.7),
                             ("altman_z", df.altman_z.median(), 0.5, 8.0)]:
        etat = "OK" if lo <= med <= hi else "ÉCHEC"
        if etat == "ÉCHEC": ok = False
        print(f"  [{etat:5s}] médiane {nom:<22}{med:>8.2f}   (attendu {lo} à {hi})")

    # 3. trajectoire calculée partout, pas seulement sur les défaillantes
    part_saines = df.loc[df.societe_defaillante == 0, "roa_d1"].notna().mean()
    etat = "OK" if part_saines > 0.5 else "ÉCHEC"
    if etat == "ÉCHEC": ok = False
    print(f"  [{etat:5s}] trajectoire renseignée sur {100*part_saines:.0f} % des lignes de "
          f"sociétés SAINES (doit être élevé)")

    # 4. aucune valeur infinie
    num = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    n_inf = int(np.isinf(num).sum())
    etat = "OK" if n_inf == 0 else "ÉCHEC"
    if etat == "ÉCHEC": ok = False
    print(f"  [{etat:5s}] valeurs infinies : {n_inf}")

    # 5. découpage
    print("  [INFO ] " + " | ".join(f"{k} {v:,}" for k, v in df.periode.value_counts().items()))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args()

    src = SCRIPT_DIR / "v8_base.csv"
    if not src.exists():
        sys.exit("v8_base.csv introuvable — lancez d'abord 01_build_dataset.py")
    df = pd.read_csv(src)
    print(f"Entrée : {src.name}  ({df.shape[0]:,} lignes)")

    df = etiqueter(df, a.horizon)
    df = ratios(df)
    df = croissances(df)
    df = trajectoire(df)
    df = periodes(df)

    if not controles(df, a.horizon):
        sys.exit("\nAU MOINS UN CONTRÔLE A ÉCHOUÉ — fichier non écrit.")

    out = SCRIPT_DIR / "v8_features.csv"
    df.to_csv(out, index=False)
    n_var = df.select_dtypes(include=[np.number]).shape[1]
    print("\n" + "=" * 66)
    print(f"ÉCRIT : {out.name}  ({df.shape[0]:,} lignes × {df.shape[1]} colonnes, "
          f"{n_var} numériques)")
    print("=" * 66)


if __name__ == "__main__":
    main()
