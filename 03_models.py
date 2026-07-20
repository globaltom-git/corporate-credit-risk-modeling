# -*- coding: utf-8 -*-
"""
V8 — Étape 3/4 : comparaison des familles de modèles, réglage, calibration.

Entrée : v8_features.csv        Sorties : v8_resultats.json, v8_model_lgbm.txt,
                                          v8_predictions_test.csv

Protocole des auteurs du dataset :
    entraînement 1999-2011 · validation 2012-2014 · test 2015-2018
Le réglage des hyperparamètres et la calibration se font sur la VALIDATION.
Le test n'est utilisé qu'une fois, pour le résultat final.

Familles comparées :
  1. Clustering + règles (K-Means : score = taux de défaut observé du groupe)
  2. Régression logistique (référence linéaire)
  3. Random Forest
  4. LightGBM
  5. Réseau de neurones (perceptron multicouche)

Options :
  --smote        rééquilibrage SMOTE de l'entraînement (pour MESURER son effet)
  --no-fyear     exclure l'année fiscale des variables
"""
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
EXCLURE = ["company_name", "status_label", "y", "societe_defaillante", "last_fyear",
           "division_name", "majorgroup_name", "periode"]


def auc_mw(y, s):
    """AUC par Mann-Whitney — sans dépendance externe."""
    y = np.asarray(y); s = np.asarray(s, float)
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if (n1 == 0 or n0 == 0) else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def se_auc(a, n1, n0):
    """Erreur-type de l'AUC (Hanley-McNeil)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))


def metriques(y, p, part=0.20):
    """AUC + rappel et précision sur les 'part' % de scores les plus élevés + Brier."""
    y = np.asarray(y); p = np.asarray(p, float)
    seuil = np.quantile(p, 1 - part)
    alerte = p >= seuil
    return {
        "auc": round(float(auc_mw(y, p)), 4),
        "rappel_top20": round(float((alerte & (y == 1)).sum() / max(y.sum(), 1)), 3),
        "precision_top20": round(float((alerte & (y == 1)).sum() / max(alerte.sum(), 1)), 3),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
    }


def charger(no_fyear):
    src = SCRIPT_DIR / "v8_features.csv"
    if not src.exists():
        sys.exit("v8_features.csv introuvable — lancez d'abord 02_features.py")
    df = pd.read_csv(src)
    feat = [c for c in df.columns
            if c not in EXCLURE and pd.api.types.is_numeric_dtype(df[c])]
    if no_fyear and "fyear" in feat:
        feat.remove("fyear")
    parts = {p: df[df.periode == p].reset_index(drop=True) for p in ("train", "validation", "test")}
    return df, parts, feat


def preparer(parts, feat):
    """Imputation (médiane apprise sur le train seul) + standardisation."""
    tr = parts["train"]
    med = tr[feat].median()
    ecart = tr[feat].std().replace(0, 1).fillna(1)
    X, Xs, Y = {}, {}, {}
    for k, g in parts.items():
        x = g[feat].replace([np.inf, -np.inf], np.nan).fillna(med)
        X[k] = x.values
        Xs[k] = ((x - med) / ecart).values          # version standardisée
        Y[k] = g["y"].values
    return X, Xs, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--no-fyear", action="store_true")
    a = ap.parse_args()

    df, parts, feat = charger(a.no_fyear)
    X, Xs, Y = preparer(parts, feat)
    print(f"Variables : {len(feat)}   (année fiscale {'exclue' if a.no_fyear else 'incluse'})")
    for k in ("train", "validation", "test"):
        print(f"  {k:<11}{len(Y[k]):>8,} lignes   {int(Y[k].sum()):>5,} défauts "
              f"({100*Y[k].mean():.2f} %)")

    Xtr, Ytr = X["train"], Y["train"]
    Xtr_s = Xs["train"]
    if a.smote:
        try:
            from imblearn.over_sampling import SMOTE
            y0 = Y["train"]
            Xtr, Ytr = SMOTE(random_state=42).fit_resample(Xtr, y0)      # version brute
            Xtr_s, _ = SMOTE(random_state=42).fit_resample(Xtr_s, y0)    # version standardisée
            print(f"\nSMOTE actif : entraînement porté à {len(Ytr):,} lignes "
                  f"({100*Ytr.mean():.1f} % de défauts)")
        except ImportError:
            sys.exit("imblearn manquant :  pip install imbalanced-learn")

    res, modeles = {}, {}
    print("\n" + "=" * 74)
    print("ENTRAÎNEMENT")
    print("=" * 74)

    # ---------------------------------------------------------- 1. clustering
    from sklearn.cluster import KMeans
    print("  1/5  Clustering + règles (K-Means)...")
    best = (None, -1)
    for k in (6, 10, 16):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr_s)
        taux = pd.Series(Ytr).groupby(km.labels_).mean()          # règle : risque du groupe
        s_val = pd.Series(km.predict(Xs["validation"])).map(taux).fillna(Ytr.mean()).values
        u = auc_mw(Y["validation"], s_val)
        if u > best[1]: best = ((km, taux), u)
    km, taux = best[0]
    s_test = pd.Series(km.predict(Xs["test"])).map(taux).fillna(Ytr.mean()).values
    res["Clustering + règles"] = metriques(Y["test"], s_test)
    print(f"       k={km.n_clusters} retenu sur validation (AUC {best[1]:.3f})")

    # ------------------------------------------------- 2. régression logistique
    from sklearn.linear_model import LogisticRegression
    print("  2/5  Régression logistique...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, Ytr)
    res["Régression logistique"] = metriques(Y["test"], lr.predict_proba(Xs["test"])[:, 1])

    # ------------------------------------------------------- 3. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    print("  3/5  Random Forest (réglage sur validation)...")
    best = (None, -1)
    for prof in (8, 14, None):
        rf = RandomForestClassifier(n_estimators=400, max_depth=prof, min_samples_leaf=5,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=42).fit(Xtr, Ytr)
        u = auc_mw(Y["validation"], rf.predict_proba(X["validation"])[:, 1])
        if u > best[1]: best = (rf, u)
    rf = best[0]; modeles["rf"] = rf
    res["Random Forest"] = metriques(Y["test"], rf.predict_proba(X["test"])[:, 1])
    print(f"       profondeur {rf.max_depth} retenue (AUC validation {best[1]:.3f})")

    # ------------------------------------------------------------ 4. LightGBM
    import lightgbm as lgb
    print("  4/5  LightGBM (réglage sur validation)...")
    spw = (Ytr == 0).sum() / max((Ytr == 1).sum(), 1)
    dtr = lgb.Dataset(Xtr, label=Ytr, feature_name=feat)
    best = (None, -1, None)
    for lr_ in (0.03, 0.06):
        for leaves in (31, 63):
            m = lgb.train({"objective": "binary", "learning_rate": lr_, "num_leaves": leaves,
                           "min_data_in_leaf": 40, "feature_fraction": 0.8,
                           "bagging_fraction": 0.8, "bagging_freq": 1,
                           "scale_pos_weight": spw, "verbose": -1, "seed": 42},
                          dtr, num_boost_round=400)
            u = auc_mw(Y["validation"], m.predict(X["validation"]))
            if u > best[1]: best = (m, u, (lr_, leaves))
    gbm = best[0]; modeles["lgbm"] = gbm
    p_test_lgbm = gbm.predict(X["test"])
    res["LightGBM"] = metriques(Y["test"], p_test_lgbm)
    print(f"       lr={best[2][0]} feuilles={best[2][1]} (AUC validation {best[1]:.3f})")

    # -------------------------------------------------- 5. réseau de neurones
    from sklearn.neural_network import MLPClassifier
    print("  5/5  Réseau de neurones...")
    mlp = MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=260, early_stopping=True,
                        n_iter_no_change=12, random_state=42).fit(Xtr_s, Ytr)
    res["Réseau de neurones"] = metriques(Y["test"], mlp.predict_proba(Xs["test"])[:, 1])

    # ------------------------------------------------------------ calibration
    from sklearn.isotonic import IsotonicRegression
    print("\nCalibration des probabilités (isotonique, ajustée sur la validation)...")
    iso = IsotonicRegression(out_of_bounds="clip").fit(gbm.predict(X["validation"]), Y["validation"])
    p_cal = iso.predict(p_test_lgbm)
    res["LightGBM calibré"] = metriques(Y["test"], p_cal)

    # ---------------------------------------------------------------- rapport
    print("\n" + "=" * 74)
    print(f"{'RÉSULTATS — test 2015-2018':<34}{'AUC':>8}{'rappel':>9}{'précis.':>9}{'Brier':>9}")
    print("=" * 74)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {k:<32}{v['auc']:>8.3f}{v['rappel_top20']:>9.2f}"
              f"{v['precision_top20']:>9.2f}{v['brier']:>9.4f}")
    n1 = int(Y["test"].sum()); n0 = len(Y["test"]) - n1
    a_best = max(v["auc"] for v in res.values())
    print(f"\n  marge d'erreur sur l'AUC (95 %) : ±{1.96*se_auc(a_best,n1,n0):.3f}  "
          f"({n1} défauts en test)")

    # --------------------------------------------------------------- contrôles
    print("\n" + "=" * 74); print("CONTRÔLES"); print("=" * 74)
    ok = True
    an = {k: (int(parts[k].fyear.min()), int(parts[k].fyear.max())) for k in parts}
    disjoint = an["train"][1] < an["validation"][0] < an["validation"][1] < an["test"][0]
    print(f"  [{'OK   ' if disjoint else 'ÉCHEC'}] périodes disjointes et ordonnées : "
          f"{an['train']} < {an['validation']} < {an['test']}")
    ok &= disjoint
    faibles = [k for k, v in res.items() if v["auc"] < 0.5]
    print(f"  [{'OK   ' if not faibles else 'ÉCHEC'}] tous les modèles font mieux que le hasard"
          + (f" — sauf {faibles}" if faibles else ""))
    ok &= not faibles
    # la calibration est ajustée sur la validation (2,57 % de défauts) et appliquée au
    # test (2,08 %) : un écart de l'ordre du demi-point est normal et attendu.
    moy, obs = float(p_cal.mean()), float(Y["test"].mean())
    bon = abs(moy - obs) < 0.015
    print(f"  [{'OK   ' if bon else 'ÉCHEC'}] calibration : probabilité moyenne prédite "
          f"{100*moy:.2f} % vs taux observé {100*obs:.2f} %")
    ok &= bon
    if not ok:
        sys.exit("\nAU MOINS UN CONTRÔLE A ÉCHOUÉ.")

    # ---------------------------------------------------------------- sorties
    gbm.save_model(str(SCRIPT_DIR / "v8_model_lgbm.txt"))
    pd.DataFrame({"company_name": parts["test"].company_name, "fyear": parts["test"].fyear,
                  "y": Y["test"], "score": p_test_lgbm, "proba_calibree": p_cal}
                 ).to_csv(SCRIPT_DIR / "v8_predictions_test.csv", index=False)
    json.dump({"protocole": "train 1999-2011 / validation 2012-2014 / test 2015-2018",
               "variables": len(feat), "smote": a.smote, "fyear_exclue": a.no_fyear,
               "defauts_test": n1, "resultats": res},
              open(SCRIPT_DIR / "v8_resultats.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nÉcrit : v8_resultats.json · v8_model_lgbm.txt · v8_predictions_test.csv")


if __name__ == "__main__":
    main()
