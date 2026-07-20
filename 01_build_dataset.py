# -*- coding: utf-8 -*-
"""
V8 — Étape 1/4 : construction du jeu de données de base, avec les BONS intitulés.

Le fichier public est distribué anonymisé (X1..X18). La correspondance réelle a été
établie le 20/07/2026 par deux voies indépendantes :
  a) les fichiers annexes du papier (financial_train/validation/test.csv) portent les
     noms en clair, dans cet ordre ;
  b) recherche à l'aveugle des identités comptables sur les 4 896 triplets possibles :
     les 10 meilleures correspondances sont toutes de vraies identités sous ce mapping
     (EBITDA - EBIT = D&A à 58,6 %, CA - charges expl. = EBITDA à 57,7 %, etc.).

Sortie : v8_base.csv
Contrôles bloquants en fin de script — le fichier n'est écrit que s'ils passent tous.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- mapping réel
MAPPING = {
    "X1":  "current_assets",
    "X2":  "total_assets",
    "X3":  "cost_of_goods_sold",
    "X4":  "total_long_term_debt",
    "X5":  "depreciation_amortization",
    "X6":  "ebit",
    "X7":  "ebitda",
    "X8":  "gross_profit",
    "X9":  "inventory",
    "X10": "total_current_liabilities",
    "X11": "net_income",
    "X12": "retained_earnings",
    "X13": "total_receivables",
    "X14": "total_revenue",
    "X15": "market_value",
    "X16": "total_liabilities",
    "X17": "net_sales",
    "X18": "total_operating_expenses",
}
# Postes de BILAN : un négatif est structurellement impossible -> tolérance zéro.
STOCKS = ["current_assets", "total_assets", "total_long_term_debt", "inventory",
          "total_current_liabilities", "total_receivables", "market_value", "total_liabilities"]
# Postes de FLUX : un négatif est rare mais réel (reprises, sociétés d'investissement)
# -> toléré jusqu'à 0,1 % des lignes, toujours décompté dans le rapport.
FLUX = ["cost_of_goods_sold", "depreciation_amortization", "total_revenue",
        "net_sales", "total_operating_expenses"]
SEUIL_FLUX = 0.1  # %


def trouver_source() -> Path:
    """Localise le CSV brut anonymisé (colonnes X1..X18)."""
    noms = ["american_bankruptcy_dataset.csv"]
    bases = [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent,
             SCRIPT_DIR.parent.parent / "Machine learning exercise"]
    for b in bases:
        for n in noms:
            for p in list(b.glob(n)) + list(b.rglob(n)):
                try:
                    if "X1" in pd.read_csv(p, nrows=1).columns:
                        return p
                except Exception:
                    continue
    sys.exit("Introuvable : american_bankruptcy_dataset.csv (version brute, colonnes X1..X18).\n"
             "Placez-le dans le dossier V8 ou un dossier parent.")


def noms_secteurs(df: pd.DataFrame) -> pd.DataFrame:
    """Rattache les libellés Division / MajorGroup depuis l'ancien fichier enrichi si présent."""
    for p in (SCRIPT_DIR.parent).rglob("american_bankruptcy_dataset_enriched.csv"):
        try:
            old = pd.read_csv(p, usecols=["division", "majorgroup", "division_name", "majorgroup_name"])
            d = old.drop_duplicates("division").set_index("division")["division_name"]
            m = old.drop_duplicates("majorgroup").set_index("majorgroup")["majorgroup_name"]
            df["division_name"] = df["division"].map(d)
            df["majorgroup_name"] = df["majorgroup"].map(m)
            print(f"  libellés secteurs repris de : {p.name}")
            return df
        except Exception:
            continue
    print("  (libellés secteurs non trouvés — colonnes 'division'/'majorgroup' conservées seules)")
    return df


def reparer_milliers(df: pd.DataFrame) -> pd.DataFrame:
    """Répare les valeurs ayant perdu leur séparateur de milliers.

    Environ 22 % des cellules du fichier public sont stockées « 701.854 » au lieu de
    « 701854 » : le séparateur de milliers a été interprété comme séparateur décimal
    lors de la préparation du fichier. On les repère sans ambiguïté — une valeur non
    entière dont le produit par 1 000 tombe sur un entier — et on les multiplie par mille.

    Validation : cette réparation fait passer les identités comptables de ~57 % à ~98 %.

    market_value en est EXCLUE : cette colonne est nativement décimale (63 000 valeurs
    à 4 décimales) et exprimée en millions. Lui appliquer la règle gonflait 6,3 % des
    valeurs d'un facteur mille et portait le market-to-book à 26 au 3e quartile, ce qui
    n'a pas de sens. Elle est traitée séparément par convertir_market_value().
    """
    cols = [c for c in MAPPING.values() if c != "market_value"]
    print("\nRéparation du séparateur de milliers :")
    total = 0
    for c in cols:
        v = df[c]
        susp = (v.notna() & (v.abs() > 0)
                & ((v - v.round()).abs() > 1e-9)
                & (((v * 1000) - (v * 1000).round()).abs() < 1e-3))
        n = int(susp.sum())
        if n:
            df.loc[susp, c] = v[susp] * 1000
            total += n
    print(f"    {total:,} cellule(s) corrigée(s) sur {len(df) * len(cols):,} "
          f"({100 * total / (len(df) * len(cols)):.1f} %)")
    return df


def convertir_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne market_value sur l'unité des postes comptables.

    La capitalisation est exprimée en MILLIONS, les postes comptables en MILLIERS.
    Sans conversion, le market-to-book médian ressort à 0,003 au lieu de ~2,6, et le
    score d'Altman — qui rapporte la capitalisation aux dettes — serait faux.
    """
    df["market_value"] = df["market_value"] * 1000
    print("\nConversion d'unité : market_value (millions) -> milliers, x1000")
    return df


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise les valeurs structurellement impossibles sur les postes de bilan.

    Un actif, un stock ou une dette ne peut pas être négatif : ces cellules sont des
    erreurs de la source. Elles passent à NaN — la ligne entière est conservée, car
    ses autres colonnes restent exploitables. Tout est décompté, rien n'est silencieux.
    """
    print("\nNettoyage des valeurs impossibles (postes de bilan) :")
    total = 0
    for c in STOCKS:
        if c not in df:
            continue
        m = df[c] < 0
        n = int(m.sum())
        if n:
            df.loc[m, c] = np.nan
            total += n
            print(f"    {c:<28} {n:>3} valeur(s) négative(s) -> NaN")
    print(f"    total neutralisé : {total} cellule(s) sur {len(df) * len(STOCKS):,}")
    return df


def controles(df: pd.DataFrame) -> bool:
    """Contrôles bloquants. Renvoie True si tout passe."""
    ok = True
    print("\n" + "=" * 62)
    print("CONTRÔLES")
    print("=" * 62)

    def verif(a, b, tol=0.01):
        m = a.notna() & b.notna() & (b.abs() > 1)
        return 100 * (((a[m] - b[m]).abs() / b[m].abs()) < tol).mean()

    id1 = verif(df.ebitda - df.depreciation_amortization, df.ebit)
    id2 = verif(df.net_sales - df.cost_of_goods_sold, df.gross_profit)
    id3 = verif(df.net_sales - df.total_operating_expenses, df.ebitda)
    SEUIL_ID = 95  # après réparation des milliers, les identités doivent tenir à ~98 %
    for lbl, v in [("EBITDA - D&A = EBIT", id1), ("CA - coût des ventes = marge brute", id2),
                   ("CA - charges d'exploitation = EBITDA", id3)]:
        etat = "OK" if v > SEUIL_ID else "ÉCHEC"
        if v <= SEUIL_ID: ok = False
        print(f"  [{etat:5s}] {lbl:<40} {v:5.1f} %   (seuil {SEUIL_ID} %)")

    v = 100 * (df.total_assets >= df.current_assets).mean()
    etat = "OK" if v > 98 else "ÉCHEC"
    if v <= 98: ok = False
    print(f"  [{etat:5s}] {'actif total >= actif courant':<40} {v:5.1f} %   (seuil 98 %)")

    # postes de bilan : tolérance zéro
    faute = {c: int((df[c] < 0).sum()) for c in STOCKS if c in df and (df[c] < 0).any()}
    if faute:
        ok = False
        print(f"  [ÉCHEC] négatifs sur postes de bilan : {faute}")
    else:
        print(f"  [OK   ] aucun négatif sur les {len(STOCKS)} postes de bilan")

    # postes de flux : rares négatifs admis, mais comptés
    for c in FLUX:
        if c not in df: continue
        n = int((df[c] < 0).sum()); pct = 100 * n / len(df)
        if n == 0: continue
        if pct > SEUIL_FLUX:
            ok = False
            print(f"  [ÉCHEC] {c} : {n} négatifs ({pct:.3f} % > {SEUIL_FLUX} %)")
        else:
            print(f"  [OK   ] {c} : {n} négatifs ({pct:.3f} %) — sous le seuil, conservés")

    n_lig, n_soc = len(df), df.company_name.nunique()
    att_lig, att_soc = 78682, 8971
    etat = "OK" if (n_lig == att_lig and n_soc == att_soc) else "ÉCHEC"
    if etat == "ÉCHEC": ok = False
    print(f"  [{etat:5s}] {n_lig:,} lignes / {n_soc:,} sociétés (attendu {att_lig:,} / {att_soc:,})")

    a, b = int(df.fyear.min()), int(df.fyear.max())
    etat = "OK" if (a, b) == (1999, 2018) else "ÉCHEC"
    if etat == "ÉCHEC": ok = False
    print(f"  [{etat:5s}] période {a}-{b} (attendu 1999-2018)")

    # cohérence d'unité entre capitalisation et bilan
    eq = df.total_assets - df.total_liabilities
    mb = (df.market_value / eq.where(eq > 0)).replace([np.inf, -np.inf], np.nan).median()
    etat = "OK" if 0.5 < mb < 6 else "ÉCHEC"
    if not (0.5 < mb < 6): ok = False
    print(f"  [{etat:5s}] {'market-to-book médian':<40} {mb:5.2f}     (attendu 0,5 à 6)")

    mixtes = (df.groupby("company_name").status_label.nunique() > 1).sum()
    print(f"  [INFO ] sociétés à statut mixte selon l'année : {mixtes} "
          f"(0 attendu — le ré-étiquetage se fait à l'étape 2)")
    return ok


def main():
    src = trouver_source()
    print(f"Source : {src}")
    df = pd.read_csv(src)

    df = df.rename(columns=MAPPING)
    df = df.rename(columns={"Division": "division", "MajorGroup": "majorgroup"})
    df["fyear"] = df["fyear"].astype(int)
    for c in MAPPING.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = noms_secteurs(df)
    ordre = ["company_name", "fyear", "status_label"] + list(MAPPING.values()) + \
            [c for c in ["division", "majorgroup", "division_name", "majorgroup_name"] if c in df]
    df = df[ordre]
    df = reparer_milliers(df)
    df = convertir_market_value(df)
    df = nettoyer(df)

    if not controles(df):
        sys.exit("\nAU MOINS UN CONTRÔLE A ÉCHOUÉ — fichier non écrit.")

    out = SCRIPT_DIR / "v8_base.csv"
    df.to_csv(out, index=False)
    print("\n" + "=" * 62)
    print(f"ÉCRIT : {out}  ({df.shape[0]:,} lignes × {df.shape[1]} colonnes)")
    print("=" * 62)


if __name__ == "__main__":
    main()
