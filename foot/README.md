# Génie du Foot — Prédiction de scores Ligue 1

App : https://thibautgras.github.io/test/foot/

- **Modèle** : Dixon-Coles (Poisson bivarié, correction faibles scores, pondération
  temporelle exponentielle, demi-vie ~2 saisons) + Elo interne + couche de calibration
  1N2 (régression logistique multinomiale) blendant modèle, cotes de marché de-viggées
  et contexte (repos, forme 5 matchs, promus).
- **Données** : football-data.co.uk, 10 saisons de Ligue 1 (2016-17 → 2025-26),
  3 476 matchs, cotes pré-match incluses.
- **Backtest walk-forward** (réajustement mensuel, 911 matchs 2023-26) :
  log-loss 1N2 — Dixon-Coles 1,001 · blend 0,996 · marché 0,981.
  Score exact trouvé : 11,9 % (état de l'art ≈ 10-12 %).
- **Réentraînement** : GitHub Action hebdomadaire (`foot-train.yml`) qui télécharge
  les données fraîches et committe `model.json`.
- `train.py --local data/` pour un entraînement hors-ligne.

L'app fait toute l'inférence côté client (matrice de scores, 1N2 calibré, over/under,
BTTS) à partir du seul `model.json`.

## v2 — Saison 2026-27
- **Vue par journée** : les 34 journées du calendrier officiel (306 matchs, source
  openfootball) avec score prédit (case la plus probable dans l'issue 1N2 dominante),
  sa probabilité, et le 1N2 calibré. Repos entre journées calculé depuis le calendrier.
- **Signal actu** (`news.json`) : ajustements modérés attaque/défense par club issus de
  la presse mercato (départs Greenwood/Aubameyang à l'OM, investissements Paris FC,
  budget contraint du Mans…), sourcés et affichés dans l'onglet « Modèle & actu ».
  Mis à jour à la demande via Claude — le réentraînement hebdo ne l'écrase pas.
- Le Mans (sans historique L1 récent) reçoit un prior de promu ; `train.py` le
  régénère automatiquement tant que l'équipe est au calendrier.

## v2.2 — Classement projeté & contexte structurel
- **Onglet Classement** : 5 000 saisons simulées par Monte-Carlo (tirage des scores dans
  les matrices calibrées, départage points > diff > buts) → xPts, proba de titre,
  Top 4 (C1), relégation. PRNG xorshift déterministe.
- **Contexte structurel** dans `news.json` : Coupe d'Europe 2026-27 et ancienneté du banc
  par club, injectés dans les commentaires (« banc récent », « enchaîne avec la C1 »).

## v2.3 — Comparateur de clubs & indice de chaos
- **Onglet Clubs** : graphe à barres des 18 clubs, filtrable sur 6 critères —
  Performance, Attaque, Défense (ratings DC + actu), Forme, Régularité
  (constance des points 2025-26, calculée par `train.py`), Impact actu.
- **Indice de chaos** par match : 55 % entropie du 1N2 calibré + 45 % irrégularité
  moyenne des deux clubs. Au-delà du seuil (~30 % des matchs), la carte affiche
  un **scénario surprise** : le score le plus probable dans l'issue inverse
  (l'upset), avec sa probabilité réelle. Toujours visible dans le simulateur.

## v2.5 — Onglet Validation
- Backtest décliné **par saison** (2023-24, 2024-25, 2025-26) : log-loss modèle vs marché,
  précision 1N2, scores exacts.
- **Saisons simulées à l'aveugle** : projection avec le seul modèle d'avant-saison
  (2 000 sims sur le calendrier réel) vs classement final — MAE points, Spearman,
  plus gros ratés (Brest 2023-24, Rennes ×2, Nice 2025-26…).
- **Chaos validé rétrospectivement** : log-loss 0,999 vs 0,943 et favori battu
  24,4 % vs 21,5 % entre tiers haut/bas d'irrégularité.
- Synthèse tendances & feuille de route d'amélioration dans l'app.

## v3.0 — Modèle amélioré + distribution des totaux
- **Fusion tirs cadrés** (`dc_fit_fused`, proxy xG, poids 30 %) et **avantage domicile
  par équipe** avec rétrécissement L2 : blend 0,9964 → 0,9953 en log-loss, corrélation
  des rangs améliorée sur les 3 saisons (ρ 0,66/0,70/0,75), léger recul du score exact.
- **Affichage des nuls corrigé** : marge de tolérance 0,16 en faveur du nul dans le
  choix de la classe affichée, calibrée sur le taux historique (27,5 % de nuls affichés
  vs 25,2 % réels ; auparavant 0,3 %).
- **Validation** : distribution des totaux de buts modèle vs réel (backtest 911 matchs),
  filtrable par équipe — 3 buts : 21,6 % prédit vs 21,5 % observé.
