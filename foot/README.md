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
