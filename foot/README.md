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

## v3.1 — Optimisation étagée (`optimize.py`)
- Recherche par étages sur le backtest 2023-26 : poids tirs cadrés (0,15 optimal,
  0,30 dégrade le hit@1), avantage domicile par équipe (retiré — aucun gain),
  décroissance xi=0,001 confirmée, règle d'affichage à marge de nul 0,16 optimale
  en hit@1 (12,07 %) ET en réalisme (29 % de nuls affichés vs 25 % réels).
- Métrique « score exact » du pipeline alignée sur la règle de production.

## v3.2 — Validation étendue à 2022-23
- Backtest 4 saisons (1 281 matchs éval) : 2022-23 ajoutée APRÈS l'optimisation des
  hyperparamètres → test quasi hors-échantillon réussi (blend 0,983 vs marché 0,975).
- Champion prédit 4/4, chaos confirmé sur 854 matchs par tiers.

## v4.0 — Six chantiers d'amélioration
1. **Priors promus informés par la L2** (`load_l2_stats`, `promus_regression`) :
   régression stats de montée → ratings L1 (R² 0,37 attaque), anti-fuite dans les
   simulations. ρ 2022-23 : 0,64 → 0,73. Troyes/Le Mans priorisés sur leur vraie L2.
2. **Signal mercato** : données Transfermarkt inaccessibles après 2021 — calibration
   reportée sur les résidus live (voir 6).
3. **Recalage over/under** : testé sur 1 281 matchs — améliore les probabilités
   (ll 1N2 −0,002, totaux −0,008) mais coûte 1,2 pt de hit@1. Intégré uniquement au
   simulateur (saisie des cotes +/−2,5).
4. **Chaos appris** : logistique favori_battu ~ entropie + irrégularité, réajustée à
   chaque réentraînement. 29,9 % vs 18,4 % de favoris battus (tiers haut/bas).
   L'app affiche directement P(upset).
5. **Atténuation inter-saisons** : testée (delta 0,9/0,8) et rejetée — aucun gain.
6. **Journal de prédictions figées** (`journal.py`) : l'Action (mardi + vendredi) fige
   les prédictions des 9 prochains jours et évalue les précédentes — hit@1 et log-loss
   2026-27 traçables et inviolables dès août.

## v4.1 — Signal mercato calibré sur données
- Historique Transfermarkt récupéré (ewenme/transfers, 17 581 transferts L1 1992-2022).
- Calibration sur 108 club-saisons (étés 2017-2022) : la dérive du rating attaque en
  cours de saison est prédite par le **solde net estival** — +0,0218 par écart-type
  (t=2,33, p≈0,02). Défense et dépenses brutes : non significatifs.
- Les amplitudes du signal actu (±0,03-0,08) sont dans la fourchette calibrée.
- `mercato.py` : calcule les z-scores et applique le coefficient depuis un CSV de
  transferts (clôture du 1er septembre, ou export récent via la GitHub Action).

## v5.0 — Audit externe intégré
- **Deux calibrateurs séparés** (audit §2) : `blend` (sans cotes, mode journées, L2=5 pour
  éviter le surapprentissage → 0,999 vs 1,003) et `blend_marche` (avec cotes, mode simulateur).
  Validation affiche 4 colonnes : journées / +cotes / marché / exact.
- **Parité Python↔JS** (audit §3, §14) : `pick_score_pb` + `production_matrix` reproduisent
  exactement la règle JS ; `displayedClass` unifie score affiché et commentaire.
- **Début de saison sécurisé** (audit §6) : régularité anti-division-zéro (span→0 ⇒ 0,5).
- **Départage classement** (audit §8) : points > diff > buts, réel et simulé.
- **xG finaux** (audit §12) : espérances calculées sur la matrice recalée, pas lam/mu bruts.
- **Libellés** (audit §9, §13) : « scénario central retenu » + score modal global affiché ;
  chaos = « relation observée ».
- **Empreinte mercato/contexte** : onglet Validation montre les résidus (réel − attendu)
  par club et saison — base objective pour calibrer le signal actu sur données réelles.
- Version du modèle dans meta (audit §20.5).
