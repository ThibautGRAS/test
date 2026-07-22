#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation étagée v5 — mode JOURNÉES (sans marché), règle de production.
Backtest walk-forward mensuel, éval 2223->2526. Métriques : log-loss 1N2 du
blend base régularisé, score exact hit@1 (règle pick_score_pb), hit@3, nuls."""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train as T

def collect(m, ycls, xi, w_sot, l2):
    """Backtest : renvoie la liste des prédictions (blend base) par saison."""
    preds = []
    test_seasons = [s for s in sorted(m.Season.unique()) if s >= "2122"]
    for s in test_seasons:
        sm = m[m.Season == s]
        for _, grp in sm.groupby(sm.DateP.dt.to_period("M")):
            t0 = grp.DateP.min(); tr = m[m.DateP < t0]
            teams = sorted(set(tr.HomeTeam) | set(tr.AwayTeam))
            fg = T.dc_fit(tr, t0, xi, teams=teams, team_home=False)
            if w_sot > 0:
                ss = tr[tr["HST"].notna() & tr["AST"].notna()].copy()
                conv = (ss.FTHG.sum()+ss.FTAG.sum())/max(ss.HST.sum()+ss.AST.sum(), 1)
                ss["PgH"] = np.rint(ss.HST*conv).astype(float)
                ss["PgA"] = np.rint(ss.AST*conv).astype(float)
                fs = T.dc_fit(ss, t0, xi, teams=teams, cols=("PgH","PgA"), team_home=False)
                fit = {"teams": teams, "rho": fg["rho"],
                       "attack": (1-w_sot)*fg["attack"]+w_sot*fs["attack"],
                       "defense": (1-w_sot)*fg["defense"]+w_sot*fs["defense"],
                       "gamma": (1-w_sot)*fg["gamma"]+w_sot*fs["gamma"],
                       "delta_home": (1-w_sot)*fg["delta_home"]+w_sot*fs["delta_home"]}
            else:
                fit = fg
            for ridx, r in grp.iterrows():
                p = T.dc_predict(fit, r.HomeTeam, r.AwayTeam)
                if p is None: continue
                preds.append({"idx": ridx, "season": s, "dc": p,
                              "fb": T.feat_base(r, p), "y": int(ycls[ridx]),
                              "gh": int(r.FTHG), "ga": int(r.FTAG)})
    # blend base walk-forward (L2 paramétrable)
    ev_seasons = test_seasons[1:]
    for s in ev_seasons:
        hist = [q for q in preds if q["season"] < s]
        cur = [q for q in preds if q["season"] == s]
        W = T.blend_fit(np.array([q["fb"] for q in hist]),
                        np.array([q["y"] for q in hist]), l2=l2)
        for q in cur:
            q["pb"] = list(T.blend_predict(W, q["fb"]))
    return [q for q in preds if q["season"] in ev_seasons]

def metrics(ev, margin):
    T.MARGE_NUL = margin
    ll = np.mean([-math.log(max(q["pb"][q["y"]], 1e-9)) for q in ev])
    h1 = h3 = nuls = 0
    for q in ev:
        Mx = T.production_matrix(q["dc"]["lam"], q["dc"]["mu"], q["dc"]["rho"], q["pb"])
        bi, bj = T._best_in_class(Mx, q["pb"])
        if bi == bj: nuls += 1
        if (bi, bj) == (q["gh"], q["ga"]): h1 += 1
        flat = sorted(((Mx[i, j], i, j) for i in range(Mx.shape[0]) for j in range(Mx.shape[1])),
                      reverse=True)
        if any((i, j) == (q["gh"], q["ga"]) for _, i, j in flat[:3]): h3 += 1
    n = len(ev)
    return {"ll": ll, "h1": 100*h1/n, "h3": 100*h3/n, "nuls": 100*nuls/n}

if __name__ == "__main__":
    print("Chargement…")
    m = T.load_matches(str(ROOT / "data"))
    m, _ = T.add_context(m)
    ycls = np.where(m.FTHG > m.FTAG, 0, np.where(m.FTHG == m.FTAG, 1, 2))
    # nuls réels
    ev_mask = m.Season.isin(["2223","2324","2425","2526"])
    nuls_reels = 100*(m[ev_mask].FTHG == m[ev_mask].FTAG).mean()
    print(f"nuls réels: {nuls_reels:.1f}%\n")

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("1", "all"):
        print("=== ÉTAGE 1 : poids tirs cadrés (L2=5, marge=0.16) ===")
        for w in [0.0, 0.15, 0.30]:
            ev = collect(m, ycls, 0.001, w, 5.0)
            r = metrics(ev, 0.16)
            print(f"  w_sot={w:<4}  ll={r['ll']:.4f}  hit@1={r['h1']:.2f}%  hit@3={r['h3']:.2f}%  nuls={r['nuls']:.1f}%")
    if stage in ("2", "all"):
        print("=== ÉTAGE 2 : régularisation L2 du blend base (w_sot=0.15) ===")
        for l2 in [1.0, 3.0, 5.0, 8.0]:
            ev = collect(m, ycls, 0.001, 0.15, l2)
            r = metrics(ev, 0.16)
            print(f"  L2={l2:<4}  ll={r['ll']:.4f}  hit@1={r['h1']:.2f}%")
    if stage in ("3", "all"):
        print("=== ÉTAGE 3 : marge de nul (w_sot=0.15, L2=5) ===")
        ev = collect(m, ycls, 0.001, 0.15, 5.0)
        for mg in [0.10, 0.14, 0.16, 0.18, 0.22]:
            r = metrics(ev, mg)
            print(f"  marge={mg:<5}  hit@1={r['h1']:.2f}%  nuls={r['nuls']:.1f}%  (réel {nuls_reels:.1f}%)")
