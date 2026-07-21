#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recherche d'hyperparamètres par étages pour Génie du Foot.
Backtest walk-forward (refit mensuel) sur 2023-24 -> 2025-26.
Métriques : log-loss 1N2 (DC brut), score exact hit@1 et hit@3."""
import json, math, sys
import numpy as np
sys.path.insert(0, '/home/claude/foot')
import train as T

def backtest(m, ycls, xi, w_sot, pen_home, team_home=True, seasons=("2324","2425","2526")):
    out = {s: {"ll": [], "h1": 0, "h3": 0, "n": 0} for s in seasons}
    for s in seasons:
        sm = m[m.Season == s]
        for period, grp in sm.groupby(sm.DateP.dt.to_period("M")):
            t0 = grp.DateP.min()
            tr = m[m.DateP < t0]
            teams = sorted(set(tr.HomeTeam) | set(tr.AwayTeam))
            # fusion buts + tirs cadrés paramétrable
            fg = T.dc_fit(tr, t0, xi, teams=teams, team_home=team_home, pen_home=pen_home)
            if w_sot > 0:
                ss = tr[tr["HST"].notna() & tr["AST"].notna()].copy()
                conv = (ss.FTHG.sum()+ss.FTAG.sum())/max(ss.HST.sum()+ss.AST.sum(),1)
                ss["PgH"] = np.rint(ss.HST*conv).astype(float)
                ss["PgA"] = np.rint(ss.AST*conv).astype(float)
                fs = T.dc_fit(ss, t0, xi, teams=teams, cols=("PgH","PgA"),
                              team_home=team_home, pen_home=pen_home)
                fit = {"teams": teams,
                       "attack": (1-w_sot)*fg["attack"]+w_sot*fs["attack"],
                       "defense": (1-w_sot)*fg["defense"]+w_sot*fs["defense"],
                       "gamma": (1-w_sot)*fg["gamma"]+w_sot*fs["gamma"],
                       "delta_home": (1-w_sot)*fg["delta_home"]+w_sot*fs["delta_home"],
                       "rho": fg["rho"]}
            else:
                fit = fg
            for ridx, r in grp.iterrows():
                p = T.dc_predict(fit, r.HomeTeam, r.AwayTeam)
                if p is None: continue
                probs = [p["pH"], p["pD"], p["pA"]]
                out[s]["ll"].append(-math.log(max(probs[ycls[ridx]], 1e-9)))
                Mx = T.dc_matrix(p["lam"], p["mu"], fit["rho"])
                flat = [(i, j, Mx[i, j]) for i in range(Mx.shape[0]) for j in range(Mx.shape[1])]
                flat.sort(key=lambda t: -t[2])
                gh, ga = int(r.FTHG), int(r.FTAG)
                if (flat[0][0], flat[0][1]) == (gh, ga): out[s]["h1"] += 1
                if any((i, j) == (gh, ga) for i, j, _ in flat[:3]): out[s]["h3"] += 1
                out[s]["n"] += 1
    res = {}
    for s, d in out.items():
        res[s] = {"ll": float(np.mean(d["ll"])), "h1": 100*d["h1"]/d["n"], "h3": 100*d["h3"]/d["n"]}
    lls = [x for s in seasons for x in out[s]["ll"]]
    n = sum(out[s]["n"] for s in seasons)
    res["global"] = {"ll": float(np.mean(lls)), "se": float(np.std(lls)/math.sqrt(len(lls))),
                     "h1": 100*sum(out[s]["h1"] for s in seasons)/n,
                     "h3": 100*sum(out[s]["h3"] for s in seasons)/n}
    return res

def fmt(name, r):
    g = r["global"]
    per = " | ".join(f"{s}:{r[s]['ll']:.4f}" for s in ("2324","2425","2526"))
    print(f"{name:34s} ll={g['ll']:.4f}±{g['se']:.4f}  h@1={g['h1']:.1f}%  h@3={g['h3']:.1f}%   [{per}]")

if __name__ == "__main__":
    print("Chargement…")
    m = T.load_matches("/home/claude/foot/data")
    ycls = np.where(m.FTHG > m.FTAG, 0, np.where(m.FTHG == m.FTAG, 1, 2))
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    if stage == "1":
        print("=== ÉTAGE 1 : poids tirs cadrés (xi=0.001, pen_home=25) ===")
        for w in [0.0, 0.15, 0.30, 0.45]:
            fmt(f"w_sot={w}", backtest(m, ycls, 0.001, w, 25.0))
    elif stage == "2":
        w = float(sys.argv[2])
        print(f"=== ÉTAGE 2 : domicile (w_sot={w}, xi=0.001) ===")
        fmt("global (sans delta équipe)", backtest(m, ycls, 0.001, w, 25.0, team_home=False))
        for ph in [10.0, 50.0]:
            fmt(f"pen_home={ph}", backtest(m, ycls, 0.001, w, ph))
    elif stage == "3":
        w = float(sys.argv[2]); ph = float(sys.argv[3])
        print(f"=== ÉTAGE 3 : décroissance (w_sot={w}, pen_home={ph}) ===")
        for xi in [0.0007, 0.0015]:
            fmt(f"xi={xi}", backtest(m, ycls, xi, w, ph))
