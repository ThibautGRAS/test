#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Journal de prédictions figées — Génie du Foot.
À chaque exécution (hebdo, via GitHub Action) :
1. évalue les prédictions déjà journalisées dont le résultat réel est connu ;
2. fige les prédictions des matchs des 9 prochains jours (non encore journalisés).
Le journal est inviolable : une prédiction émise n'est jamais modifiée.
"""
import json, math, os, sys
from datetime import datetime, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name, default):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default

def softmax(z):
    z = np.asarray(z); z = z - z.max()
    e = np.exp(z); return e / e.sum()

def predire(M, NEWS, h, a, rh=7, ra=7):
    th, ta = M["equipes"][h], M["equipes"][a]
    nh = NEWS.get("equipes", {}).get(h, {"att": 0, "def": 0})
    na = NEWS.get("equipes", {}).get(a, {"att": 0, "def": 0})
    lam = math.exp(th["attaque"]+nh["att"] - (ta["defense"]+na["def"]) + M["gamma"])
    mu = math.exp(ta["attaque"]+na["att"] - (th["defense"]+nh["def"]))
    Mx = T.dc_matrix(lam, mu, M["rho"])
    pH = float(np.tril(Mx, -1).sum()); pD = float(np.trace(Mx)); pA = float(np.triu(Mx, 1).sum())
    lg = lambda p: math.log(max(p, 1e-9))
    f = [lg(pH), lg(pD), lg(pA), lg(pH), lg(pD), lg(pA), 0.0,
         (th["elo"] + M["meta"]["elo_home"] - ta["elo"]) / 100.0,
         (rh - ra) / 3.0, (th["forme5"] - ta["forme5"]) / 3.0,
         (1 if th.get("promu") else 0) - (1 if ta.get("promu") else 0)]
    pb = softmax([sum(w*x for w, x in zip(row, f)) for row in M["blend"]["W"]])
    bi, bj = T.pick_score(lam, mu, M["rho"])
    # score dans la classe calibrée (marge nul) — répliquer la règle app sur pb
    mx = max(pb[0], pb[2])
    cls = 1 if pb[1] >= mx - 0.16 else (0 if pb[0] > pb[2] else 2)
    best, bi, bj = -1.0, 0, 0
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            c = 0 if i > j else (1 if i == j else 2)
            if c == cls and Mx[i, j] > best:
                best, bi, bj = float(Mx[i, j]), i, j
    return {"score": f"{bi}-{bj}", "p_score": round(best/Mx.sum(), 4),
            "p1n2": [round(float(x), 4) for x in pb], "lam": round(lam, 3), "mu": round(mu, 3),
            "news_adj": {"h": nh, "a": na}}

def main():
    M = load("model.json", None)
    FIX = load("fixtures.json", {"fixtures": []})
    NEWS = load("news.json", {"equipes": {}})
    J = load("journal.json", {"entrees": [], "bilan": {}})
    if M is None:
        print("model.json manquant"); return

    # 1) évaluation des prédictions passées contre les résultats connus
    try:
        m = T.load_matches(sys.argv[1] if len(sys.argv) > 1 else None)
        reels = {(str(r.DateP.date()), r.HomeTeam, r.AwayTeam): (int(r.FTHG), int(r.FTAG))
                 for _, r in m.iterrows()}
    except Exception as e:
        print("évaluation ignorée:", e); reels = {}
    n_ev, h1, lls = 0, 0, []
    for e in J["entrees"]:
        key = (e["date"], e["h"], e["a"])
        if key in reels and "reel" not in e:
            gh, ga = reels[key]
            e["reel"] = f"{gh}-{ga}"
            e["hit"] = int(e["pred"]["score"] == e["reel"])
            y = 0 if gh > ga else (1 if gh == ga else 2)
            e["ll"] = round(-math.log(max(e["pred"]["p1n2"][y], 1e-9)), 4)
        if "reel" in e:
            n_ev += 1; h1 += e.get("hit", 0); lls.append(e.get("ll", 0))
    if n_ev:
        J["bilan"] = {"n": n_ev, "hit1_pct": round(100*h1/n_ev, 2),
                      "logloss": round(float(np.mean(lls)), 4),
                      "maj": datetime.utcnow().strftime("%Y-%m-%d")}
        print(f"bilan live: {n_ev} matchs, hit@1 {J['bilan']['hit1_pct']}%, ll {J['bilan']['logloss']}")

    # 2) figer les matchs des 9 prochains jours non journalisés
    today = datetime.utcnow().date()
    deja = {(e["date"], e["h"], e["a"]) for e in J["entrees"]}
    n_new = 0
    for f in FIX["fixtures"]:
        d = datetime.strptime(f["date"], "%Y-%m-%d").date()
        if not (today <= d <= today + timedelta(days=9)): continue
        if (f["date"], f["h"], f["a"]) in deja: continue
        if f["h"] not in M["equipes"] or f["a"] not in M["equipes"]: continue
        pred = predire(M, NEWS, f["h"], f["a"], f.get("rh", 7), f.get("ra", 7))
        J["entrees"].append({"j": f["j"], "date": f["date"], "h": f["h"], "a": f["a"],
                             "pred": pred,
                             "fige_le": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")})
        n_new += 1
    print(f"{n_new} prédictions figées, {len(J['entrees'])} au total")
    json.dump(J, open(os.path.join(HERE, "journal.json"), "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
