#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génie du Foot — Pipeline d'entraînement Ligue 1
================================================
Modèle Dixon-Coles (Poisson bivarié + correction faibles scores) avec
pondération temporelle exponentielle, ratings Elo, features contextuelles
(jours de repos, forme, promus) et couche de calibration 1N2 blendant
le modèle avec les cotes du marché.

Sortie : model.json (paramètres + backtest) consommé par l'app web.

Usage :
  python3 train.py --local data/     # CSVs locaux F1_XXXX.csv
  python3 train.py                   # télécharge depuis football-data.co.uk
"""
import argparse, json, math, os, sys, urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_GOALS = 8          # taille de la matrice de scores
ELO_K = 20.0
ELO_HOME = 60.0
ELO_START = 1500.0
ELO_PROMOTED = 1450.0
DECAY_GRID = [0.0010, 0.0015, 0.0020, 0.0030]   # xi par jour
BLEND_L2 = 1.0

# ---------------------------------------------------------------- données

def season_codes(first="1617"):
    """Codes saisons de first jusqu'à la saison en cours."""
    now = datetime.utcnow()
    last_start = now.year if now.month >= 7 else now.year - 1
    codes, y = [], int(first[:2]) + 2000
    while y <= last_start:
        codes.append(f"{y%100:02d}{(y+1)%100:02d}")
        y += 1
    return codes

def load_matches(local_dir=None):
    frames = []
    for code in season_codes():
        df = None
        if local_dir:
            p = os.path.join(local_dir, f"F1_{code}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
        else:
            url = f"https://www.football-data.co.uk/mmz4281/{code}/F1.csv"
            try:
                df = pd.read_csv(url, encoding="utf-8", encoding_errors="replace")
            except Exception as e:
                print(f"  saison {code}: indisponible ({e})", file=sys.stderr)
        if df is None or df.empty:
            continue
        df = df[df["HomeTeam"].notna() & df["FTHG"].notna()].copy()
        df["Season"] = code
        frames.append(df)
        print(f"  saison {code}: {len(df)} matchs")
    m = pd.concat(frames, ignore_index=True)
    m["DateP"] = pd.to_datetime(m["Date"], dayfirst=True, format="mixed")
    m = m.sort_values("DateP").reset_index(drop=True)
    for c in ["FTHG", "FTAG"]:
        m[c] = m[c].astype(int)
    # cotes marché : Avg si dispo, sinon B365, sinon Pinnacle
    for side in ["H", "D", "A"]:
        m[f"Mkt{side}"] = np.nan
        for pref in ["Avg", "B365", "PS"]:
            col = pref + side
            if col in m.columns:
                m[f"Mkt{side}"] = m[f"Mkt{side}"].fillna(pd.to_numeric(m[col], errors="coerce"))
    return m

# ---------------------------------------------------- features contextuelles

def add_context(m):
    elo, last_date, form = {}, {}, {}
    season_teams = {s: set(g) for s, g in
                    m.groupby("Season").apply(lambda g: set(g.HomeTeam) | set(g.AwayTeam)).items()}
    codes = sorted(season_teams)
    promoted = {}
    for i, s in enumerate(codes):
        prev = season_teams[codes[i-1]] if i > 0 else season_teams[s]
        promoted[s] = {t for t in season_teams[s] if t not in prev}

    cols = {k: [] for k in ["EloH", "EloA", "RestH", "RestA", "FormH", "FormA",
                            "PromH", "PromA"]}
    for _, r in m.iterrows():
        h, a, d, s = r.HomeTeam, r.AwayTeam, r.DateP, r.Season
        for t in (h, a):
            if t not in elo:
                elo[t] = ELO_PROMOTED if t in promoted[s] else ELO_START
        cols["EloH"].append(elo[h]); cols["EloA"].append(elo[a])
        cols["RestH"].append(min((d - last_date.get(h, d - timedelta(days=7))).days, 14))
        cols["RestA"].append(min((d - last_date.get(a, d - timedelta(days=7))).days, 14))
        fh, fa = form.get(h, []), form.get(a, [])
        cols["FormH"].append(sum(fh[-5:]) / max(len(fh[-5:]), 1))
        cols["FormA"].append(sum(fa[-5:]) / max(len(fa[-5:]), 1))
        cols["PromH"].append(1.0 if h in promoted[s] else 0.0)
        cols["PromA"].append(1.0 if a in promoted[s] else 0.0)
        # mises à jour post-match
        gh, ga = r.FTHG, r.FTAG
        exp_h = 1 / (1 + 10 ** (-((elo[h] + ELO_HOME) - elo[a]) / 400))
        res_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        mult = math.log(abs(gh - ga) + 1) + 1
        delta = ELO_K * mult * (res_h - exp_h)
        elo[h] += delta; elo[a] -= delta
        last_date[h] = d; last_date[a] = d
        ph = 3 if gh > ga else (1 if gh == ga else 0)
        pa = 3 if ga > gh else (1 if gh == ga else 0)
        form.setdefault(h, []).append(ph); form.setdefault(a, []).append(pa)
    for k, v in cols.items():
        m[k] = v
    state = {"elo": elo, "last_date": {t: str(d.date()) for t, d in last_date.items()},
             "form5": {t: sum(v[-5:]) / max(len(v[-5:]), 1) for t, v in form.items()},
             "promoted_current": sorted(promoted[codes[-1]])}
    return m, state

# ------------------------------------------------------------- Dixon-Coles

def dc_fit(sub, ref_date, xi, teams=None, x0=None):
    """Ajuste attaque/défense/gamma/rho par max de vraisemblance pondérée."""
    if teams is None:
        teams = sorted(set(sub.HomeTeam) | set(sub.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = sub.HomeTeam.map(idx).to_numpy()
    ai = sub.AwayTeam.map(idx).to_numpy()
    x = sub.FTHG.to_numpy(float); y = sub.FTAG.to_numpy(float)
    w = np.exp(-xi * (ref_date - sub.DateP).dt.days.to_numpy(float))

    def unpack(p):
        return p[:n], p[n:2*n], p[2*n], p[2*n+1]

    def nll_grad(p):
        att, dfn, gam, rho = unpack(p)
        lam = np.exp(att[hi] - dfn[ai] + gam)
        mu = np.exp(att[ai] - dfn[hi])
        # correction tau et dérivées
        tau = np.ones_like(lam); dtl = np.zeros_like(lam)
        dtm = np.zeros_like(lam); dtr = np.zeros_like(lam)
        m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)
        tau[m00] = 1 - lam[m00]*mu[m00]*rho; dtl[m00] = -mu[m00]*rho
        dtm[m00] = -lam[m00]*rho;            dtr[m00] = -lam[m00]*mu[m00]
        tau[m01] = 1 + lam[m01]*rho; dtl[m01] = rho; dtr[m01] = lam[m01]
        tau[m10] = 1 + mu[m10]*rho;  dtm[m10] = rho; dtr[m10] = mu[m10]
        tau[m11] = 1 - rho;          dtr[m11] = -1.0
        tau = np.clip(tau, 1e-10, None)
        nll = np.sum(w * (-np.log(tau) + lam - x*np.log(lam) + mu - y*np.log(mu)))
        # gradients
        dlam = w * (lam - x - dtl/tau*lam)      # d/d(log lam) * ... via chain
        dmu  = w * (mu  - y - dtm/tau*mu)
        g = np.zeros_like(p)
        np.add.at(g, hi, dlam);        np.add.at(g, n + ai, -dlam)
        np.add.at(g, ai, dmu);         np.add.at(g, n + hi, -dmu)
        g[2*n] = dlam.sum()
        g[2*n+1] = -np.sum(w * dtr / tau)
        # identifiabilité : pénalise sum(att) et sum(def)
        pen = 100.0
        nll += pen * (att.sum()**2 + dfn.sum()**2)
        g[:n] += 2*pen*att.sum(); g[n:2*n] += 2*pen*dfn.sum()
        return nll, g

    p0 = x0 if x0 is not None else np.concatenate([np.zeros(2*n), [0.25, -0.05]])
    res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    att, dfn, gam, rho = unpack(res.x)
    return {"teams": teams, "attack": att, "defense": dfn,
            "gamma": float(gam), "rho": float(np.clip(rho, -0.3, 0.3)),
            "x": res.x}

def dc_matrix(lam, mu, rho, kmax=MAX_GOALS):
    k = np.arange(kmax + 1)
    ph = np.exp(-lam) * lam**k / np.array([math.factorial(i) for i in k])
    pa = np.exp(-mu) * mu**k / np.array([math.factorial(i) for i in k])
    M = np.outer(ph, pa)
    M[0,0] *= 1 - lam*mu*rho; M[0,1] *= 1 + lam*rho
    M[1,0] *= 1 + mu*rho;     M[1,1] *= 1 - rho
    return np.clip(M, 0, None) / np.clip(M, 0, None).sum()

def dc_predict(fit, home, away):
    if home not in fit["teams"] or away not in fit["teams"]:
        return None
    i = fit["teams"].index(home); j = fit["teams"].index(away)
    lam = math.exp(fit["attack"][i] - fit["defense"][j] + fit["gamma"])
    mu = math.exp(fit["attack"][j] - fit["defense"][i])
    M = dc_matrix(lam, mu, fit["rho"])
    return {"lam": lam, "mu": mu,
            "pH": float(np.tril(M, -1).sum()),
            "pD": float(np.trace(M)),
            "pA": float(np.triu(M, 1).sum())}

# ----------------------------------------------------------- couche blend

def devig(oh, od, oa):
    if not (oh and od and oa) or min(oh, od, oa) <= 1.0:
        return None
    r = np.array([1/oh, 1/od, 1/oa])
    return r / r.sum()

def blend_features(row, dcp):
    mkt = devig(row.get("MktH"), row.get("MktD"), row.get("MktA"))
    f = [math.log(max(dcp["pH"], 1e-9)), math.log(max(dcp["pD"], 1e-9)),
         math.log(max(dcp["pA"], 1e-9))]
    if mkt is not None:
        f += [math.log(mkt[0]), math.log(mkt[1]), math.log(mkt[2]), 1.0]
    else:
        f += [f[0], f[1], f[2], 0.0]   # repli : modèle seul
    f += [(row["EloH"] + ELO_HOME - row["EloA"]) / 100.0,
          (row["RestH"] - row["RestA"]) / 3.0,
          (row["FormH"] - row["FormA"]) / 3.0,
          row["PromH"] - row["PromA"]]
    return np.array(f)

def blend_fit(F, Y):
    """Régression logistique multinomiale L2 : 3 classes, poids partagés
    par softmax(W·f) où W est (3, d)."""
    d = F.shape[1]
    def nll_grad(wf):
        W = wf.reshape(3, d)
        Z = F @ W.T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        nll = -np.sum(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None)))
        nll += BLEND_L2 * np.sum(W**2)
        G = (P - np.eye(3)[Y]).T @ F + 2 * BLEND_L2 * W
        return nll, G.ravel()
    res = minimize(nll_grad, np.zeros(3*d), jac=True, method="L-BFGS-B")
    return res.x.reshape(3, d)

def blend_predict(W, f):
    z = W @ f; z -= z.max()
    p = np.exp(z); return p / p.sum()

# ---------------------------------------------------------------- backtest

def rps(p, outcome):  # Ranked Probability Score (H=0,D=1,A=2)
    c = np.cumsum(p); o = np.cumsum(np.eye(3)[outcome])
    return float(np.sum((c - o)**2) / 2)

def run(local_dir):
    print("Chargement des données…")
    m = load_matches(local_dir)
    print(f"Total : {len(m)} matchs, {m.DateP.min().date()} -> {m.DateP.max().date()}")
    m, state = add_context(m)
    ycls = np.where(m.FTHG > m.FTAG, 0, np.where(m.FTHG == m.FTAG, 1, 2))

    # --- choix du decay xi par validation (log-loss DC sur saison 2122)
    val = m[m.Season == "2122"]
    best_xi, best_ll = None, 1e9
    for xi in DECAY_GRID:
        train = m[m.DateP < val.DateP.min()]
        fit = dc_fit(train, val.DateP.min(), xi)
        lls = []
        for _, r in val.iterrows():
            p = dc_predict(fit, r.HomeTeam, r.AwayTeam)
            if p:
                lls.append(-math.log(max([p["pH"], p["pD"], p["pA"]][
                    0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2)], 1e-9)))
        ll = float(np.mean(lls))
        print(f"  xi={xi}: log-loss val={ll:.4f}")
        if ll < best_ll:
            best_ll, best_xi = ll, xi
    xi = best_xi
    print(f"Decay retenu : xi={xi}/jour (demi-vie {math.log(2)/xi:.0f} j)")

    # --- backtest walk-forward, refit mensuel, saisons 2223 -> fin
    test_seasons = [s for s in sorted(m.Season.unique()) if s >= "2223"]
    preds = []
    fit_cache_x = None
    for s in test_seasons:
        sm = m[m.Season == s]
        for period, grp in sm.groupby(sm.DateP.dt.to_period("M")):
            t0 = grp.DateP.min()
            train = m[m.DateP < t0]
            teams = sorted(set(train.HomeTeam) | set(train.AwayTeam))
            fit = dc_fit(train, t0, xi, teams=teams)
            for ridx, r in grp.iterrows():
                p = dc_predict(fit, r.HomeTeam, r.AwayTeam)
                if p is None:
                    continue
                preds.append({"idx": ridx, "season": s, "dc": p,
                              "f": blend_features(r, p), "y": int(ycls[ridx]),
                              "mkt": devig(r.MktH, r.MktD, r.MktA)})
        print(f"  backtest {s}: ok ({len([q for q in preds if q['season']==s])} matchs)")

    # --- blend walk-forward par saison (entraîné sur saisons de test antérieures)
    metrics = {}
    blend_eval_seasons = test_seasons[1:]
    for s in blend_eval_seasons:
        hist = [q for q in preds if q["season"] < s]
        cur = [q for q in preds if q["season"] == s]
        W = blend_fit(np.array([q["f"] for q in hist]), np.array([q["y"] for q in hist]))
        for q in cur:
            q["blend"] = blend_predict(W, q["f"])
    ev = [q for q in preds if q["season"] in blend_eval_seasons and q["mkt"] is not None]
    def summarize(get_p, name):
        ll = np.mean([-math.log(max(get_p(q)[q["y"]], 1e-9)) for q in ev])
        r = np.mean([rps(np.array(get_p(q)), q["y"]) for q in ev])
        acc = np.mean([int(np.argmax(get_p(q)) == q["y"]) for q in ev])
        metrics[name] = {"logloss": round(float(ll), 4), "rps": round(float(r), 4),
                         "accuracy": round(float(acc), 4)}
    summarize(lambda q: [q["dc"]["pH"], q["dc"]["pD"], q["dc"]["pA"]], "dixon_coles")
    summarize(lambda q: list(q["blend"]), "blend")
    summarize(lambda q: list(q["mkt"]), "marche")
    # taux de score exact (argmax de la matrice DC)
    hits, tot = 0, 0
    for q in ev:
        lam, mu = q["dc"]["lam"], q["dc"]["mu"]
        M = dc_matrix(lam, mu, 0.0)
        bx, by = np.unravel_index(M.argmax(), M.shape)
        r = m.loc[q["idx"]]
        hits += int(bx == r.FTHG and by == r.FTAG); tot += 1
    metrics["score_exact_pct"] = round(100.0 * hits / tot, 2)
    metrics["n_matchs_eval"] = tot
    print("Backtest :", json.dumps(metrics, indent=1, ensure_ascii=False))

    # --- blend final (toutes les prédictions de backtest)
    W = blend_fit(np.array([q["f"] for q in preds]), np.array([q["y"] for q in preds]))

    # --- modèle final sur tout l'historique
    now = m.DateP.max() + timedelta(days=1)
    final = dc_fit(m, now, xi)
    cur_teams = sorted(set(m[m.Season == m.Season.max()].HomeTeam) |
                       set(m[m.Season == m.Season.max()].AwayTeam))
    model = {
        "meta": {"genere": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                 "n_matchs": int(len(m)), "saisons": sorted(m.Season.unique()),
                 "derniere_date": str(m.DateP.max().date()),
                 "xi": xi, "demi_vie_jours": round(math.log(2)/xi),
                 "elo_home": ELO_HOME, "max_goals": MAX_GOALS},
        "gamma": final["gamma"], "rho": final["rho"],
        "equipes": {t: {"attaque": round(float(final["attack"][final["teams"].index(t)]), 4),
                        "defense": round(float(final["defense"][final["teams"].index(t)]), 4),
                        "elo": round(state["elo"].get(t, ELO_START), 1),
                        "forme5": round(state["form5"].get(t, 1.0), 2),
                        "dernier_match": state["last_date"].get(t, ""),
                        "actif": t in cur_teams,
                        "promu": t in state["promoted_current"]}
                    for t in final["teams"]},
        "blend": {"W": [[round(float(v), 5) for v in row] for row in W],
                  "features": ["log_pH_dc", "log_pD_dc", "log_pA_dc",
                               "log_pH_mkt", "log_pD_mkt", "log_pA_mkt", "mkt_dispo",
                               "elo_diff_100", "repos_diff_3", "forme_diff_3", "promu_diff"]},
        "backtest": metrics,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    print(f"-> {out} ({os.path.getsize(out)//1024} Ko)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="répertoire de CSVs locaux")
    run(ap.parse_args().local)
