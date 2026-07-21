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

def dc_fit(sub, ref_date, xi, teams=None, x0=None, cols=("FTHG", "FTAG"),
           team_home=True, pen_home=25.0):
    """Ajuste attaque/défense/gamma(+delta_i par équipe, rétréci)/rho par MV pondérée."""
    if teams is None:
        teams = sorted(set(sub.HomeTeam) | set(sub.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = sub.HomeTeam.map(idx).to_numpy()
    ai = sub.AwayTeam.map(idx).to_numpy()
    x = sub[cols[0]].to_numpy(float); y = sub[cols[1]].to_numpy(float)
    w = np.exp(-xi * (ref_date - sub.DateP).dt.days.to_numpy(float))

    def unpack(p):
        return p[:n], p[n:2*n], p[2*n], p[2*n+1], (p[2*n+2:2*n+2+n] if team_home else np.zeros(n))

    def nll_grad(p):
        att, dfn, gam, rho, dlt = unpack(p)
        lam = np.exp(att[hi] - dfn[ai] + gam + dlt[hi])
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
        if team_home:
            np.add.at(g, 2*n+2 + hi, dlam)
            # rétrécissement des deltas domicile vers 0 (shrinkage)
            nll += pen_home * np.sum(dlt**2)
            g[2*n+2:2*n+2+n] += 2*pen_home*dlt
        # identifiabilité : pénalise sum(att), sum(def) et sum(delta)
        pen = 100.0
        nll += pen * (att.sum()**2 + dfn.sum()**2 + (dlt.sum()**2 if team_home else 0))
        g[:n] += 2*pen*att.sum(); g[n:2*n] += 2*pen*dfn.sum()
        if team_home: g[2*n+2:2*n+2+n] += 2*pen*dlt.sum()
        return nll, g

    npar = 2*n + 2 + (n if team_home else 0)
    p0 = x0 if x0 is not None and len(x0) == npar else \
         np.concatenate([np.zeros(2*n), [0.25, -0.05], np.zeros(n if team_home else 0)])
    res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 600})
    att, dfn, gam, rho, dlt = unpack(res.x)
    return {"teams": teams, "attack": att, "defense": dfn,
            "gamma": float(gam), "rho": float(np.clip(rho, -0.3, 0.3)),
            "delta_home": dlt, "x": res.x}

W_SOT = 0.15   # poids tirs cadrés — optimisé backtest 2324-2526 (optimize.py)

def dc_fit_fused(sub, ref_date, xi, teams=None):
    """Fusionne un DC sur les buts et un DC sur les tirs cadrés (proxy xG,
    moins bruité). Les tirs cadrés sont rescalés au taux de conversion moyen
    de la ligue pour rester sur l'échelle des buts."""
    fit_g = dc_fit(sub, ref_date, xi, teams=teams, team_home=False)
    sub_s = sub[sub["HST"].notna() & sub["AST"].notna()].copy()
    conv = (sub_s.FTHG.sum() + sub_s.FTAG.sum()) / max(sub_s.HST.sum() + sub_s.AST.sum(), 1)
    sub_s["PgH"] = np.rint(sub_s.HST * conv).astype(float)   # pseudo-buts entiers
    sub_s["PgA"] = np.rint(sub_s.AST * conv).astype(float)
    fit_s = dc_fit(sub_s, ref_date, xi, teams=fit_g["teams"], cols=("PgH", "PgA"),
                   team_home=False)
    w = W_SOT
    return {"teams": fit_g["teams"],
            "attack": (1-w)*fit_g["attack"] + w*fit_s["attack"],
            "defense": (1-w)*fit_g["defense"] + w*fit_s["defense"],
            "gamma": (1-w)*fit_g["gamma"] + w*fit_s["gamma"],
            "delta_home": (1-w)*fit_g["delta_home"] + w*fit_s["delta_home"],
            "rho": fit_g["rho"]}

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
    dlt = fit.get("delta_home")
    lam = math.exp(fit["attack"][i] - fit["defense"][j] + fit["gamma"] +
                   (dlt[i] if dlt is not None else 0.0))
    mu = math.exp(fit["attack"][j] - fit["defense"][i])
    M = dc_matrix(lam, mu, fit["rho"])
    return {"lam": lam, "mu": mu, "rho": fit["rho"],
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

MARGE_NUL = 0.16   # marge d'affichage du nul — optimale en hit@1 ET en réalisme

def pick_score(lam, mu, rho):
    """Règle de production : meilleure case dans la classe retenue,
    avec marge de tolérance pour le nul."""
    Mx = dc_matrix(lam, mu, rho)
    pH = float(np.tril(Mx, -1).sum()); pD = float(np.trace(Mx)); pA = float(np.triu(Mx, 1).sum())
    cls = 1 if pD >= max(pH, pA) - MARGE_NUL else (0 if pH > pA else 2)
    best, bi, bj = -1.0, 0, 0
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            c = 0 if i > j else (1 if i == j else 2)
            if c == cls and Mx[i, j] > best:
                best, bi, bj = Mx[i, j], i, j
    return bi, bj

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
        fit = dc_fit_fused(train, val.DateP.min(), xi)
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
            fit = dc_fit_fused(train, t0, xi, teams=teams)
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
        bx, by = pick_score(q["dc"]["lam"], q["dc"]["mu"], q["dc"].get("rho", -0.05))
        r = m.loc[q["idx"]]
        hits += int(bx == r.FTHG and by == r.FTAG); tot += 1
    metrics["score_exact_pct"] = round(100.0 * hits / tot, 2)
    metrics["n_matchs_eval"] = tot
    print("Backtest :", json.dumps(metrics, indent=1, ensure_ascii=False))

    # --- blend final (toutes les prédictions de backtest)
    W = blend_fit(np.array([q["f"] for q in preds]), np.array([q["y"] for q in preds]))

    # --- VALIDATION par saison : métriques, simulation de saison, chaos
    validation = {"saisons": {}, "chaos": {}}
    rng = np.random.default_rng(42)
    for s in blend_eval_seasons:
        cur = [q for q in preds if q["season"] == s and q["mkt"] is not None]
        vs = {}
        for name, get_p in [("dc", lambda q: [q["dc"]["pH"], q["dc"]["pD"], q["dc"]["pA"]]),
                            ("blend", lambda q: list(q["blend"])),
                            ("marche", lambda q: list(q["mkt"]))]:
            vs[name] = {
                "logloss": round(float(np.mean([-math.log(max(get_p(q)[q["y"]], 1e-9)) for q in cur])), 4),
                "acc": round(float(np.mean([int(np.argmax(get_p(q)) == q["y"]) for q in cur])), 4)}
        hits = 0
        for q in cur:
            bx, by = pick_score(q["dc"]["lam"], q["dc"]["mu"], q["dc"].get("rho", -0.05))
            r = m.loc[q["idx"]]
            hits += int(bx == r.FTHG and by == r.FTAG)
        vs["score_exact_pct"] = round(100.0 * hits / len(cur), 1)
        vs["n"] = len(cur)

        # simulation de la saison complète avec le modèle d'AVANT-saison
        sm = m[m.Season == s]
        t0 = sm.DateP.min()
        pre = dc_fit_fused(m[m.DateP < t0], t0, xi)
        teams_s = sorted(set(sm.HomeTeam) | set(sm.AwayTeam))
        ti = {t: i for i, t in enumerate(teams_s)}
        # matrices de chaque match du calendrier réel
        mats = []
        for _, r in sm.iterrows():
            p = dc_predict(pre, r.HomeTeam, r.AwayTeam)
            if p is None:  # promu sans historique : prior moyen
                p = {"lam": 1.25, "mu": 1.35}
            mats.append((ti[r.HomeTeam], ti[r.AwayTeam],
                         dc_matrix(p["lam"], p["mu"], pre["rho"]).ravel()))
        NS, K = 2000, MAX_GOALS + 1
        pts_sim = np.zeros((NS, len(teams_s)))
        for hi_, ai_, mat in mats:
            draws = rng.choice(K*K, size=NS, p=mat/mat.sum())
            gh, ga = draws // K, draws % K
            pts_sim[:, hi_] += np.where(gh > ga, 3, np.where(gh == ga, 1, 0))
            pts_sim[:, ai_] += np.where(ga > gh, 3, np.where(gh == ga, 1, 0))
        xpts = pts_sim.mean(axis=0)
        # classement réel
        real = {t: 0 for t in teams_s}
        for _, r in sm.iterrows():
            if r.FTHG > r.FTAG: real[r.HomeTeam] += 3
            elif r.FTHG < r.FTAG: real[r.AwayTeam] += 3
            else: real[r.HomeTeam] += 1; real[r.AwayTeam] += 1
        order_p = sorted(teams_s, key=lambda t: -xpts[ti[t]])
        order_r = sorted(teams_s, key=lambda t: -real[t])
        rk_p = {t: i+1 for i, t in enumerate(order_p)}
        rk_r = {t: i+1 for i, t in enumerate(order_r)}
        d_rank = [rk_p[t] - rk_r[t] for t in teams_s]
        n_t = len(teams_s)
        rho_s = 1 - 6*sum(d*d for d in d_rank) / (n_t*(n_t**2 - 1))
        errs = sorted(teams_s, key=lambda t: -abs(rk_p[t]-rk_r[t]))
        vs["simulation"] = {
            "mae_pts": round(float(np.mean([abs(xpts[ti[t]] - real[t]) for t in teams_s])), 1),
            "spearman": round(float(rho_s), 3),
            "champion_predit": order_p[0], "champion_reel": order_r[0],
            "rates": [{"club": t, "predit": rk_p[t], "reel": rk_r[t]} for t in errs[:3]],
            "reussites": [{"club": t, "predit": rk_p[t], "reel": rk_r[t]}
                          for t in teams_s if rk_p[t] == rk_r[t]][:3]}
        validation["saisons"][s] = vs

    # chaos rétrospectif — deux questions distinctes :
    # (1) l'IRRÉGULARITÉ (notre apport au-delà de l'entropie) rend-elle les matchs
    #     plus durs à prédire ? -> log-loss du blend par tiers d'irrégularité
    # (2) le favori du modèle est-il plus souvent BATTU dans les matchs irréguliers ?
    def season_ptstd(season):
        sm_ = m[m.Season == season]; d = {}
        for _, r in sm_.iterrows():
            ph = 3 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 0)
            pa = 3 if r.FTAG > r.FTHG else (1 if r.FTHG == r.FTAG else 0)
            d.setdefault(r.HomeTeam, []).append(ph); d.setdefault(r.AwayTeam, []).append(pa)
        return {t: float(np.std(v)) for t, v in d.items()}
    seasons_sorted = sorted(m.Season.unique())
    rows = []
    for q in preds:
        if q["season"] not in blend_eval_seasons or q["mkt"] is None: continue
        prev = seasons_sorted[seasons_sorted.index(q["season"]) - 1]
        stds = season_ptstd(prev)
        vals = sorted(stds.values()); lo_, hi_ = vals[0], vals[-1]
        r = m.loc[q["idx"]]
        irr = float(np.mean([(stds.get(t, vals[len(vals)//2]) - lo_) / (hi_ - lo_)
                             for t in (r.HomeTeam, r.AwayTeam)]))
        pb = list(q["blend"]); fav = int(np.argmax(pb))
        beaten = int((fav == 0 and q["y"] == 2) or (fav == 2 and q["y"] == 0))
        ll = -math.log(max(pb[q["y"]], 1e-9))
        rows.append((irr, ll, beaten))
    rows.sort()
    n3 = len(rows) // 3
    lo_t, hi_t = rows[:n3], rows[-n3:]
    validation["chaos"] = {
        "logloss_irr_basse": round(float(np.mean([x[1] for x in lo_t])), 3),
        "logloss_irr_haute": round(float(np.mean([x[1] for x in hi_t])), 3),
        "favori_battu_irr_basse_pct": round(100*float(np.mean([x[2] for x in lo_t])), 1),
        "favori_battu_irr_haute_pct": round(100*float(np.mean([x[2] for x in hi_t])), 1),
        "n_par_tiers": n3,
        "lecture": "tiers bas vs haut d'irrégularité (constance des points saison précédente)"}
    print("Validation chaos :", validation["chaos"])

    # --- distribution des totaux de buts (backtest) : réel vs modèle, par équipe
    KMAXD = 7
    def empty(): return [0.0]*(KMAXD+1)
    dist = {"reel": {"Toutes": empty()}, "modele": {"Toutes": empty()},
            "n": {"Toutes": 0}}
    for q in preds:
        if q["season"] not in blend_eval_seasons: continue
        r = m.loc[q["idx"]]
        tot_r = min(int(r.FTHG + r.FTAG), KMAXD)
        Mx = dc_matrix(q["dc"]["lam"], q["dc"]["mu"], 0.0)
        pt = [0.0]*(KMAXD+1)
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                pt[min(i+j, KMAXD)] += Mx[i, j]
        for key in ("Toutes", r.HomeTeam, r.AwayTeam):
            for d in (dist["reel"], dist["modele"], dist["n"]):
                d.setdefault(key, empty() if d is not dist["n"] else 0)
            dist["reel"][key][tot_r] += 1
            dist["modele"][key] = [a+b for a, b in zip(dist["modele"][key], pt)]
            dist["n"][key] += 1
    for kind in ("reel", "modele"):
        for key, v in dist[kind].items():
            n_ = max(dist["n"][key], 1)
            dist[kind][key] = [round(x/n_, 4) for x in v]
    validation["dist_buts"] = {"reel": dist["reel"], "modele": dist["modele"],
                               "n": dist["n"], "kmax": KMAXD}

    # --- modèle final sur tout l'historique
    now = m.DateP.max() + timedelta(days=1)
    final = dc_fit_fused(m, now, xi)
    cur_teams = sorted(set(m[m.Season == m.Season.max()].HomeTeam) |
                       set(m[m.Season == m.Season.max()].AwayTeam))
    # calendrier de la saison cible : statuts actifs + équipes sans historique
    fix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.json")
    fixture_teams, new_teams = None, []
    if os.path.exists(fix_path):
        fx = json.load(open(fix_path, encoding="utf-8"))
        fixture_teams = sorted({f["h"] for f in fx["fixtures"]} |
                               {f["a"] for f in fx["fixtures"]})
        cur_teams = fixture_teams
        new_teams = [t for t in fixture_teams if t not in final["teams"]]
    model = {
        "meta": {"genere": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                 "n_matchs": int(len(m)), "saisons": sorted(m.Season.unique()),
                 "derniere_date": str(m.DateP.max().date()),
                 "xi": xi, "demi_vie_jours": round(math.log(2)/xi),
                 "elo_home": ELO_HOME, "max_goals": MAX_GOALS},
        "gamma": final["gamma"], "rho": final["rho"],
        "equipes": {t: {"attaque": round(float(final["attack"][final["teams"].index(t)]), 4),
                        "defense": round(float(final["defense"][final["teams"].index(t)]), 4),
                        "delta_dom": round(float(final["delta_home"][final["teams"].index(t)]), 4),
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
        "validation": validation,
    }
    if new_teams:
        ref = [t for t in ["Angers", "Le Havre", "Auxerre", "Lorient"]
               if t in model["equipes"]]
        p_att = float(np.mean([model["equipes"][t]["attaque"] for t in ref])) - 0.05
        p_def = float(np.mean([model["equipes"][t]["defense"] for t in ref])) - 0.05
        for t in new_teams:
            model["equipes"][t] = {"attaque": round(p_att, 4), "defense": round(p_def, 4),
                                   "delta_dom": 0.0,
                                   "elo": ELO_PROMOTED, "forme5": 1.0, "dernier_match": "",
                                   "actif": True, "promu": True,
                                   "prior": "promu (moyenne clubs modestes - malus)"}
    if fixture_teams:
        prev = set(m[m.Season == m.Season.max()].HomeTeam) | \
               set(m[m.Season == m.Season.max()].AwayTeam)
        for t, e in model["equipes"].items():
            e["actif"] = t in fixture_teams
            e["promu"] = bool(t in fixture_teams and t not in prev)
    # régularité (constance des points, dernière saison) et volatilité (écart de buts)
    last = m[m.Season == m.Season.max()]
    pts_l, gd_l = {}, {}
    for _, r in last.iterrows():
        ph = 3 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 0)
        pa = 3 if r.FTAG > r.FTHG else (1 if r.FTHG == r.FTAG else 0)
        pts_l.setdefault(r.HomeTeam, []).append(ph); pts_l.setdefault(r.AwayTeam, []).append(pa)
        gd_l.setdefault(r.HomeTeam, []).append(r.FTHG - r.FTAG)
        gd_l.setdefault(r.AwayTeam, []).append(r.FTAG - r.FTHG)
    stdp = {t: float(np.std(v)) for t, v in pts_l.items()}
    ref_reg = sorted(stdp.get(t, None) for t in cur_teams if stdp.get(t) is not None)
    p75 = ref_reg[int(0.75 * len(ref_reg))]
    lo_r, hi_r = min(ref_reg), max(ref_reg)
    for t, e in model["equipes"].items():
        v = stdp.get(t, p75)
        e["volatilite"] = round(float(np.std(gd_l[t])) if t in gd_l else 1.9, 3)
        e["regularite"] = round(max(0.0, min(1.0, 1 - (v - lo_r) / (hi_r - lo_r))), 3)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    print(f"-> {out} ({os.path.getsize(out)//1024} Ko)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="répertoire de CSVs locaux")
    run(ap.parse_args().local)
