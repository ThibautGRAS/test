#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcule le solde net estival par club L1 depuis un CSV de transferts
(schéma ewenme/transfers) et propose les ajustements attaque calibrés
(+0,0218 par z-score, calibré sur 108 club-saisons 2017-2022).
Usage : python3 mercato.py <transfers.csv> <annee>   (ex : 2026)
La GitHub Action peut télécharger un export Transfermarkt récent
(runners = accès internet complet) et invoquer ce script."""
import csv, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
COEF = 0.0218
MAP = {"Paris Saint-Germain":"Paris SG","Olympique Marseille":"Marseille",
"Olympique Lyon":"Lyon","LOSC Lille":"Lille","AS Monaco":"Monaco","OGC Nice":"Nice",
"RC Lens":"Lens","Stade Rennais FC":"Rennes","RC Strasbourg Alsace":"Strasbourg",
"Stade Brestois 29":"Brest","FC Toulouse":"Toulouse","Toulouse FC":"Toulouse",
"Le Havre AC":"Le Havre","AJ Auxerre":"Auxerre","Angers SCO":"Angers",
"FC Lorient":"Lorient","ES Troyes AC":"Troyes","Le Mans FC":"Le Mans","Paris FC":"Paris FC"}

def main(path, year):
    net = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get('transfer_period') != 'Summer' or r.get('year') != str(year): continue
        club = MAP.get(r['club_name'])
        if not club: continue
        try: fee = float(r['fee_cleaned'])
        except (ValueError, TypeError): continue
        net[club] = net.get(club, 0.0) + (fee if r['transfer_movement']=='in' else -fee)
    if len(net) < 8:
        print(f"seulement {len(net)} clubs — données incomplètes, abandon"); return
    vals = list(net.values())
    mu = sum(vals)/len(vals)
    sd = math.sqrt(sum((v-mu)**2 for v in vals)/len(vals)) or 1.0
    z = {t: (v-mu)/sd for t, v in net.items()}
    print(f"{'club':14s} {'net M€':>8s} {'z':>6s} {'adj att':>8s}")
    for t, v in sorted(net.items(), key=lambda x: -x[1]):
        print(f"{t:14s} {v:+8.1f} {z[t]:+6.2f} {COEF*z[t]:+8.4f}")
    p = os.path.join(HERE, "news.json")
    news = json.load(open(p, encoding="utf-8"))
    news.setdefault("mercato", {})["z_net_definitif_"+str(year)] = {t: round(x,2) for t,x in z.items()}
    for t, x in z.items():
        e = news["equipes"].setdefault(t, {"att":0.0,"def":0.0,"titres":[]})
        e["att"] = round(COEF*x, 4)
        e["titres"] = [f"Solde net été {year} : {net[t]:+.0f} M€ (z={x:+.2f}) — ajustement calibré"] + \
                      [h for h in e.get("titres", []) if "Solde net" not in h]
    json.dump(news, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> news.json mis à jour (ajustements attaque = calibration)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
