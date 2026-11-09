"""Étape 6 - Analyse du test A/B + rapport (CML) + décision de promotion.

Métriques MÉTIER par variante : CTR, taux d'achat, revenu par impression.
Test statistique : z-test de deux proportions sur le CTR vs contrôle,
avec correction de Bonferroni (3 comparaisons -> alpha/3).
Décision : gagnant = meilleur CTR parmi les variantes significativement
meilleures que le contrôle ET avec assez d'impressions ; sinon on garde le contrôle.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from src.ab import load_config


def z_test(c1, n1, c2, n2):
    """Test bilatéral H0: p1 == p2. Retourne (z, p-value)."""
    p = (c1 + c2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 0.0, 1.0
    z = (c2 / n2 - c1 / n1) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def load_events(path="data/events.jsonl"):
    return pd.read_json(path, lines=True)


def summarize(ev):
    imp = ev[ev.type == "impression"].groupby("variant").size().rename("impressions")
    # une impression = 1 liste de 5 articles ; on compte les impressions-articles
    ev_imp = ev[ev.type == "impression"].copy()
    ev_imp["n_items"] = ev_imp["items"].apply(len)
    shown = ev_imp.groupby("variant").n_items.sum().rename("items_shown")
    clicks = ev[ev.type == "click"].groupby("variant").size().rename("clicks")
    buys = ev[ev.type == "purchase"].groupby("variant").size().rename("purchases")
    rev = ev[ev.type == "purchase"].groupby("variant").price.sum().rename("revenue")
    df = pd.concat([imp, shown, clicks, buys, rev], axis=1).fillna(0)
    df["ctr"] = df.clicks / df.items_shown
    df["purchase_rate"] = df.purchases / df.items_shown
    df["revenue_per_visit"] = df.revenue / df.impressions
    return df


def decide(df, config):
    control, alpha = config["control"], config["alpha"]
    others = [v for v in df.index if v != control]
    adj = alpha / max(len(others), 1)  # Bonferroni
    rows = []
    for v in others:
        z, p = z_test(df.loc[control, "clicks"], df.loc[control, "items_shown"], df.loc[v, "clicks"], df.loc[v, "items_shown"])
        enough = df.loc[v, "impressions"] >= config["min_impressions"]
        rows.append({"variant": v, "z": z, "p_value": p, "significant": bool(p < adj and z > 0 and enough)})
    stats = pd.DataFrame(rows).set_index("variant")
    winners = stats[stats.significant].index.tolist()
    winner = max(winners, key=lambda v: df.loc[v, "ctr"]) if winners else control
    return stats, winner, adj


def main():
    config = load_config()
    df = summarize(load_events())
    stats, winner, adj = decide(df, config)
    Path("reports").mkdir(exist_ok=True)

    ax = df.ctr.plot.bar(title="CTR par variante", color="steelblue")
    ax.set_ylabel("CTR")
    plt.tight_layout()
    plt.savefig("reports/ctr.png", dpi=120)

    md = ["## Rapport du test A/B", "", df.round(4).to_markdown(), "", f"Seuil (Bonferroni): p < {adj:.4f}", "",
          stats.round(4).to_markdown(), "", f"**Gagnant : `{winner}`** (contrôle : `{config['control']}`)", "",
          "![CTR](ctr.png)"]
    Path("reports/ab_report.md").write_text("\n".join(md), encoding="utf-8")
    Path("reports/ab_result.json").write_text(json.dumps({"winner": winner, "control": config["control"]}))
    print("Gagnant:", winner)


if __name__ == "__main__":
    main()
