"""Gráficas para el documento y la presentación.

- results/figuras/comparacion_espera.png : espera promedio por política y escenario (barras + desviación)
- results/figuras/curva_aprendizaje.png  : espera media por episodio durante el entrenamiento

Uso:  python src/plots.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

FIG_DIR = config.RESULTS_DIR / "figuras"

# Baselines en grises (contexto); agentes en los tres primeros colores categóricos, en orden fijo.
COLORS = {
    "fijo": "#b4b3ab",
    "actuado": "#6f6e68",
    "qlearning": "#2a78d6",
    "dqn": "#eb6834",
    "ppo": "#1baf7a",
}
LABELS = {"fijo": "Tiempo fijo", "actuado": "Actuado", "qlearning": "Q-learning", "dqn": "DQN", "ppo": "PPO"}
SCEN_LABELS = {"valle": "Valle", "pico": "Pico", "variable": "Variable"}
INK, INK2, GRID = "#2b2b29", "#6f6e68", "#e6e5e0"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_eval() -> pd.DataFrame:
    files = sorted(config.RESULTS_DIR.glob("evaluacion*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])
    if "red" in df:  # solo el cruce sintético (los CSV antiguos no tienen la columna 'red')
        df = df[df.red.fillna("single") == "single"]
    df = df[df.escenario.isin(SCEN_LABELS)].drop_duplicates(["politica", "escenario", "semilla"], keep="last")
    return df[df.politica.isin(COLORS)]


def plot_comparison(df: pd.DataFrame) -> None:
    stats = df.groupby(["escenario", "politica"]).espera_promedio_s.agg(["mean", "std"])
    policies = [p for p in COLORS if p in df.politica.unique()]
    scenarios = [s for s in SCEN_LABELS if s in df.escenario.unique()]
    width = 0.8 / len(policies)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for i, pol in enumerate(policies):
        xs = [j + (i - (len(policies) - 1) / 2) * width for j in range(len(scenarios))]
        means = [stats.loc[(s, pol), "mean"] for s in scenarios]
        stds = [stats.loc[(s, pol), "std"] for s in scenarios]
        ax.bar(xs, means, width * 0.9, color=COLORS[pol], label=LABELS[pol],
               yerr=stds, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        for x, m, sd in zip(xs, means, stds):
            ax.text(x, m + sd + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(range(len(scenarios)), [SCEN_LABELS[s] for s in scenarios])
    ax.set_ylabel("Espera promedio por vehículo (s)", color=INK)
    ax.set_title("Q-learning y PPO reducen la espera a la mitad del actuado; DQN es inestable en hora pico",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, ncol=len(policies), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_espera.png")
    plt.close(fig)


def load_curves() -> dict:
    curves = {}
    q = config.RESULTS_DIR / "entrenamiento_qlearning.csv"
    if q.exists():
        d = pd.read_csv(q)
        curves["qlearning"] = d.espera_media_s.reset_index(drop=True)
    for algo in ("dqn", "ppo"):
        f = config.RESULTS_DIR / f"entrenamiento_{algo}.monitor.csv"
        if f.exists():
            d = pd.read_csv(f, skiprows=1)  # la primera línea del Monitor es un comentario JSON
            if len(d):
                curves[algo] = d.espera_media_s.reset_index(drop=True)
    return curves


def plot_learning(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ends = []
    for algo, s in curves.items():
        ep = range(1, len(s) + 1)
        ax.plot(ep, s, color=COLORS[algo], alpha=0.25, linewidth=1)
        smooth = s.rolling(5, min_periods=1).mean()
        ax.plot(ep, smooth, color=COLORS[algo], linewidth=2, label=LABELS[algo])
        ends.append([smooth.iloc[-1], len(s), algo])
    # Etiquetas directas al final de cada línea, separadas para que no se encimen
    ymax = max(c.max() for c in curves.values())
    gap = 0.045 * ymax
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, algo in ends:
        ax.annotate(LABELS[algo], (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("Episodio (1 hora simulada, escenario variable)", color=INK)
    ax.set_ylabel("Espera media en la red (s)", color=INK)
    ax.set_title("Curva de aprendizaje (línea gruesa: media móvil de 5 episodios)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curva_aprendizaje.png")
    plt.close(fig)


def plot_saturated() -> None:
    """Escenario saturado: espera (detenido) y pérdida de tiempo por política, en dos paneles."""
    files = sorted(config.RESULTS_DIR.glob("evaluacion_saturado*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    order = ["fijo", "fijo_opt", "fijo_corto", "actuado", "qlearning_saturado", "ppo_saturado"]
    names = {"fijo": "Tiempo fijo (90 s)", "fijo_opt": "Fijo optimizado", "fijo_corto": "Fijo corto (30 s)",
             "actuado": "Actuado", "qlearning_saturado": "Q-learning", "ppo_saturado": "PPO"}
    colors = {"fijo": COLORS["fijo"], "fijo_opt": COLORS["fijo"], "fijo_corto": COLORS["fijo"],
              "actuado": COLORS["actuado"], "qlearning_saturado": COLORS["qlearning"], "ppo_saturado": COLORS["ppo"]}
    order = [p for p in order if p in df.politica.unique()]
    stats = df.groupby("politica")[["espera_total_promedio_s", "perdida_tiempo_promedio_s"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    panels = [("espera_total_promedio_s", "Tiempo detenido por vehículo (s)"),
              ("perdida_tiempo_promedio_s", "Pérdida de tiempo por vehículo (s)")]
    ys = range(len(order))
    for ax, (col, label) in zip(axes, panels):
        vals = [stats.loc[p, col] for p in order]
        ax.barh(list(ys), vals, color=[colors[p] for p in order], height=0.65)
        for y, v in zip(ys, vals):
            ax.text(v + 1.5, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.set_xlabel(label, color=INK)
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ys), [names[p] for p in order])
    axes[0].invert_yaxis()
    fig.suptitle("Escenario saturado: PPO detiene menos a los vehículos que el actuado, con la misma pérdida de tiempo",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_saturado.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(load_eval())
    plot_saturated()
    curves = load_curves()
    if curves:
        plot_learning(curves)
    print(f"Figuras en {FIG_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
