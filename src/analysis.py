"""Análisis de comportamiento: uso de fases y equidad entre movimientos.

Responde dos preguntas del documento:
1. ¿Por qué ganan los agentes? Hipótesis: se saltan fases vacías (sobre todo los giros izquierdos
   protegidos), mientras fijo y actuado las recorren siempre en el mismo orden.
2. ¿Es justo? La espera promedio puede esconder vehículos que esperan mucho. Se reporta el
   percentil 95 y el máximo, separados por tipo de giro.

Uso:  python src/analysis.py                  # todas las políticas, 3 escenarios, semillas 101 y 202
Salida: results/analisis_fases.csv, results/analisis_equidad.csv
"""
import argparse
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

import config
from envs import make_env, net_file, tripinfo_path
from evaluate import available_policies, load_policy
from generate_routes import TURNS

PHASE_NAMES = ["Recto N-S", "Izquierda N-S", "Recto E-O", "Izquierda E-O"]


def green_states() -> list:
    """Estados (cadenas rGy) de las 4 fases verdes, leídos del programa de la red."""
    root = ET.parse(net_file("static")).getroot()
    phases = root.find("tlLogic").findall("phase")
    return [p.get("state") for p in phases if "y" not in p.get("state")]


def turn_of(vehicle_id: str) -> str:
    # id de SUMO: f<periodo>_<origen>_<destino>.<n>
    _, origin, dest = vehicle_id.split(".")[0].split("_")
    return next(t for t, d in TURNS[origin].items() if d == dest)


def run(policy: str, scenario: str, seed: int, greens: list):
    env_kwargs, act = load_policy(policy)
    trip = tripinfo_path(f"analisis_{policy}_{scenario}_{seed}")
    env = make_env(scenario, seed=seed, tripinfo=trip, **env_kwargs)
    obs, _ = env.reset()
    samples, done = [], False
    while not done:
        obs, _, terminated, truncated, _ = env.step(act(obs))
        state = env.sumo.trafficlight.getRedYellowGreenState("C")
        samples.append(greens.index(state) if state in greens else -1)  # -1 = amarillo
        done = terminated or truncated
    env.close()

    # Uso de fases: % del tiempo en cada verde y duración media de cada verde
    counts = Counter(samples)
    runs, prev = [], None
    for s in samples:
        if s >= 0 and s == prev:
            runs[-1][1] += 1
        elif s >= 0:
            runs.append([s, 1])
        prev = s
    phase_rows = []
    for i, name in enumerate(PHASE_NAMES):
        durations = [n * config.DELTA_TIME for p, n in runs if p == i]
        phase_rows.append({
            "politica": policy, "escenario": scenario, "semilla": seed, "fase": name,
            "pct_tiempo": 100 * counts[i] / len(samples),
            "veces_activada": len(durations),
            "verde_medio_s": float(np.mean(durations)) if durations else 0.0,
        })

    # Equidad: espera por vehículo según tipo de giro
    trips = ET.parse(trip).getroot().findall("tripinfo")
    df = pd.DataFrame({"giro": [turn_of(t.get("id")) for t in trips],
                       "espera": [float(t.get("waitingTime")) for t in trips]})
    fair_rows = []
    for giro, g in [("todos", df)] + list(df.groupby("giro")):
        fair_rows.append({
            "politica": policy, "escenario": scenario, "semilla": seed, "giro": giro,
            "vehiculos": len(g), "espera_media_s": g.espera.mean(),
            "espera_p95_s": g.espera.quantile(0.95), "espera_max_s": g.espera.max(),
        })
    return phase_rows, fair_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=config.SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    args = parser.parse_args()

    greens = green_states()
    phases, fairness = [], []
    for policy in args.policies or available_policies():
        for scenario in args.scenarios:
            for seed in args.seeds:
                p, f = run(policy, scenario, seed, greens)
                phases += p
                fairness += f
                print(f"{policy:>10} | {scenario:>8} | semilla {seed} | listo")

    pd.DataFrame(phases).to_csv(config.RESULTS_DIR / "analisis_fases.csv", index=False)
    pd.DataFrame(fairness).to_csv(config.RESULTS_DIR / "analisis_equidad.csv", index=False)

    ph = pd.DataFrame(phases).groupby(["escenario", "politica", "fase"])[["pct_tiempo", "verde_medio_s"]].mean()
    print("\n% del tiempo en cada fase y duración media del verde:")
    print(ph.round(1).unstack("fase").to_string())
    fa = pd.DataFrame(fairness).groupby(["escenario", "politica", "giro"])[["espera_media_s", "espera_p95_s", "espera_max_s"]].mean()
    print("\nEspera por tipo de giro (s):")
    print(fa.round(1).to_string())


if __name__ == "__main__":
    main()
