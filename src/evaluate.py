"""Evalúa políticas de control en todos los escenarios con las mismas semillas.

Políticas disponibles:
- fijo:       programa de tiempo fijo generado por netconvert (ciclo de 90 s)
- actuado:    programa actuado de SUMO (alarga el verde si los detectores ven vehículos)
- qlearning:  agente tabular entrenado (models/qlearning.pkl)
- dqn, ppo:   agentes de Stable-Baselines3 (models/<nombre>.zip), si existen

Uso:
  python src/evaluate.py                         # todas las políticas disponibles
  python src/evaluate.py --policies fijo actuado # solo baselines
  python src/evaluate.py --gui --policies qlearning --scenarios pico --seeds 101
"""
import argparse
import time

import pandas as pd

import config
from envs import make_env, tripinfo_path
from metrics import episode_metrics


def load_policy(name: str):
    """Devuelve (kwargs para make_env, función obs -> acción)."""
    if name == "fijo":
        return {"tls": "static", "fixed_ts": True}, lambda obs: 0
    if name == "actuado":
        return {"tls": "actuated", "fixed_ts": True}, lambda obs: 0
    if name == "qlearning":
        from agents.qlearning import QLearningAgent

        agent = QLearningAgent.load(config.MODELS_DIR / "qlearning.pkl")
        return {"observation": "discrete"}, lambda obs: agent.act(obs, greedy=True)
    if name.startswith("dqn") or name.startswith("ppo"):
        from stable_baselines3 import DQN, PPO

        algo = DQN if name.startswith("dqn") else PPO
        model = algo.load(config.MODELS_DIR / f"{name}.zip", device="cpu")
        return {}, lambda obs: int(model.predict(obs, deterministic=True)[0])
    raise ValueError(f"Política desconocida: {name}")


def run_episode(policy: str, scenario: str, seed: int, gui: bool = False) -> dict:
    env_kwargs, act = load_policy(policy)
    trip = tripinfo_path(f"{policy}_{scenario}_{seed}")
    env = make_env(scenario, seed=seed, gui=gui, tripinfo=trip, **env_kwargs)
    obs, info = env.reset()
    infos, done = [info], False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        infos.append(info)
        done = terminated or truncated
    env.close()  # cierra SUMO y termina de escribir el tripinfo
    return {"politica": policy, "escenario": scenario, "semilla": seed, **episode_metrics(trip, infos)}


def available_policies() -> list:
    pols = ["fijo", "actuado"]
    if (config.MODELS_DIR / "qlearning.pkl").exists():
        pols.append("qlearning")
    pols += sorted(p.stem for p in config.MODELS_DIR.glob("*.zip"))
    return pols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=config.SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.EVAL_SEEDS)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", default="evaluacion.csv")
    args = parser.parse_args()

    policies = args.policies or available_policies()
    rows = []
    for policy in policies:
        for scenario in args.scenarios:
            for seed in args.seeds:
                t0 = time.time()
                rows.append(run_episode(policy, scenario, seed, gui=args.gui))
                r = rows[-1]
                print(f"{policy:>10} | {scenario:>8} | semilla {seed} | espera {r['espera_promedio_s']:6.1f} s "
                      f"| cola {r['cola_promedio_veh']:5.1f} | {time.time() - t0:4.1f} s")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    cols = ["espera_promedio_s", "tiempo_viaje_promedio_s", "cola_promedio_veh", "vehiculos_completados"]
    summary = df.groupby(["escenario", "politica"])[cols].agg(["mean", "std"]).round(1)
    print("\nResumen (promedio y desviación estándar sobre semillas):")
    print(summary.to_string())
    print(f"\nResultados por episodio guardados en {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
