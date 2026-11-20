"""Evalúa políticas de control en todos los escenarios con las mismas semillas.

Políticas disponibles:
- fijo:       programa de tiempo fijo generado por netconvert (ciclo de 90 s)
- actuado:    programa actuado de SUMO (alarga el verde si los detectores ven vehículos)
- fijo_corto: tiempo fijo de 30 s, solo fases rectas (replica la secuencia aprendida por los agentes)
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


def models_dir(net: str = "single"):
    """Los modelos del cruce sintético viven en models/; los de cada red real, en models/<red>/."""
    return config.MODELS_DIR if net == "single" else config.MODELS_DIR / net


def load_policy(name: str, net: str = "single"):
    """Devuelve (kwargs para make_env, función obs -> acción)."""
    if name == "fijo":
        return {"tls": "static", "fixed_ts": True}, lambda obs: 0
    if name == "actuado":
        return {"tls": "actuated", "fixed_ts": True}, lambda obs: 0
    if name == "fijo_corto":
        return {"tls": "corto", "fixed_ts": True}, lambda obs: 0
    if name == "fijo_opt":
        return {"tls": "opt", "fixed_ts": True}, lambda obs: 0
    if name == "qlearning":
        from agents.qlearning import QLearningAgent

        agent = QLearningAgent.load(models_dir(net) / "qlearning.pkl")
        return {"observation": "discrete"}, lambda obs: agent.act(obs, greedy=True)
    if name.startswith("dqn") or name.startswith("ppo"):
        from stable_baselines3 import DQN, PPO

        algo = DQN if name.startswith("dqn") else PPO
        model = algo.load(models_dir(net) / f"{name}.zip", device="cpu")
        return {}, lambda obs: int(model.predict(obs, deterministic=True)[0])
    raise ValueError(f"Política desconocida: {name}")


def run_episode(policy: str, scenario: str, seed: int, gui: bool = False, net: str = "single") -> dict:
    env_kwargs, act = load_policy(policy, net)
    trip = tripinfo_path(f"{net}_{policy}_{scenario}_{seed}")
    env = make_env(scenario, net=net, seed=seed, gui=gui, tripinfo=trip, **env_kwargs)
    obs, info = env.reset()
    infos, done = [info], False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        infos.append(info)
        done = terminated or truncated
    env.close()  # cierra SUMO y termina de escribir el tripinfo
    return {"red": net, "politica": policy, "escenario": scenario, "semilla": seed, **episode_metrics(trip, infos)}


def available_policies(net: str = "single") -> list:
    pols = ["fijo", "actuado"]
    from envs import net_file
    from pathlib import Path

    pols += [p for p, tls in (("fijo_corto", "corto"), ("fijo_opt", "opt")) if Path(net_file(tls, net)).exists()]
    if (models_dir(net) / "qlearning.pkl").exists():
        pols.append("qlearning")
    pols += sorted(p.stem for p in models_dir(net).glob("*.zip"))
    return pols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="single", help="single (cruce sintético) o una red real: petapa_53, reforma_12")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=config.SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.EVAL_SEEDS)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", default="evaluacion.csv")
    args = parser.parse_args()

    policies = args.policies or available_policies(args.net)
    rows = []
    for policy in policies:
        for scenario in args.scenarios:
            for seed in args.seeds:
                t0 = time.time()
                rows.append(run_episode(policy, scenario, seed, gui=args.gui, net=args.net))
                r = rows[-1]
                print(f"{policy:>10} | {scenario:>8} | semilla {seed} | espera {r['espera_total_promedio_s']:6.1f} s "
                      f"| cola {r['cola_promedio_veh']:5.1f} | {time.time() - t0:4.1f} s")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    cols = ["espera_total_promedio_s", "espera_promedio_s", "tiempo_viaje_promedio_s", "cola_promedio_veh", "vehiculos_completados"]
    summary = df.groupby(["escenario", "politica"])[cols].agg(["mean", "std"]).round(1)
    print("\nResumen (promedio y desviación estándar sobre semillas):")
    print(summary.to_string())
    print(f"\nResultados por episodio guardados en {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
