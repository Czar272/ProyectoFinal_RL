"""Fase 2 - Entrena el agente Q-learning tabular.

Se entrena en el escenario 'variable' (la demanda cambia durante la hora) con una
semilla de SUMO distinta por episodio, para que el agente no memorice un tráfico concreto.

Uso:  python src/train_qlearning.py --episodes 60
      python src/train_qlearning.py --scenario saturado --name qlearning_saturado
Salida:
  models/qlearning.pkl               tabla Q
  results/entrenamiento_qlearning.csv curva de aprendizaje por episodio
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from agents.qlearning import QLearningAgent
from envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--scenario", default="variable")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps-decay", type=float, default=0.93)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="qlearning", help="nombre del modelo en models/ (sin extensión)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = None
    agent = None
    log = []
    for ep in range(1, args.episodes + 1):
        sumo_seed = int(rng.integers(1_000_000))
        env = make_env(args.scenario, observation="discrete", seed=sumo_seed)
        obs, _ = env.reset()
        if agent is None:
            agent = QLearningAgent(env.action_space.n, alpha=args.alpha, gamma=args.gamma,
                                   eps_decay=args.eps_decay, seed=args.seed)
        t0, total_r, waits, td, done = time.time(), 0.0, [], [], False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            td.append(abs(agent.update(obs, action, reward, next_obs, terminated)))
            obs, total_r, done = next_obs, total_r + reward, terminated or truncated
            waits.append(info["system_mean_waiting_time"])
        env.close()

        log.append({
            "episodio": ep,
            "recompensa_total": total_r,
            "espera_media_s": float(np.mean(waits)),
            "epsilon": agent.epsilon,
            "td_error_medio": float(np.mean(td)),
            "estados_visitados": len(agent.q),
        })
        print(f"ep {ep:3d} | recompensa {total_r:9.1f} | espera {log[-1]['espera_media_s']:6.1f} s "
              f"| eps {agent.epsilon:.3f} | estados {len(agent.q):4d} | {time.time() - t0:4.1f} s")
        agent.end_episode()

    agent.save(config.MODELS_DIR / f"{args.name}.pkl")
    out = config.RESULTS_DIR / f"entrenamiento_{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out, index=False)
    print(f"Modelo: models/{args.name}.pkl | Curva: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
