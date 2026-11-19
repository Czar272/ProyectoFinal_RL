"""Fases 3 y 4 - Entrena DQN o PPO (Stable-Baselines3) con observación continua.

Observación por defecto de sumo-rl: fase actual (one-hot), bandera de verde mínimo,
densidad y cola normalizadas por carril (vector de 4 + 1 + 12 + 12 = 29 valores).

- DQN: valor-acción con red neuronal, experience replay y red objetivo (extiende Q-learning).
- PPO: gradiente de política actor-crítico con objetivo recortado (clipped).

Uso:
  python src/train_sb3.py --algo dqn --episodes 100
  python src/train_sb3.py --algo ppo --episodes 100
  python src/train_sb3.py --algo dqn --reward queue --name dqn_queue   # comparar recompensas
Salida: models/<name>.zip, results/entrenamiento_<name>.monitor.csv, results/tensorboard/
"""
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import config
from envs import make_env


class WaitingTimeInfo(gym.Wrapper):
    """Agrega al final del episodio la espera media de la red, para registrarla en el Monitor."""

    def reset(self, **kwargs):
        self._waits = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._waits.append(info["system_mean_waiting_time"])
        info["espera_media_s"] = float(np.mean(self._waits))
        return obs, reward, terminated, truncated, info


def build_model(algo: str, env, seed: int):
    common = {"verbose": 0, "seed": seed, "device": "cpu",
              "tensorboard_log": str(config.RESULTS_DIR / "tensorboard")}
    if algo == "dqn":
        return DQN(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, batch_size=64,
            gamma=0.99, train_freq=1, target_update_interval=500,
            exploration_fraction=0.3, exploration_final_eps=0.02,
            policy_kwargs={"net_arch": [64, 64]}, **common,
        )
    return PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]}, **common,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", default="variable")
    parser.add_argument("--reward", default="diff-waiting-time",
                        choices=["diff-waiting-time", "queue", "pressure", "average-speed"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    name = args.name or args.algo

    config.RESULTS_DIR.mkdir(exist_ok=True)
    env = make_env(args.scenario, reward_fn=args.reward, seed="random")
    env = Monitor(WaitingTimeInfo(env), str(config.RESULTS_DIR / f"entrenamiento_{name}"),
                  info_keywords=("espera_media_s",))

    steps_per_episode = config.NUM_SECONDS // config.DELTA_TIME
    # Checkpoint cada 10 episodios: si el proceso se interrumpe, no se pierde todo el entrenamiento.
    checkpoints = CheckpointCallback(save_freq=10 * steps_per_episode,
                                     save_path=str(config.MODELS_DIR / "checkpoints"), name_prefix=name)
    model = build_model(args.algo, env, args.seed)
    model.learn(total_timesteps=args.episodes * steps_per_episode, tb_log_name=name,
                callback=checkpoints, progress_bar=False)
    model.save(config.MODELS_DIR / f"{name}.zip")
    env.close()
    print(f"Modelo: models/{name}.zip | Curva: results/entrenamiento_{name}.monitor.csv")


if __name__ == "__main__":
    main()
