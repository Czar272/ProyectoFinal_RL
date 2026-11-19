"""Fase 2 - Agente Q-learning tabular (visto en clase).

Actualización (off-policy, TD(0)):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Exploración epsilon-greedy con epsilon decreciente por episodio.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.93, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(self.n_actions))

    @staticmethod
    def key(obs) -> tuple:
        return tuple(int(x) for x in obs)

    def act(self, obs, greedy=False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q[self.key(obs)]
        best = np.flatnonzero(q == q.max())  # desempate aleatorio entre acciones iguales
        return int(self.rng.choice(best))

    def update(self, obs, action, reward, next_obs, terminal=False) -> float:
        s, s2 = self.key(obs), self.key(next_obs)
        target = reward if terminal else reward + self.gamma * self.q[s2].max()
        td_error = target - self.q[s][action]
        self.q[s][action] += self.alpha * td_error
        return td_error

    def end_episode(self) -> None:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"n_actions": self.n_actions, "q": {k: v.tolist() for k, v in self.q.items()}}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(data["n_actions"], eps_start=0.0, eps_end=0.0)
        for k, v in data["q"].items():
            agent.q[k] = np.array(v)
        return agent
