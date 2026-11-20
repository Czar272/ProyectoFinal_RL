"""Construcción del entorno de RL sobre SUMO (vía sumo-rl) y observaciones propias.

El MDP queda definido así:
- Estado:     ver DiscreteQueueObservation (tabular) o la observación por defecto de sumo-rl
              (fase one-hot, bandera de verde mínimo, densidad y cola por carril) para DQN/PPO.
- Acción:     índice de la próxima fase verde.
- Recompensa: por defecto 'diff-waiting-time' = reducción del tiempo de espera acumulado
              entre dos decisiones (positiva si la espera total bajó).
- Transición: la simula SUMO.
"""
import tempfile
from pathlib import Path

import numpy as np
from gymnasium import spaces

import config  # noqa: F401  (define SUMO_HOME antes de importar sumo_rl)
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction

QUEUE_BINS = (3, 8)  # colas por acceso: [0-2] bajo, [3-7] medio, [8+] alto


class DiscreteQueueObservation(ObservationFunction):
    """Estado discreto y pequeño para Q-learning tabular.

    (fase verde actual, puede_cambiar, cola_N, cola_S, cola_E, cola_W)
    donde cada cola es el número de vehículos detenidos en el acceso, discretizado en 3 niveles.
    Con 4 fases: 4 x 2 x 3^4 = 648 estados posibles.
    """

    def __call__(self) -> np.ndarray:
        ts = self.ts
        can_change = int(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)
        halted = {}
        for lane in ts.lanes:
            edge = lane.rsplit("_", 1)[0]
            halted[edge] = halted.get(edge, 0) + ts.sumo.lane.getLastStepHaltingNumber(lane)
        queues = [int(np.digitize(halted[e], QUEUE_BINS)) for e in sorted(halted)]
        return np.array([ts.green_phase, can_change, *queues], dtype=np.int64)

    def observation_space(self) -> spaces.Space:
        n_edges = len({lane.rsplit("_", 1)[0] for lane in self.ts.lanes})
        return spaces.MultiDiscrete([self.ts.num_green_phases, 2] + [len(QUEUE_BINS) + 1] * n_edges)


def net_file(tls: str = "static", net: str = "single") -> str:
    """Red de SUMO. net='single' es el cruce sintético; si no, una intersección real (nets/<net>/)."""
    prefix = "cruce" if net == "single" else net
    return str(config.NETS_DIR / net / f"{prefix}_{tls}.net.xml")


def route_file(scenario: str, net: str = "single") -> str:
    return str(config.NETS_DIR / net / f"{scenario}.rou.xml")


def tripinfo_path(tag: str) -> Path:
    # sumo-rl separa additional_sumo_cmd por espacios, y la carpeta del proyecto tiene espacios:
    # por eso el tripinfo se escribe en la carpeta temporal del sistema.
    return Path(tempfile.gettempdir()) / f"tripinfo_{tag}.xml"


def make_env(
    scenario: str,
    net: str = "single",
    tls: str = "static",
    observation: str = "default",
    reward_fn: str = "diff-waiting-time",
    seed="random",
    fixed_ts: bool = False,
    gui: bool = False,
    tripinfo: Path | None = None,
) -> SumoEnvironment:
    kwargs = {}
    if observation == "discrete":
        kwargs["observation_class"] = DiscreteQueueObservation
    extra = ["--no-step-log", "true"]
    if tripinfo is not None:
        extra += ["--tripinfo-output", str(tripinfo), "--tripinfo-output.write-unfinished", "true"]
    return SumoEnvironment(
        net_file=net_file(tls, net),
        route_file=route_file(scenario, net),
        single_agent=True,
        use_gui=gui,
        num_seconds=config.NUM_SECONDS,
        delta_time=config.DELTA_TIME,
        yellow_time=config.YELLOW_TIME,
        min_green=config.MIN_GREEN,
        max_green=config.MAX_GREEN,
        reward_fn=reward_fn,
        sumo_seed=seed,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=" ".join(extra),
        **kwargs,
    )
