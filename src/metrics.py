"""Métricas de tráfico de un episodio.

Se combinan dos fuentes:
- tripinfo de SUMO (un registro por vehículo): espera, tiempo de viaje y pérdida de tiempo por vehículo.
- info de sumo-rl en cada paso: vehículos detenidos en la red (cola).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def episode_metrics(tripinfo: Path, step_infos: list) -> dict:
    trips = ET.parse(tripinfo).getroot().findall("tripinfo")
    finished = [t for t in trips if float(t.get("arrival", -1)) >= 0]
    wait = np.array([float(t.get("waitingTime")) for t in trips])
    dur = np.array([float(t.get("duration")) for t in finished])
    loss = np.array([float(t.get("timeLoss")) for t in trips])
    stopped = np.array([i["system_total_stopped"] for i in step_infos])
    return {
        "espera_promedio_s": float(wait.mean()) if len(wait) else 0.0,
        "tiempo_viaje_promedio_s": float(dur.mean()) if len(dur) else 0.0,
        "perdida_tiempo_promedio_s": float(loss.mean()) if len(loss) else 0.0,
        "vehiculos_completados": len(finished),
        "vehiculos_sin_terminar": len(trips) - len(finished),
        "cola_promedio_veh": float(stopped.mean()),
        "cola_maxima_veh": int(stopped.max()),
    }
