"""Configuración central del proyecto.

Importar este módulo ANTES que sumo_rl: define SUMO_HOME a partir del paquete
`eclipse-sumo` instalado con pip, que sumo-rl necesita para encontrar SUMO.
"""
import os
from pathlib import Path

try:
    import sumo  # paquete eclipse-sumo

    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
except ImportError:  # SUMO instalado manualmente: SUMO_HOME debe venir del sistema
    pass

ROOT = Path(__file__).resolve().parents[1]
NETS_DIR = ROOT / "nets"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# --- Parámetros de simulación (compartidos por baselines y agentes) ---
NUM_SECONDS = 3600  # duración de un episodio: 1 hora simulada
DELTA_TIME = 5      # segundos simulados entre decisiones del agente
YELLOW_TIME = 3     # amarillo entre fases (s)
MIN_GREEN = 10      # verde mínimo por seguridad (s)
MAX_GREEN = 60      # verde máximo (s)

# Semillas fijas para evaluación: todas las políticas ven exactamente el mismo tráfico.
EVAL_SEEDS = [101, 202, 303, 404, 505]

SCENARIOS = ["valle", "pico", "variable"]


def bin_path(tool: str) -> str:
    """Ruta a un binario de SUMO (netconvert, sumo, sumo-gui)."""
    exe = tool + (".exe" if os.name == "nt" else "")
    return str(Path(os.environ["SUMO_HOME"]) / "bin" / exe)
