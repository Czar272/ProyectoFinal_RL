"""Baseline fuerte: el mejor plan de tiempo fijo para un escenario, buscado por fuerza bruta.

Un agente de RL solo demuestra valor si le gana a la mejor alternativa sencilla, no a una mal
configurada. Este script prueba combinaciones de verde para cada fase y se queda con la de menor
espera total. Se ajusta con semillas DISTINTAS a las de evaluación (11 y 12), para no hacer trampa.

Uso:  python src/tune_fixed.py --scenario saturado
Salida: nets/single/cruce_opt.net.xml (plan ganador), results/ajuste_fijo_<escenario>.csv
"""
import argparse
import itertools
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import config

NET_DIR = config.NETS_DIR / "single"
TUNING_SEEDS = [11, 12]
GREEN_OPTIONS = [15, 25, 35, 50]  # verde de las fases rectas (s)
LEFT_OPTIONS = [0, 8]             # verde del giro izquierdo protegido (0 = sin fase protegida)


def build_plan(g_ns: int, g_ew: int, left: int, out: Path) -> None:
    tree = ET.parse(NET_DIR / "cruce_static.net.xml")
    tl = tree.getroot().find("tlLogic")
    phases = tl.findall("phase")
    # Orden de netconvert: 0 recto N-S, 1 amarillo, 2 izq N-S, 3 amarillo, 4 recto E-O, 5, 6 izq E-O, 7
    plan = [(0, g_ns), (1, 3)]
    if left:
        plan += [(2, left), (3, 3)]
    plan += [(4, g_ew), (5, 3)]
    if left:
        plan += [(6, left), (7, 3)]
    for p in phases:
        tl.remove(p)
    for idx, dur in plan:
        phase = phases[idx]
        phase.set("duration", str(dur))
        tl.append(phase)
    tree.write(out, encoding="utf-8", xml_declaration=True)


def total_wait(net: Path, scenario: str, seed: int) -> float:
    trip = Path(tempfile.gettempdir()) / "tripinfo_tune.xml"
    subprocess.run([
        config.bin_path("sumo"), "-n", str(net), "-r", str(NET_DIR / f"{scenario}.rou.xml"),
        "--end", str(config.NUM_SECONDS), "--time-to-teleport", "-1", "--seed", str(seed),
        "--no-step-log", "--no-warnings", "--tripinfo-output", str(trip),
        "--tripinfo-output.write-unfinished",
    ], check=True)
    trips = ET.parse(trip).getroot().findall("tripinfo")
    return float(np.mean([float(t.get("waitingTime")) + float(t.get("departDelay")) for t in trips]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="saturado")
    args = parser.parse_args()

    candidate = Path(tempfile.gettempdir()) / "cruce_candidato.net.xml"
    rows = []
    for g_ns, g_ew, left in itertools.product(GREEN_OPTIONS, GREEN_OPTIONS, LEFT_OPTIONS):
        build_plan(g_ns, g_ew, left, candidate)
        wait = np.mean([total_wait(candidate, args.scenario, s) for s in TUNING_SEEDS])
        cycle = g_ns + g_ew + 6 + (2 * left + 6 if left else 0)
        rows.append({"verde_ns": g_ns, "verde_eo": g_ew, "verde_izq": left, "ciclo_s": cycle, "espera_total_s": wait})
        print(f"N-S {g_ns:2d} s | E-O {g_ew:2d} s | izq {left} s | ciclo {cycle:3d} s | espera {wait:6.1f} s")

    df = pd.DataFrame(rows).sort_values("espera_total_s")
    df.to_csv(config.RESULTS_DIR / f"ajuste_fijo_{args.scenario}.csv", index=False)
    best = df.iloc[0]
    build_plan(int(best.verde_ns), int(best.verde_eo), int(best.verde_izq), NET_DIR / "cruce_opt.net.xml")
    print(f"\nMejor plan: N-S {best.verde_ns:.0f} s, E-O {best.verde_eo:.0f} s, izquierda {best.verde_izq:.0f} s "
          f"(ciclo {best.ciclo_s:.0f} s) -> {best.espera_total_s:.1f} s de espera. Guardado en nets/single/cruce_opt.net.xml")


if __name__ == "__main__":
    main()
