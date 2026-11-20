"""Fase 1 - Genera la demanda de tráfico (archivos .rou.xml) por escenario.

Cada acceso genera vehículos con una probabilidad por segundo (llegadas
aleatorias tipo Poisson). La semilla de SUMO decide la realización concreta,
así que un mismo archivo produce tráfico distinto con cada semilla.

Escenarios (vehículos/hora por acceso):
- valle:    tráfico bajo y balanceado
- pico:     hora pico con una arteria Norte-Sur dominante (como la Roosevelt)
- variable: la demanda cambia durante la hora (E-O cargado -> balanceado -> N-S cargado);
            es el escenario de entrenamiento porque obliga a adaptarse.

Uso:  python src/generate_routes.py
"""
import config

OUT = config.NETS_DIR / "single"

# Giros desde cada acceso (manejo por la derecha). Ej.: quien viene del Norte va hacia el Sur.
TURNS = {
    "N": {"recto": "S", "derecha": "W", "izquierda": "E"},
    "S": {"recto": "N", "derecha": "E", "izquierda": "W"},
    "E": {"recto": "W", "derecha": "N", "izquierda": "S"},
    "W": {"recto": "E", "derecha": "S", "izquierda": "N"},
}
TURN_RATIO = {"recto": 0.70, "derecha": 0.15, "izquierda": 0.15}

# Cada escenario es una lista de periodos: (inicio_s, fin_s, {acceso: veh/h})
SCENARIOS = {
    "valle": [(0, 3600, {"N": 300, "S": 300, "E": 250, "W": 250})],
    "pico": [(0, 3600, {"N": 800, "S": 700, "E": 300, "W": 300})],
    "variable": [
        (0, 1200, {"N": 250, "S": 250, "E": 700, "W": 650}),
        (1200, 2400, {"N": 450, "S": 450, "E": 450, "W": 450}),
        (2400, 3600, {"N": 800, "S": 750, "E": 250, "W": 250}),
    ],
    # Cerca de la capacidad y con la dirección dominante cambiando: aquí cada cambio de fase
    # (3 s de amarillo perdidos) cuesta capacidad, y ningún plan fijo es óptimo toda la hora.
    # Nivel calibrado con un barrido de demanda: a este nivel el ciclo corto de 30 s ya pierde
    # contra el actuado; con 20 % más demanda todos los controladores colapsan.
    "saturado": [
        (0, 1200, {"N": 1760, "S": 1680, "E": 640, "W": 640}),
        (1200, 2400, {"N": 1200, "S": 1200, "E": 1200, "W": 1200}),
        (2400, 3600, {"N": 640, "S": 640, "E": 1760, "W": 1680}),
    ],
}


def routes_xml(periods) -> str:
    lines = [
        "<routes>",
        '    <vType id="auto" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>',
    ]
    for origin, dests in TURNS.items():
        for dest in dests.values():
            lines.append(f'    <route id="r_{origin}_{dest}" edges="{origin}2C C2{dest}"/>')
    for k, (begin, end, demand) in enumerate(periods):
        for origin, vph in demand.items():
            for turn, dest in TURNS[origin].items():
                prob = vph * TURN_RATIO[turn] / 3600.0  # probabilidad de salida por segundo
                lines.append(
                    f'    <flow id="f{k}_{origin}_{dest}" type="auto" route="r_{origin}_{dest}" '
                    f'begin="{begin}" end="{end}" probability="{prob:.5f}" '
                    f'departLane="best" departSpeed="max"/>'
                )
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, periods in SCENARIOS.items():
        path = OUT / f"{name}.rou.xml"
        path.write_text(routes_xml(periods), encoding="utf-8")
        total = sum(sum(d.values()) * (e - b) / 3600 for b, e, d in periods)
        print(f"{path.relative_to(config.ROOT)}: ~{total:.0f} vehículos esperados en la hora")


if __name__ == "__main__":
    main()
