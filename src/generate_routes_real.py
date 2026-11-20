"""Fase 5 - Demanda estimada para las intersecciones reales.

No hay conteos vehiculares públicos de estos cruces, así que la demanda se ESTIMA por tipo de calle:
- Avenida principal (Petapa, Reforma y sus carriles auxiliares): más vehículos por carril.
- Calle secundaria: menos vehículos por carril.
Cada entrada genera (vehículos/hora por carril) x (número de carriles). Los destinos se reparten
entre las salidas alcanzables, con peso 3 para seguir por la misma calle y 1 para girar.

Esto produce escenarios plausibles ("qué pasaría si"), no una réplica del tráfico real; así se
reporta en el documento.

Uso:  python src/generate_routes_real.py petapa_53 reforma_12
"""
import sys

import sumolib

import config
from envs import net_file

ARTERIAL_KEYWORDS = ("Petapa", "Reforma")

# veh/h por carril de entrada: (avenida principal, calle secundaria), por periodo
SCENARIOS = {
    "valle": [(0, 3600, 200, 150)],
    "pico": [(0, 3600, 500, 250)],
    # La dominancia cambia: primero cargan las secundarias, al final la avenida
    "variable": [(0, 1200, 250, 350), (1200, 2400, 400, 250), (2400, 3600, 550, 150)],
}


def is_arterial(edge) -> bool:
    return any(k in (edge.getName() or "") for k in ARTERIAL_KEYWORDS)


def od_pairs(net):
    ok = lambda e: e.allows("passenger") and not e.isSpecial()  # noqa: E731
    origins = [e for e in net.getEdges() if ok(e) and not [x for x in e.getIncoming() if ok(x)]]
    dests = [e for e in net.getEdges() if ok(e) and not [x for x in e.getOutgoing() if ok(x)]]
    pairs = {}
    for o in origins:
        routes = []
        for d in dests:
            path, _ = net.getShortestPath(o, d, vClass="passenger")
            if path:
                weight = 3.0 if (o.getName() or "") == (d.getName() or "") else 1.0
                routes.append((d, path, weight))
        if routes:
            pairs[o] = routes
    return pairs


def routes_xml(pairs, periods) -> str:
    lines = ["<routes>",
             '    <vType id="auto" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5"/>']
    for o, routes in pairs.items():
        for d, path, _ in routes:
            edges = " ".join(e.getID() for e in path)
            lines.append(f'    <route id="r_{o.getID()}__{d.getID()}" edges="{edges}"/>')
    for k, (begin, end, arterial_vph, secondary_vph) in enumerate(periods):
        for o, routes in pairs.items():
            vph = o.getLaneNumber() * (arterial_vph if is_arterial(o) else secondary_vph)
            total_w = sum(w for _, _, w in routes)
            for d, _, w in routes:
                prob = vph * (w / total_w) / 3600.0
                lines.append(
                    f'    <flow id="f{k}_{o.getID()}__{d.getID()}" type="auto" route="r_{o.getID()}__{d.getID()}" '
                    f'begin="{begin}" end="{end}" probability="{prob:.5f}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def main() -> None:
    for name in sys.argv[1:] or ["petapa_53", "reforma_12"]:
        net = sumolib.net.readNet(net_file("static", name))
        pairs = od_pairs(net)
        print(f"== {name}: {len(pairs)} entradas, {sum(len(r) for r in pairs.values())} pares origen-destino")
        for scenario, periods in SCENARIOS.items():
            path = config.NETS_DIR / name / f"{scenario}.rou.xml"
            path.write_text(routes_xml(pairs, periods), encoding="utf-8")
            total = sum(
                o.getLaneNumber() * (a if is_arterial(o) else s) * (e - b) / 3600
                for b, e, a, s in periods for o in pairs)
            print(f"  {path.relative_to(config.ROOT)}: ~{total:.0f} vehículos en la hora")


if __name__ == "__main__":
    main()
