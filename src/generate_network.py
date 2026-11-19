"""Fase 1 - Genera la red de un cruce sintético de 4 vías.

Cuatro accesos (N, S, E, O) de 300 m, 50 km/h, con un semáforo en el centro.
Cada acceso entra con 3 carriles (derecha+recto, recto, giro izquierdo exclusivo)
y cada salida tiene 2 carriles. Se generan dos versiones de la misma red:

- cruce_static.net.xml   -> semáforo de tiempo fijo (baseline 1 y red que controla el agente)
- cruce_actuated.net.xml -> semáforo actuado por detectores (baseline 2)

Uso:  python src/generate_network.py
"""
import subprocess

import config

OUT = config.NETS_DIR / "single"
ARM = 300.0       # largo de cada acceso (m)
SPEED = 13.89     # 50 km/h en m/s
LANES = 2

NODES = f"""<nodes>
    <node id="C" x="0" y="0" type="traffic_light"/>
    <node id="N" x="0" y="{ARM}" type="priority"/>
    <node id="S" x="0" y="-{ARM}" type="priority"/>
    <node id="E" x="{ARM}" y="0" type="priority"/>
    <node id="W" x="-{ARM}" y="0" type="priority"/>
</nodes>
"""


def edges_xml() -> str:
    rows = []
    for arm in ["N", "S", "E", "W"]:
        # Los accesos llevan un carril extra exclusivo para girar a la izquierda: así netconvert
        # genera fases protegidas de giro izquierdo (4 fases verdes en lugar de 2).
        rows.append(f'    <edge id="{arm}2C" from="{arm}" to="C" numLanes="{LANES + 1}" speed="{SPEED}"/>')
        rows.append(f'    <edge id="C2{arm}" from="C" to="{arm}" numLanes="{LANES}" speed="{SPEED}"/>')
    return "<edges>\n" + "\n".join(rows) + "\n</edges>\n"


def build(tls_type: str) -> None:
    out = OUT / f"cruce_{tls_type}.net.xml"
    cmd = [
        config.bin_path("netconvert"),
        "--node-files", str(OUT / "cruce.nod.xml"),
        "--edge-files", str(OUT / "cruce.edg.xml"),
        "--output-file", str(out),
        "--tls.default-type", tls_type,
        "--tls.yellow.time", str(config.YELLOW_TIME),
        "--no-turnarounds", "true",
    ]
    subprocess.run(cmd, check=True)
    print(f"Red generada: {out.relative_to(config.ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cruce.nod.xml").write_text(NODES, encoding="utf-8")
    (OUT / "cruce.edg.xml").write_text(edges_xml(), encoding="utf-8")
    build("static")
    build("actuated")


if __name__ == "__main__":
    main()
