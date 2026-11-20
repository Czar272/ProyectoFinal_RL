"""Fase 5 (versión B) - Redes de las intersecciones reales construidas con datos medidos en OSM.

La importación automática de OpenStreetMap (src/import_osm.py) producía bloqueos permanentes:
los cruces de vías con camellón quedaban como varios nodos con tramos de 12-20 m entre ellos, y los
vehículos se trababan dentro del cruce. Aquí cada cruce se construye como UN nodo con los datos reales
de OSM (carriles, sentidos y velocidades), lo que equivale a tratar las dos mitades del cruce como un
único semáforo coordinado.

Datos de OpenStreetMap (descargados en nets/<red>/<red>.osm.xml):
- Av. Petapa y 53 calle (zona 12): ambas vías divididas por camellón. Petapa: 2 carriles por sentido,
  70 km/h. 53 calle: 2 carriles por sentido, 50 km/h.
- Av. La Reforma y 12 calle (zona 10): Reforma corre de norte a sur con 3 carriles centrales (60 km/h)
  y 2 auxiliares (40 km/h) por sentido; aquí se modelan como 5 carriles a 50 km/h. La 12 calle es de un
  solo sentido (hacia el este), con 3 carriles.

Uso:  python src/build_real.py
"""
import subprocess

import config

ARM = 250.0  # largo de cada acceso (m)

# (nodo origen, nodo destino, carriles, velocidad m/s, nombre de la calle)
INTERSECTIONS = {
    "petapa_53": [
        ("N", "C", 2, 19.44, "Avenida Petapa"), ("C", "N", 2, 19.44, "Avenida Petapa"),
        ("S", "C", 2, 19.44, "Avenida Petapa"), ("C", "S", 2, 19.44, "Avenida Petapa"),
        ("E", "C", 2, 13.89, "53 Calle"), ("C", "E", 2, 13.89, "53 Calle"),
        ("W", "C", 2, 13.89, "53 Calle"), ("C", "W", 2, 13.89, "53 Calle"),
    ],
    "reforma_12": [
        ("N", "C", 5, 13.89, "Avenida Reforma"), ("C", "N", 5, 13.89, "Avenida Reforma"),
        ("S", "C", 5, 13.89, "Avenida Reforma"), ("C", "S", 5, 13.89, "Avenida Reforma"),
        # 12 calle: un solo sentido, de oeste a este
        ("W", "C", 3, 13.89, "12 Calle"), ("C", "E", 3, 13.89, "12 Calle"),
    ],
}
COORDS = {"C": (0, 0), "N": (0, ARM), "S": (0, -ARM), "E": (ARM, 0), "W": (-ARM, 0)}


def write_inputs(name: str, edges) -> None:
    folder = config.NETS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    used = {n for e in edges for n in e[:2]}
    nodes = ["<nodes>"]
    for n in sorted(used):
        x, y = COORDS[n]
        kind = "traffic_light" if n == "C" else "priority"
        nodes.append(f'    <node id="{n}" x="{x}" y="{y}" type="{kind}"/>')
    nodes.append("</nodes>")
    (folder / f"{name}.nod.xml").write_text("\n".join(nodes) + "\n", encoding="utf-8")
    rows = ["<edges>"]
    for a, b, lanes, speed, street in edges:
        rows.append(f'    <edge id="{a}2{b}" from="{a}" to="{b}" numLanes="{lanes}" speed="{speed}" name="{street}"/>')
    rows.append("</edges>")
    (folder / f"{name}.edg.xml").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(name: str, tls_type: str) -> None:
    folder = config.NETS_DIR / name
    out = folder / f"{name}_{tls_type}.net.xml"
    subprocess.run([
        config.bin_path("netconvert"),
        "--node-files", str(folder / f"{name}.nod.xml"),
        "--edge-files", str(folder / f"{name}.edg.xml"),
        "--output-file", str(out),
        "--tls.default-type", tls_type,
        "--tls.yellow.time", str(config.YELLOW_TIME),
        "--no-turnarounds", "true",
        "--output.street-names", "true",
    ], check=True)
    print(f"Red generada: {out.relative_to(config.ROOT)}")


def main() -> None:
    for name, edges in INTERSECTIONS.items():
        write_inputs(name, edges)
        build(name, "static")
        build(name, "actuated")


if __name__ == "__main__":
    main()
