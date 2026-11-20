"""Fase 5 - Importa intersecciones reales de Ciudad de Guatemala desde OpenStreetMap a SUMO.

Para cada intersección:
1. Descarga del API de Overpass las calles en un recuadro alrededor del cruce.
2. Convierte con netconvert, uniendo los nodos de semáforo cercanos en un solo controlador
   (una avenida con camellón tiene varias cabezas de semáforo que funcionan como uno solo).
3. Genera la versión de tiempo fijo (programa por defecto) y la actuada.

Uso:  python src/import_osm.py              # todas
      python src/import_osm.py petapa_53    # una
"""
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import config

INTERSECTIONS = {
    "petapa_53": {"nombre": "Avenida Petapa y 53 calle, zona 12", "lat": 14.55780, "lon": -90.55150},
    "reforma_12": {"nombre": "Avenida La Reforma y 12 calle, zona 10", "lat": 14.60095, "lon": -90.51670},
}
HALF_SIZE_DEG = 0.0030  # descarga: ~330 m a cada lado del cruce
CROP_DEG = 0.0016       # red final: ~175 m a cada lado, para que quede un solo semáforo (el del cruce)

OVERPASS = "https://overpass-api.de/api/interpreter"


def download(name: str, lat: float, lon: float) -> None:
    out = config.NETS_DIR / name / f"{name}.osm.xml"
    if out.exists():
        print(f"{name}: OSM ya descargado")
        return
    s, w, n, e = lat - HALF_SIZE_DEG, lon - HALF_SIZE_DEG, lat + HALF_SIZE_DEG, lon + HALF_SIZE_DEG
    query = f'[out:xml][timeout:60];(way({s},{w},{n},{e})["highway"];);(._;>;);out body;'
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": "sumo-rl-course-project"})
    for attempt in range(5):  # el servidor público a veces está saturado: reintento con espera creciente
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            if body.lstrip().startswith(b"<?xml"):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(body)
                print(f"{name}: OSM descargado ({len(body) // 1024} KB)")
                return
        except OSError as exc:
            print(f"{name}: intento {attempt + 1} falló ({exc})")
        time.sleep(15 * 2 ** attempt)
    raise RuntimeError(f"No se pudo descargar {name} de Overpass")


def convert(name: str, tls_type: str, lat: float, lon: float) -> None:
    folder = config.NETS_DIR / name
    out = folder / f"{name}_{tls_type}.net.xml"
    crop = f"{lon - CROP_DEG},{lat - CROP_DEG},{lon + CROP_DEG},{lat + CROP_DEG}"
    cmd = [
        config.bin_path("netconvert"),
        "--osm-files", str(folder / f"{name}.osm.xml"),
        "--output-file", str(out),
        # Recorte alrededor del cruce: sumo-rl controla TODOS los semáforos de la red,
        # así que la red debe contener solo el del cruce estudiado.
        "--keep-edges.in-geo-boundary", crop,
        # Solo vías para autos; se descartan peatonales, ciclovías y servicios
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.by-type", "highway.service,highway.track,highway.footway,highway.path,highway.steps",
        # Limpieza de geometría típica de OSM
        "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
        "--junctions.join", "--junctions.join-dist", "25",
        # Semáforos: unir los nodos cercanos de un mismo cruce en un solo controlador
        "--tls.guess-signals", "--tls.join", "--tls.join-dist", "40", "--tls.discard-simple",
        "--tls.default-type", tls_type,
        "--tls.yellow.time", str(config.YELLOW_TIME),
        "--no-turnarounds", "true",
        "--output.street-names", "true",
    ]
    subprocess.run(cmd, check=True)
    keep_only_main_tls(out)
    print(f"Red generada: {out.relative_to(config.ROOT)}")


def keep_only_main_tls(net_path) -> None:
    """Deja solo el semáforo principal (el de más conexiones) y convierte los vecinos en cruces sin semáforo.

    sumo-rl controla todos los semáforos de la red, así que los vecinos que caen dentro del recorte
    (p. ej. 12 calle y 1a avenida junto a la Reforma) se quitan. Es una simplificación que se documenta.
    Se hace en una segunda pasada porque netconvert fusiona los nodos y sus IDs finales no se conocen antes.
    """
    import sumolib

    net = sumolib.net.readNet(str(net_path))
    tls = sorted(net.getTrafficLights(), key=lambda t: len(t.getConnections()), reverse=True)
    if len(tls) <= 1:
        return
    junctions = sorted({c[1].getEdge().getFromNode().getID() for t in tls[1:] for c in t.getConnections()})
    print(f"  se quitan {len(tls) - 1} semáforo(s) vecino(s): {', '.join(t.getID() for t in tls[1:])}")
    subprocess.run([config.bin_path("netconvert"), "--sumo-net-file", str(net_path), "--output-file", str(net_path),
                    "--tls.unset", ",".join(junctions)], check=True)


def main() -> None:
    names = sys.argv[1:] or list(INTERSECTIONS)
    for name in names:
        info = INTERSECTIONS[name]
        print(f"== {info['nombre']} ==")
        download(name, info["lat"], info["lon"])
        convert(name, "static", info["lat"], info["lon"])
        convert(name, "actuated", info["lat"], info["lon"])


if __name__ == "__main__":
    main()
