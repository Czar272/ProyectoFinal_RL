# Control adaptativo de semáforos con Aprendizaje por Refuerzo

Proyecto Final de Aprendizaje por Refuerzo, Universidad del Valle de Guatemala.

Agentes de RL que deciden qué fase de un semáforo activar según el tráfico, entrenados en el simulador
[SUMO](https://eclipse.dev/sumo/) y comparados contra un semáforo de tiempo fijo y uno actuado.

## Progreso

| Fase | Estado |
| --- | --- |
| 1. Escenario sintético (cruce de 4 vías con giros protegidos) y baselines | Listo |
| 2. Q-learning tabular | Listo |
| 3. DQN | En curso |
| 4. PPO | En curso |

## Resultados preliminares

Espera promedio por vehículo (s), 5 semillas por escenario. Q-learning se entrenó solo en `variable`.

| Escenario | Tiempo fijo | Actuado | Q-learning |
| --- | --- | --- | --- |
| Valle | 16.4 ± 0.7 | 8.5 ± 0.4 | **4.0 ± 0.2** |
| Pico | 18.7 ± 0.5 | 10.0 ± 0.2 | **5.7 ± 0.4** |
| Variable | 17.7 ± 0.5 | 9.6 ± 0.3 | **5.0 ± 0.1** |
| 5. Intersección real de Ciudad de Guatemala (OpenStreetMap) | Pendiente |
| 6. Corredor multi-agente (opcional) | Pendiente |

## Instalación

Requiere Python 3.10 o superior. SUMO se instala con pip (`eclipse-sumo`), no hace falta instalarlo aparte.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Uso

```bash
python src/generate_network.py      # red del cruce (tiempo fijo y actuado)
python src/generate_routes.py       # demanda: valle, pico, variable
python src/train_qlearning.py       # entrena Q-learning -> models/qlearning.pkl
python src/evaluate.py              # compara todas las políticas -> results/evaluacion.csv

# Ver un agente en la interfaz gráfica de SUMO
python src/evaluate.py --gui --policies qlearning --scenarios pico --seeds 101
```

## Estructura

```
src/
  config.py            parámetros de simulación y rutas
  generate_network.py  red del cruce (netconvert)
  generate_routes.py   escenarios de demanda
  envs.py              entorno Gymnasium sobre SUMO y observación discreta
  agents/qlearning.py  agente Q-learning tabular
  train_qlearning.py   entrenamiento tabular
  evaluate.py          evaluación de políticas con semillas fijas
  metrics.py           métricas por episodio (tripinfo de SUMO)
nets/single/           red y demanda generadas
models/                agentes entrenados
results/               CSV de entrenamiento y evaluación
docs/                  propuesta, documento final y registro de prompts
```

## Formulación (MDP)

- **Estado:** fase actual, si ya se cumplió el verde mínimo y vehículos detenidos por acceso
  (discretizados para Q-learning; continuos para DQN/PPO).
- **Acción:** próxima fase verde, entre 4: recto N-S, giro izquierdo N-S, recto E-O y giro izquierdo E-O.
- **Recompensa:** reducción del tiempo de espera acumulado entre decisiones.
- **Decisión** cada 5 s simulados, con amarillo de 3 s y verde mínimo de 10 s.
