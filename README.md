# Control adaptativo de semáforos con Aprendizaje por Refuerzo

Proyecto Final de Aprendizaje por Refuerzo, Universidad del Valle de Guatemala.
**Autor:** Pablo César López Medina (22535).

Agentes de RL (Q-learning, DQN, PPO) que deciden qué fase de un semáforo activar según el tráfico, entrenados
en el simulador [SUMO](https://eclipse.dev/sumo/) y comparados contra semáforos de tiempo fijo, actuados y
planes fijos optimizados, en un cruce sintético y en dos cruces reales de Ciudad de Guatemala.

- **Documento final:** [`docs/Proyecto_Final_RL_Pablo_Lopez.pdf`](docs/Proyecto_Final_RL_Pablo_Lopez.pdf)
- **Registro de prompts de IA:** [`docs/registro_prompts.md`](docs/registro_prompts.md)

## Hallazgos principales

1. **Primera iteración (cruce sintético, demanda moderada):** Q-learning y PPO redujeron la espera a la mitad
   frente al semáforo actuado (de ~10 s a ~5 s). Al analizar qué aprendieron, resultó una política fija:
   cambiar de fase lo más rápido posible. Un semáforo de tiempo fijo con ciclo corto la iguala o supera.
   La mejora venía de baselines mal ajustados, no de adaptarse.
2. **Segunda iteración (escenario saturado):** con demanda cerca de la capacidad y cambiante, Q-learning
   entrenado ahí sí se adapta: 42.2 s de espera contra 72–76 s del mejor plan fijo. El actuado (36.0 s) sigue
   siendo mejor.
3. **DQN** fue inestable: colapsó en estados de congestión que no vio en el entrenamiento.
4. **Cruces reales** (Av. Petapa y 53 calle; Av. La Reforma y 12 calle): construidos con datos de OpenStreetMap,
   tras descartar la importación automática por bloqueos en los cruces con camellón.

## Instalación

Requiere Python 3.10 o superior. SUMO se instala con pip (`eclipse-sumo`).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Uso

```bash
# Redes y demanda
python src/generate_network.py           # cruce sintético (fijo, actuado, fijo corto)
python src/generate_routes.py            # escenarios valle, pico, variable, saturado
python src/build_real.py                 # cruces reales con datos de OSM
python src/generate_routes_real.py       # demanda estimada de los cruces reales
python src/tune_fixed.py --scenario saturado   # mejor plan fijo (baseline fuerte)

# Entrenamiento
python src/train_qlearning.py                                        # cruce sintético, escenario variable
python src/train_qlearning.py --scenario saturado --name qlearning_saturado
python src/train_qlearning.py --net petapa_53                        # cruce real
python src/train_sb3.py --algo ppo                                   # DQN o PPO

# Evaluación y análisis
python src/evaluate.py --scenarios saturado
python src/evaluate.py --net petapa_53 --scenarios pico
python src/analysis.py                   # uso de fases y equidad
python src/plots.py                      # figuras

# Ver un agente en la interfaz gráfica de SUMO
python src/evaluate.py --gui --policies qlearning_saturado --scenarios saturado --seeds 101
```

## Estructura

```
src/
  config.py              parámetros de simulación
  generate_network.py    cruce sintético
  generate_routes.py     escenarios de demanda del cruce sintético
  build_real.py          cruces reales con datos de OSM
  import_osm.py          importación automática de OSM (descartada; ver documento)
  generate_routes_real.py demanda estimada de los cruces reales
  envs.py                entorno Gymnasium sobre SUMO y observación discreta
  agents/qlearning.py    Q-learning tabular
  train_qlearning.py     entrenamiento tabular
  train_sb3.py           entrenamiento de DQN y PPO
  tune_fixed.py          búsqueda del mejor plan fijo
  evaluate.py            evaluación con semillas fijas
  analysis.py            uso de fases y equidad
  metrics.py             métricas por episodio
  plots.py               figuras
nets/                    redes y demanda (single = cruce sintético)
models/                  agentes entrenados
results/                 CSV de entrenamiento y evaluación, figuras
docs/                    documento final, propuesta, registro de prompts, guion del video
```

## Formulación (MDP)

- **Estado:** fase actual, si ya se cumplió el verde mínimo y vehículos detenidos por acceso
  (discretizados para Q-learning; continuos para DQN y PPO).
- **Acción:** próxima fase verde.
- **Recompensa:** reducción del tiempo de espera acumulado entre decisiones.
- **Decisión** cada 5 s simulados, con amarillo de 3 s y verde mínimo de 10 s.
