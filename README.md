# Réplica parcial de Superphot+ (de Soto et al. 2024)

Reproducción simplificada del clasificador fotométrico de supernovas del artículo
*Superphot+: Realtime Fitting and Classification of Supernova Light Curves*
(arXiv:2403.07975), más un experimento propio sobre el balanceo de clases.

**Autor:** Jesús David Vitola Gonzalez

## 1. Qué hace este repositorio

El artículo clasifica supernovas en cinco clases (SN Ia, SN II, SN Ib/c, SN IIn y SLSN-I) usando solo la
forma y el color de la curva de luz, sin redshift. Lo hace en dos etapas:

1. Ajusta cada curva (bandas g y r de ZTF) a un modelo paramétrico y se queda con sus parámetros.
2. Entrena un clasificador LightGBM con esos parámetros.

Este repositorio reimplementa ese flujo con simplificaciones y lo usa como línea base para
comparar tres formas de tratar el desbalance de clases: sin balancear, pesos de clase y SMOTE.

## 2. Estructura

```
.
├── README.md
├── requirements.txt          # versiones exactas de las librerías
├── data/
│   ├── training_set.csv      # lista de objetos: NAME, LABEL, Z (copiada del repositorio de los autores)
│   ├── muestra.csv           # objetos de las 5 clases del paper (generado)
│   ├── raw/                  # curvas descargadas de ALeRCE, un .parquet por objeto (no se versiona)
│   ├── processed/            # curvas tras el preprocesamiento (generado)
│   ├── resultado_cortes.csv  # qué curvas pasan los cortes de calidad y por qué se rechazan (generado)
│   ├── ajustes.csv           # parámetros ajustados por curva y banda (generado)
│   └── features.csv          # tabla de features para el clasificador (generado)
├── src/
│   └── descarga.py           # descarga de curvas desde ALeRCE
├── notebooks/                # ver el orden en la sección 4
└── results/                  # tablas y figuras de salida
```

## 3. Datos

| Qué | Fuente |
|---|---|
| Lista de objetos con clase espectroscópica y redshift | `data/training_set.csv`, copia del archivo `data/training_set.csv` del repositorio https://github.com/VTDA-Group/superphot-plus (licencia MIT). Las clases provienen del Transient Name Server (TNS). |
| Curvas de luz (bandas g y r de ZTF) | API pública de ALeRCE, con el cliente `alerce` (`query_detections`, `survey="ztf"`). No requiere credenciales. |

Las curvas **no están en el repositorio**: se descargan con `src/descarga.py` (sección 4). ALeRCE sigue
procesando datos, así que las curvas pueden diferir de las que usaron los autores en 2023.

## 5. Cómo ejecutarlo

### 5.1 Entorno

Se usó Python 3.11 (con 3.13 falló la instalación de una dependencia de otro paquete, por eso se fija 3.11).

```bash
git clone https://github.com/ApersonJ/replica_superphot
cd <carpeta-del-repositorio>
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Con `pip` en lugar de `uv`: `python3.11 -m venv .venv`, activar y `pip install -r requirements.txt`.

### 5.2 Orden de ejecución

Todos los comandos se lanzan desde la raíz del repositorio. Los notebooks usan rutas relativas
(`../data/...`), así que deben abrirse desde la carpeta `notebooks/` (Jupyter lo hace por defecto).

| Paso | Qué hace | Salida | Tiempo aproximado |
|---|---|---|---|
| 1. `01_muestra.ipynb` | Agrupa las 41 etiquetas de TNS en las 5 clases del paper y deja 5.356 objetos | `data/muestra.csv` | segundos |
| 2. `python src/descarga.py` | Descarga las curvas de ALeRCE | `data/raw/*.parquet`, `data/fallos_descarga.csv` | ~1 hora |
| 3. `03_preprocesamiento.ipynb` | Flujo, recorte de colas y cortes de calidad | `data/processed/`, `data/resultado_cortes.csv` | minutos |
| 4. notebook del modelo y los ajustes | Ecuación 1, ajuste por banda con `curve_fit`, corte de χ² y tabla de features | `data/ajustes.csv`, `data/features.csv` | minutos |
| 5. `05_clasificador.ipynb` | LightGBM con 10 folds estratificados y comparación con el paper | tablas en `results/` | minutos |
| 6. `06_gap_balanceo.ipynb` | Comparación de métodos de balanceo con 5 semillas | `results/comparacion_balanceo.csv`, `results/balanceo_comp_pureza.png` | decenas de minutos |

> Ajusta los nombres de los notebooks a los que tengas en `notebooks/`. El orden es el de la tabla.

El script de descarga se puede detener y volver a lanzar sin perder lo descargado.

### 5.3 Semillas

- Selección de folds y clasificador del notebook 5: `SEED = 42`.
- Comparación de balanceo (notebook 6): semillas 0 a 4. Con la misma semilla, los folds son idénticos
  entre métodos, así que la comparación es pareada.

## 6. Qué se replica y qué se simplificó

### Fiel al paper

- Modelo de curva de luz de la ecuación 1, con la misma parametrización y los rangos de la Tabla 2.
- Cortes de calidad de la Sección 2.3 (al menos 5 puntos con SNR ≥ 3 por banda, amplitud y variabilidad).
- Agrupación de clases de la Sección 2.4 (SLSN-II se funde con SN IIn; Ib, Ic e Ic-BL forman Ib/c).
- Clasificador LightGBM con DART, profundidad 5, 20 hojas y 250 estimadores.
- Validación cruzada estratificada de 10 folds y las mismas métricas (completitud, pureza, F1 macro).

### Simplificaciones y supuestos (diferencias con el paper)

| Tema | Paper | Esta réplica |
|---|---|---|
| Ajuste de la curva | Nested sampling (`dynesty`) con las dos bandas a la vez (14 parámetros) y priors correlacionados | Mínimos cuadrados (`scipy.optimize.curve_fit`), cada banda por separado (6 parámetros) |
| `σ_extra` | Incluido en el ajuste | No incluido: 10 features en vez de 12 |
| Balanceo de clases | Sobremuestreo bayesiano con muestras de la posterior | Pesos de clase (línea base) y SMOTE (experimento) |
| Corrección por extinción de la Vía Láctea | Sí | **No aplicada** |
| Corte de calidad del ajuste | χ² ≤ 1,2 | χ² conjunto ≤ 3, elegido mirando la distribución (con otra definición de χ², sin `σ_extra`, no es comparable) |
| Detecciones con diferencia negativa (`isdiffpos = -1`) | No se detalla | Descartadas (78 objetos, todos de 2018, quedan sin puntos) |
| Error del flujo | No se da la fórmula | Propagación de errores: σ_f = 0,4·ln(10)·f·σ_m |
| Recorte de colas | Apéndice A | Interpretación propia del algoritmo; la amplitud se define como máximo menos mínimo |
| `beta` | Rango 0 a 0,03 | Mínimo de 1e-4 para evitar logaritmos infinitos en las razones g/r |
| Logaritmos de la Tabla 2 | No se indica la base | Se supone base 10 (los centros de los priors lo respaldan) |
| Regularización de LightGBM | "Regularización 5" | Se interpreta como L2 (`reg_lambda=5`); sin GOSS ni partición de validación interna |
| Datos | 9.526 objetos de TNS, 6.123 tras los cortes | `training_set.csv` del repositorio de los autores (5.735 objetos; 5.356 en las 5 clases) |

Embudo de datos de esta réplica: 5.356 objetos → 5.224 pasan los cortes de calidad → 5.223 ajustes
convergen → **4.902** curvas tras el corte de χ² (SN Ia 3.694, SN II 763, SN Ib/c 189, SN IIn 187, SLSN-I 69).

## 7. Resultados principales

### Reproducción (LightGBM con pesos de clase, 10 folds)

| Métrica | Paper | Esta réplica |
|---|---|---|
| Accuracy | 0,83 ± 0,01 | 0,827 (0,809 – 0,845) |
| F1 macro | 0,61 ± 0,02 | 0,556 (0,49 – 0,588) |

Los rangos de la réplica son el segundo valor más bajo y el segundo más alto entre los 10 folds, como en el paper.

| Clase | Completitud paper | Completitud réplica | Pureza paper | Pureza réplica |
|---|---|---|---|---|
| SLSN-I | 0,75 | 0,33 | 0,27 | 0,23 |
| SN II | 0,72 | 0,73 | 0,87 | 0,70 |
| SN IIn | 0,52 | 0,52 | 0,51 | 0,45 |
| SN Ia | 0,87 | 0,89 | 0,97 | 0,96 |
| SN Ib/c | 0,80 | 0,55 | 0,28 | 0,27 |

La accuracy y las clases grandes coinciden con el paper. Las diferencias están en las clases raras
(SLSN-I e Ib/c), donde se pierde completitud.

### Experimento propio: balanceo de clases

Media de 5 semillas (la desviación está en `results/comparacion_balanceo.csv`). Estas métricas se calculan
sobre las predicciones de todos los folds juntas, por eso no coinciden exactamente con la tabla anterior.

| Método | Accuracy | F1 macro | Completitud SLSN-I | Completitud Ib/c | Pureza SLSN-I | Pureza Ib/c |
|---|---|---|---|---|---|---|
| Sin balancear | 0,876 | 0,522 | 0,154 | 0,141 | 0,447 | 0,574 |
| Pesos de clase | 0,828 | 0,540 | 0,284 | 0,531 | 0,200 | 0,263 |
| SMOTE | 0,816 | 0,541 | 0,426 | 0,544 | 0,210 | 0,240 |
| SMOTE con features estandarizadas | 0,820 | 0,545 | 0,380 | 0,564 | 0,223 | 0,234 |

Conclusiones, con los límites de tener solo 5 semillas y 69 curvas en SLSN-I:

- Balancear sube mucho la completitud de las clases raras y reduce su pureza a menos de la mitad.
- SMOTE solo se distingue de los pesos de clase en SLSN-I (gana en las 5 semillas).
- Ningún método de balanceo cierra la diferencia de F1 con el paper (0,61), lo que sugiere que pesa más la
  calidad de las features (ajuste por banda, sin `σ_extra`) que el balanceo.

## 8. Limitaciones

- No se aplicó la corrección por extinción de la Vía Láctea.
- Los ajustes por banda son poco fiables en curvas con pocos puntos (alrededor del 31 % de las curvas
  tiene 8 puntos o menos en la banda g) y `beta` aparece muy a menudo en los límites del rango.
- SLSN-I tiene solo 69 curvas, por lo que sus métricas varían mucho entre semillas.
- Las curvas de ALeRCE y la lista de objetos pueden diferir de los usados por los autores.

## 9. Referencias

- de Soto, K. M., Villar, A., Berger, E., et al. 2024, *Superphot+: Realtime Fitting and Classification of
  Supernova Light Curves*, arXiv:2403.07975. Código: https://github.com/VTDA-Group/superphot-plus
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. 2002, *SMOTE: Synthetic Minority
  Over-sampling Technique*, Journal of Artificial Intelligence Research, 16, 321.
- Ke, G., et al. 2017, *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 30.
- Förster, F., et al. 2021, *The Automatic Learning for the Rapid Classification of Events (ALeRCE) Alert
  Broker*, AJ, 161, 242.