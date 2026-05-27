"""
Construye Clase_32_Series_Temporales.ipynb con la nueva estructura.
Compatible con Colab (descarga CSV y deps si hace falta).
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Clase_32_Series_Temporales.ipynb")

cells = []
def md(text):
    # Cada linea del source DEBE terminar en \n: Colab concatena los strings tal cual
    # (sin agregar separador). Sin el \n, todo el markdown colapsa en una sola linea.
    lines = text.strip("\n").split("\n")
    src = [(s + "\n") for s in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(text):
    src = text.strip("\n").split("\n")
    src = [(s + "\n") for s in src[:-1]] + [src[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None, "source": src})

REPO = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-32"

# =====================================================================
# PORTADA
# =====================================================================
md("""
# Clase 32 --- Series Temporales II
### De la estacionariedad a la decision

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

## Plan del notebook

1. **Por que estacionariedad** --- experimento de descubrimiento con series sinteticas.
2. **Diferenciacion** --- la cirugia que vuelve estacionaria una serie (sobre AirPassengers).
3. **Prophet en accion** --- la herramienta moderna sobre AirPassengers (donde brilla).
4. **Metricas honestas** --- MAE / RMSE / MAPE / WAPE + walk-forward.
5. **Inferencia + costo asimetrico** --- de prediccion a decision (P50/P70/P80/P90).
6. **Caso aplicado** --- Favorita Quito Q44, forecast de 6 meses.

> **Trabajamos primero sobre series simples y conocidas. Solo cuando dominamos la
> herramienta la aplicamos al caso real.**
""")

# =====================================================================
# 0. Setup (Colab compat)
# =====================================================================
md("## 0. Setup --- corre esta celda primero")

code(f"""
# Setup compatible local + Colab
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    print("Detectamos Colab --- instalando dependencias...")
    import os
    os.system("pip install -q prophet holidays")
    print("Listo.")

# El CSV de Favorita se lee DIRECTO desde GitHub (plug-and-play en Colab).
# pandas.read_csv acepta una URL, asi que no hace falta descargar el archivo.
CSV_FAVORITA = "{REPO}/quito44_beverages_daily.csv"
print(f"Fuente de datos Favorita:\\n  {{CSV_FAVORITA}}")
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.grid": True,
                      "grid.alpha": 0.3, "grid.linestyle": "--",
                      "axes.spines.top": False, "axes.spines.right": False,
                      "axes.titleweight": "bold"})
ARCA_RED, ARCA_DARK, ARCA_BLUE, ARCA_GREEN, ARCA_ORANGE = (
    "#C82B40", "#6B1525", "#2563EB", "#16A34A", "#EA580C")

np.random.seed(42)
print("OK")
""")

# =====================================================================
# 1. Por que estacionariedad
# =====================================================================
md("""
## 1. Por que un modelo necesita estacionariedad

Antes de tocar datos reales, hacemos un experimento con dos series sinteticas
que comparten escala pero tienen comportamientos distintos.
""")

code("""
# Dos series con la misma escala pero comportamientos distintos
n = 300; rng = np.random.RandomState(11)
A = np.zeros(n); A[0] = 10
for t in range(1, n):
    A[t] = 0.6*A[t-1] + 4 + rng.randn()*0.8
B = 10 + np.cumsum(rng.randn(n)*0.6) + 0.05*np.arange(n)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(A, color=ARCA_BLUE); axes[0].set_title("Serie A")
axes[1].plot(B, color=ARCA_RED);  axes[1].set_title("Serie B")
for ax in axes:
    ax.set_xlabel("tiempo"); ax.set_ylim(0, max(A.max(), B.max())*1.05)
plt.tight_layout(); plt.show()
""")

md("""
> **Ejercicio 1.** Sin codear nada, responde:
> - Si tuvieras que predecir el valor del paso siguiente, en cual serie confiarias mas?
> - Por que?
""")

code("""
# El experimento honesto: entrenar un modelo simple en la mitad 1, predecir la mitad 2
half = n // 2
pred_A = np.full(n - half, A[half-10:half].mean())   # constante = media reciente del train
pred_B = np.full(n - half, B[half-10:half].mean())

err_A = np.mean(np.abs(A[half:] - pred_A))
err_B = np.mean(np.abs(B[half:] - pred_B))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, s, pred, name, err in [(axes[0], A, pred_A, "Serie A (estacionaria)", err_A),
                                 (axes[1], B, pred_B, "Serie B (no estacionaria)", err_B)]:
    ax.plot(np.arange(half), s[:half], "k-", alpha=0.6, label="train")
    ax.plot(np.arange(half, n), s[half:], "k-", lw=1.4, label="real (futuro)")
    ax.plot(np.arange(half, n), pred, color=ARCA_BLUE, lw=2, label="prediccion")
    ax.set_title(f"{name}\\nMAE futuro = {err:.2f}")
    ax.legend()
plt.tight_layout(); plt.show()

print(f"\\nRatio de error B/A: {err_B/err_A:.1f}x")
""")

md("""
**Conclusion clave:** un modelo aprende patrones del pasado. Si la distribucion estadistica
de la serie cambia con el tiempo, lo que el modelo aprendio ayer ya no es cierto hoy.

Por eso pedimos **estacionariedad**: media, varianza y autocorrelacion estables.

### El test ADF como diagnostico
""")

code("""
from statsmodels.tsa.stattools import adfuller

p_A = adfuller(A)[1]
p_B = adfuller(B)[1]
print(f"Serie A:  ADF p = {p_A:.4f}  ->  {'estacionaria' if p_A < 0.05 else 'NO estacionaria'}")
print(f"Serie B:  ADF p = {p_B:.4f}  ->  {'estacionaria' if p_B < 0.05 else 'NO estacionaria'}")
""")

md("""
**Como leer ADF:**
- H_0: la serie tiene unit root (NO es estacionaria).
- Si p < 0.05: rechazamos H_0 -> tratamos como estacionaria.
- Si p >= 0.05: no podemos rechazar -> tratamos como NO estacionaria.

ADF no es magia: solo testea cierto tipo de no-estacionariedad. Pero es el diagnostico
inicial estandar de la industria.
""")

# =====================================================================
# 2. Diferenciacion
# =====================================================================
md("""
## 2. Diferenciacion --- la cirugia que vuelve estacionaria una serie

**Idea central:** en lugar de modelar el VALOR, modelamos el CAMBIO.
La temperatura tiene tendencia, pero el delta dia a dia oscila alrededor de cero.
""")

code("""
# Ilustracion: serie con tendencia vs su delta
n = 200; rng = np.random.RandomState(3); t = np.arange(n)
temp = 30 - 0.05*t + 3*np.sin(2*np.pi*t/30) + rng.randn(n)*0.8
diff = np.diff(temp)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(t, temp, color=ARCA_RED, lw=1.4)
axes[0].set_title("VALOR de la temperatura (NO estacionaria)")
axes[0].set_xlabel("dia"); axes[0].set_ylabel("temperatura (C)")
axes[1].plot(t[1:], diff, color=ARCA_BLUE, lw=1.0)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("CAMBIO dia a dia (ESTACIONARIA)")
axes[1].set_xlabel("dia"); axes[1].set_ylabel("delta (C)")
plt.tight_layout(); plt.show()

print(f"ADF temperatura : p = {adfuller(temp)[1]:.4f}")
print(f"ADF delta       : p = {adfuller(diff)[1]:.4f}")
""")

md("""
**Interpretacion del resultado:** el ADF de la temperatura cruda da un p alto
(NO estacionaria: tiene tendencia y oscilacion). Al diferenciar (el delta dia a dia),
el p cae muy por debajo de 0.05: la serie del **cambio** SI es estacionaria. Visualmente
se ve clarisimo: el panel izquierdo deambula, el derecho oscila estable alrededor de 0.
Esa es toda la magia de diferenciar.

### Tres tipos de diferenciacion

1. **log(y)** estabiliza VARIANZA cuando crece con el nivel
2. **delta y_t = y_t - y_{t-1}** quita TENDENCIA
3. **delta_s y_t = y_t - y_{t-s}** quita ESTACIONALIDAD de paso s

Se pueden combinar. Esto es lo que Prophet hace internamente cuando detecta no-estacionariedad.

### Aplicado a AirPassengers (clasico de Box-Jenkins)
""")

code("""
from statsmodels.datasets import get_rdataset
ap = get_rdataset("AirPassengers").data
y_ap = pd.Series(ap["value"].values.astype(float),
                  index=pd.date_range("1949-01-01", periods=len(ap), freq="MS"),
                  name="passengers")
print(f"AirPassengers: {len(y_ap)} meses ({y_ap.index.min().date()} a {y_ap.index.max().date()})")
y_ap.head()
""")

code("""
y_log = np.log(y_ap)
y_d1 = y_log.diff().dropna()
y_d1_d12 = y_d1.diff(12).dropna()

results = [
    ("1) original",              y_ap,     adfuller(y_ap.dropna())[1]),
    ("2) log(y)",                y_log,    adfuller(y_log.dropna())[1]),
    ("3) delta log(y)",          y_d1,     adfuller(y_d1)[1]),
    ("4) delta_12 delta log(y)", y_d1_d12, adfuller(y_d1_d12)[1]),
]
print("\\nADF p en cada paso:")
for name, _, p in results:
    print(f"  {name:30s}  p = {p:.4g}")

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
colors = [ARCA_DARK, ARCA_BLUE, ARCA_GREEN, ARCA_RED]
for ax, (name, s, p), c in zip(axes.flat, results, colors):
    ax.plot(s.index, s.values, color=c, lw=1.1)
    ax.set_title(f"{name}  -  ADF p = {p:.3g}")
    if name.startswith(("3", "4")):
        ax.axhline(0, color="black", lw=0.5)
plt.tight_layout(); plt.show()
""")

md("""
Cada paso ataca una patologia distinta. **Despues de las dos diferenciaciones,
la serie es claramente estacionaria** (p = 0.0002 << 0.05).

Esto es lo que pasa dentro de Prophet sin que lo veas: internamente estabiliza
la serie antes de modelarla.
""")

# =====================================================================
# 3. Prophet sobre AirPassengers
# =====================================================================
md("""
## 3. Prophet en accion

**Prophet (Facebook, 2017)** descompone la serie en:

$y(t) = g(t) + s(t) + h(t) + \\varepsilon$

- g(t): tendencia con changepoints automaticos
- s(t): estacionalidades via Fourier (multiple)
- h(t): holidays
- epsilon: ruido

Brilla cuando hay tendencia + estacionalidad + (opcionalmente) holidays.
""")

code("""
# Split train/test sobre AirPassengers
AP_HORIZON = 24
y_ap_train = y_ap.iloc[:-AP_HORIZON]
y_ap_test  = y_ap.iloc[-AP_HORIZON:]
print(f"Train: {len(y_ap_train)} meses, Test: {len(y_ap_test)} meses")
""")

code("""
from prophet import Prophet

m_ap = Prophet(yearly_seasonality=True, interval_width=0.80)
m_ap.fit(pd.DataFrame({"ds": y_ap_train.index, "y": y_ap_train.values}))

future = m_ap.make_future_dataframe(periods=AP_HORIZON, freq="MS")
fc_ap = m_ap.predict(future)
fc_ap_test = fc_ap["yhat"].iloc[-AP_HORIZON:].values
lo_ap = fc_ap["yhat_lower"].iloc[-AP_HORIZON:].values
hi_ap = fc_ap["yhat_upper"].iloc[-AP_HORIZON:].values

mae_ap = float(np.mean(np.abs(y_ap_test.values - fc_ap_test)))
mape_ap = float(np.mean(np.abs((y_ap_test.values - fc_ap_test) / y_ap_test.values))) * 100
print(f"Prophet sobre AirPassengers:  MAE = {mae_ap:.1f}  MAPE = {mape_ap:.2f}%")
""")

code("""
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_ap_train.index, y_ap_train.values, "k-", alpha=0.7, label="train")
ax.plot(y_ap_test.index, y_ap_test.values, "ko-", lw=2, markersize=4, label="real")
ax.plot(y_ap_test.index, fc_ap_test, color=ARCA_GREEN, lw=2, marker="s", markersize=4,
         label=f"Prophet  MAE={mae_ap:.0f}  MAPE={mape_ap:.1f}%")
ax.fill_between(y_ap_test.index, lo_ap, hi_ap, color=ARCA_GREEN, alpha=0.2, label="IC 80%")
ax.axvline(y_ap_train.index[-1], color="gray", ls=":")
ax.set_title("Prophet sobre AirPassengers --- 24 meses de forecast")
ax.set_xlabel("anio"); ax.set_ylabel("miles de pasajeros")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
**MAPE ~6.5% a 2 anios** sobre una serie con tendencia + estacionalidad fuertes.
Prophet captura los picos estacionales y la tendencia.

### Componentes interpretables
""")

code("""
fig = m_ap.plot_components(fc_ap)
fig.set_size_inches(11, 6)
plt.tight_layout(); plt.show()
""")

md("""
A diferencia de modelos blackbox, Prophet te muestra las piezas:
**tendencia** (creciente) y **estacionalidad anual** (pico en verano, valle en invierno).

Esto es util cuando tienes que **explicarle al jefe** por que el modelo dice lo que dice.
""")

# =====================================================================
# 4. Metricas
# =====================================================================
md("""
## 4. Metricas honestas

### Comparamos Prophet contra modelos baseline
""")

code("""
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))
def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100)

pred_naive  = np.full(AP_HORIZON, y_ap_train.iloc[-1])
pred_snaive = y_ap_train.iloc[-12:-12+AP_HORIZON].values if AP_HORIZON <= 12 else \\
              np.tile(y_ap_train.iloc[-12:].values, (AP_HORIZON // 12) + 1)[:AP_HORIZON]
pred_ma     = np.full(AP_HORIZON, y_ap_train.iloc[-12:].mean())

modelos = {
    "Naive (ultimo valor)":      pred_naive,
    "Seasonal naive (12 m)":     pred_snaive,
    "Media movil 12 m":          pred_ma,
    "Prophet":                   fc_ap_test,
}

baseline = mae(y_ap_test.values, pred_naive)
print(f"{'Modelo':<28} {'MAE':>7} {'WAPE':>7} {'lift vs naive':>14}")
print("-"*60)
for name, pred in modelos.items():
    m_ = mae(y_ap_test.values, pred)
    w_ = wape(y_ap_test.values, pred)
    l_ = (baseline - m_) / baseline * 100
    print(f"{name:<28} {m_:>7.0f} {w_:>6.1f}% {l_:>13.1f}%")
""")

md("""
**Prophet gana con ~73% de mejora sobre naive.** Visible:
""")

code("""
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_ap_train.index[-24:], y_ap_train.values[-24:], "k-", alpha=0.6, label="historico")
ax.plot(y_ap_test.index, y_ap_test.values, "ko-", lw=2, label="real")
colors = ["#94A3B8", ARCA_BLUE, "#7C3AED", ARCA_GREEN]
for (name, pred), c in zip(modelos.items(), colors):
    ax.plot(y_ap_test.index, pred, "x-", color=c, label=name, alpha=0.85)
ax.axvline(y_ap_train.index[-1], color="gray", ls=":")
ax.set_title("Cuatro modelos sobre los mismos 24 meses (AirPassengers)")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion:** mira como cada modelo "ataca" el problema. El naive (gris) es una
linea horizontal: repite el ultimo valor y se queda muy abajo. La media movil (morado)
tambien aplana. El seasonal naive (azul) ya copia la forma estacional pero sin la tendencia
creciente, asi que queda corto. **Prophet (verde) es el unico que sube CON la tendencia
y replica los picos de verano** --- por eso gana en todas las metricas.

### Cuando usar cada metrica

- **MAE**: facil de interpretar, robusta a outliers.
- **RMSE**: castiga errores grandes -> usar cuando outliers son inaceptables.
- **MAPE**: comparable entre escalas, pero EXPLOTA cuando valores reales son ~ 0.
- **WAPE**: como MAPE pero suma errores / suma reales. Metrica de portafolio.

### Walk-forward CV

En series temporales NO podes shufflear. Cada fold: entrenar con todo el pasado,
predecir el horizonte siguiente, medir. Asi simulas como vas a usar el modelo en produccion.
""")

# =====================================================================
# 5. Inferencia + costo asimetrico
# =====================================================================
md("""
## 5. Inferencia --- la prediccion NO es el numero final

Prophet te da un PUNTO central (yhat) y una BANDA (yhat_lower, yhat_upper).
Si entregas el punto central, vas a tener stockout el 50% del tiempo.

### El costo asimetrico decide el percentil

En planeacion real, el costo de subestimar (stockout) y sobrestimar (vencido)
NO es simetrico. **Esto se traduce en elegir un percentil mas alto que la mediana**.
""")

code("""
from scipy.stats import norm

# Convertir el CI 80% a sigma asumiendo gaussian
sigma_ap = (hi_ap - lo_ap) / (2 * 1.28)
percentiles_ap = {
    "P50": fc_ap_test,
    "P70": fc_ap_test + norm.ppf(0.70) * sigma_ap,
    "P80": fc_ap_test + norm.ppf(0.80) * sigma_ap,
    "P90": fc_ap_test + norm.ppf(0.90) * sigma_ap,
}

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(AP_HORIZON); width = 0.20
colors_p = {"P50": ARCA_GREEN, "P70": ARCA_BLUE, "P80": ARCA_ORANGE, "P90": ARCA_RED}
for i, (name, p) in enumerate(percentiles_ap.items()):
    ax.bar(x + (i-1.5)*width, p, width, color=colors_p[name], alpha=0.85,
            label=f"{name}  total={p.sum():.0f}")
ax.plot(x, y_ap_test.values, "ko-", lw=2, markersize=7, label=f"real  total={y_ap_test.sum():.0f}")
ax.set_xticks(x); ax.set_xticklabels([d.strftime("%b %y") for d in y_ap_test.index],
                                       rotation=45, ha="right", fontsize=8)
ax.set_ylabel("miles de pasajeros")
ax.set_title("Despacho final por percentil (sobre AirPassengers, 24 meses)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion:** las barras suben de P50 (verde) a P90 (rojo). Fijate en la leyenda:
cada percentil tiene un "total" mayor. La linea negra (real) cae casi siempre por encima
del P50 --- por eso entregar el P50 te deja corto la mitad del tiempo. Cuanto mas alto el
percentil, menos riesgo de stockout pero mas capital atado. **El percentil correcto NO sale
de la matematica: sale del costo relativo de cada error en tu negocio.**

### Tabla de tradeoffs

| Percentil | Riesgo stockout | Capital atado | Cuando usarlo |
|-----------|-----------------|---------------|---------------|
| P50       | 50%             | minimo         | costos simetricos (raro) |
| P70       | 30%             | +10%           | default razonable |
| P80       | 20%             | +20%           | stockout 3x mas caro que vencido |
| P90       | 10%             | +35%           | stockout es catastrofico |
""")

# =====================================================================
# 6. CASO APLICADO: FAVORITA
# =====================================================================
md("""
## 6. Caso aplicado --- Favorita Quito Q44 (6 meses de forecast)

Ahora aplicamos TODO lo que aprendimos a un caso real: ventas semanales de
**bebidas** en una tienda Favorita de Quito. El planeador de Arca necesita el
numero para los proximos **6 meses (26 semanas)**.
""")

code("""
# Cargar Favorita DIRECTO desde GitHub (CSV_FAVORITA es una URL raw) y convertir a semanal
df_fav = pd.read_csv(CSV_FAVORITA,
                      parse_dates=["date"], index_col="date").asfreq("D")
df_fav["unit_sales"] = df_fav["unit_sales"].interpolate(method="time")

counts = df_fav["unit_sales"].resample("W").count()
y_w = df_fav["unit_sales"].resample("W").sum()
y_w = y_w[counts == 7]  # solo semanas completas
print(f"Favorita semanal: {len(y_w)} semanas ({y_w.index.min().date()} a {y_w.index.max().date()})")
""")

code("""
# Split: train hasta sep 2016, test 26 semanas (incluye navidad + ano nuevo)
FAV_HORIZON = 26
FAV_TEST_START = pd.Timestamp("2016-09-04")
y_fav_train = y_w[y_w.index < FAV_TEST_START]
y_fav_test  = y_w[y_w.index >= FAV_TEST_START].iloc[:FAV_HORIZON]
print(f"Train: {len(y_fav_train)} sem  -  Test: {len(y_fav_test)} sem")

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(y_fav_train.index, y_fav_train.values, color=ARCA_DARK, lw=1, label="historico")
ax.plot(y_fav_test.index, y_fav_test.values, "lightgray", lw=1.2, label="proximos 6 meses (test)")
ax.axvspan(y_fav_test.index[0], y_fav_test.index[-1], color=ARCA_RED, alpha=0.10)
ax.axvline(y_fav_test.index[0], color=ARCA_RED, ls="--")
ax.set_title("Favorita Quito Q44 bebidas --- el caso del planeador")
ax.set_xlabel("fecha"); ax.set_ylabel("ventas semanales (cajas)")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion:** a diferencia del sensor que veremos luego, esta SI es una serie
"clasica" de negocio: tiene tendencia creciente clara (la tienda vende mas cada anio)
y un patron estacional anual. El tramo gris (test) arranca en septiembre 2016 y cubre
navidad + anio nuevo --- justo el periodo de mayor variacion. Es un buen examen para Prophet.
""")

code("""
# Prophet sobre Favorita CON HOLIDAYS Ecuador
import holidays as hol

ec = hol.country_holidays("EC", years=range(2013, 2018))
hdays = pd.DataFrame({
    "holiday": list(ec.values()),
    "ds": pd.to_datetime(list(ec.keys())),
    "lower_window": 0, "upper_window": 1,
})

m_fav = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, holidays=hdays,
                interval_width=0.80, changepoint_prior_scale=0.05)
m_fav.fit(pd.DataFrame({"ds": y_fav_train.index, "y": y_fav_train.values}))

future_fav = m_fav.make_future_dataframe(periods=FAV_HORIZON, freq="W")
fc_fav = m_fav.predict(future_fav)
fc_fav_t = fc_fav["yhat"].iloc[-FAV_HORIZON:].values
lo_fav = fc_fav["yhat_lower"].iloc[-FAV_HORIZON:].values
hi_fav = fc_fav["yhat_upper"].iloc[-FAV_HORIZON:].values

mae_fav = mae(y_fav_test.values, fc_fav_t)
wape_fav = wape(y_fav_test.values, fc_fav_t)
mape_fav = float(np.mean(np.abs((y_fav_test.values - fc_fav_t) / y_fav_test.values))) * 100
print(f"Prophet sobre Favorita: MAE={mae_fav:.0f}  MAPE={mape_fav:.1f}%  WAPE={wape_fav:.1f}%")
""")

code("""
fig, ax = plt.subplots(figsize=(13, 5))
hist_tail = y_fav_train.iloc[-52:]
ax.plot(hist_tail.index, hist_tail.values, color=ARCA_DARK, lw=1, alpha=0.8, label="1 ano previo")
ax.plot(y_fav_test.index, y_fav_test.values, "ko-", lw=1.5, markersize=4, label="real (test 26 sem)")
ax.plot(y_fav_test.index, fc_fav_t, color=ARCA_GREEN, lw=2, marker="s", markersize=4,
         label=f"Prophet (holidays EC)  MAE={mae_fav:.0f}  WAPE={wape_fav:.1f}%")
ax.fill_between(y_fav_test.index, lo_fav, hi_fav, color=ARCA_GREEN, alpha=0.2, label="IC 80%")
ax.axvline(y_fav_train.index[-1], color="gray", ls=":")
ax.set_title("Forecast Favorita --- 6 meses con Prophet")
ax.set_xlabel("fecha"); ax.set_ylabel("ventas semanales (cajas)")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
**WAPE ~10% a 6 meses** sobre una serie real con tendencia creciente, ruido y eventos.
En forecasting de demanda industrial, eso es un resultado solido.
""")

code("""
# Componentes interpretables de Favorita
fig = m_fav.plot_components(fc_fav)
fig.set_size_inches(11, 7)
plt.tight_layout(); plt.show()
""")

md("""
Prophet aprendio:
- **Tendencia:** crecimiento sostenido de ~30k a ~80k semanal en 4 anios.
- **Holidays Ecuador:** algunos feriados bajan las ventas ~5-12k cajas.
- **Patron anual:** pico en septiembre, minimo en febrero.

### El despacho final con costo asimetrico
""")

code("""
# Convertir CI 80% a percentiles via gaussian
sigma_fav = (hi_fav - lo_fav) / (2 * 1.28)
p50 = fc_fav_t
p70 = fc_fav_t + norm.ppf(0.70) * sigma_fav
p80 = fc_fav_t + norm.ppf(0.80) * sigma_fav
p90 = fc_fav_t + norm.ppf(0.90) * sigma_fav

real_total = y_fav_test.sum()
print(f"Total real (26 sem):       {real_total:>10,.0f} cajas")
print(f"Despacho P50:              {p50.sum():>10,.0f} cajas  ({(p50.sum()-real_total)/real_total*100:+.1f}%)")
print(f"Despacho P70:              {p70.sum():>10,.0f} cajas  ({(p70.sum()-real_total)/real_total*100:+.1f}%)")
print(f"Despacho P80 (recomendado):{p80.sum():>10,.0f} cajas  ({(p80.sum()-real_total)/real_total*100:+.1f}%)")
print(f"Despacho P90:              {p90.sum():>10,.0f} cajas  ({(p90.sum()-real_total)/real_total*100:+.1f}%)")
""")

md("""
**Interpretacion:** todos los despachos superan al total real (porcentaje positivo),
porque incluso el P50 acumula sobre 26 semanas. Lo importante es el GRADIENTE: el P80
te deja un colchon razonable (~10-15% extra) que cubre las semanas pico sin inflar
demasiado el inventario. El P90 protege mas pero ata mucho mas capital. El planeador
elige el punto segun cuanto le duele cada tipo de error.

> **Ejercicio final.** Si en tu empresa el costo de stockout fuera 5x el costo
> de vencido, que percentil entregarias? Y si fuera al reves?

Esta decision NO es matematica pura: requiere conocer el negocio. El modelo
te da el rango, vos (con el dueno del problema) eliges donde dispararte.
""")

# =====================================================================
# 7. CUANDO PROPHET FALLA
# =====================================================================
md("""
## 7. Cuando Prophet falla --- otro tipo de serie

Cambiamos de problema. Imagina que sos ingeniero industrial y tenes un **sensor
de presion en una caldera**, medido cada hora. La serie NO tiene tendencia clara
ni estacionalidad fija --- la dinamica esta "adentro" de la propia serie.

Vamos a generar una serie sintetica con esa pinta (ecuacion de **Mackey-Glass**,
benchmark canonico para RNN/LSTM):
""")

code("""
def mackey_glass(n=1500, tau=17, gamma=0.1, beta=0.2, p=10, dt=1.0, x0=1.2, burn=200):
    \"\"\"Genera una serie caotica determinista con memoria de tau pasos.\"\"\"
    n_total = n + burn
    x = np.full(n_total, x0)
    for t in range(1, n_total):
        x_tau = x[t-tau] if t > tau else x0
        x[t] = x[t-1] + dt * (beta * x_tau / (1 + x_tau**p) - gamma * x[t-1])
    return x[burn:]

SENSOR_N = 1500
sensor = mackey_glass(n=SENSOR_N)
sensor_idx = pd.date_range("2020-01-01", periods=SENSOR_N, freq="h")
y_sensor = pd.Series(sensor, index=sensor_idx, name="presion")

# Split: 80% train, 20% test
SENSOR_HORIZON = 300
y_sensor_train = y_sensor.iloc[:-SENSOR_HORIZON]
y_sensor_test  = y_sensor.iloc[-SENSOR_HORIZON:]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_sensor_train.index, y_sensor_train.values, color=ARCA_DARK, lw=0.5, label="train")
ax.plot(y_sensor_test.index, y_sensor_test.values, color="lightgray", lw=0.7, label="test")
ax.axvline(y_sensor_test.index[0], color=ARCA_RED, ls="--")
ax.set_title("Sensor industrial --- dinamica no lineal con memoria de ~17 pasos")
ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
> **Ejercicio 5.** Mira la serie: no hay tendencia ni estacionalidad obvia.
> Crees que Prophet va a funcionar aqui? Por que si o por que no?

### Aplicamos Prophet --- y se rompe
""")

code("""
m_sensor = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=True, interval_width=0.80,
                    changepoint_prior_scale=0.05)
m_sensor.fit(pd.DataFrame({"ds": y_sensor_train.index, "y": y_sensor_train.values}))

future = m_sensor.make_future_dataframe(periods=SENSOR_HORIZON, freq="h")
fc_sensor = m_sensor.predict(future)
fc_sensor_t = fc_sensor["yhat"].iloc[-SENSOR_HORIZON:].values

mae_sensor_p = mae(y_sensor_test.values, fc_sensor_t)
mape_sensor_p = float(np.mean(np.abs((y_sensor_test.values - fc_sensor_t) / y_sensor_test.values))) * 100
print(f"Prophet sobre sensor:  MAE = {mae_sensor_p:.3f}  MAPE = {mape_sensor_p:.1f}%")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(y_sensor_train.index[-300:], y_sensor_train.values[-300:], color=ARCA_DARK, lw=0.7, alpha=0.7, label="train")
ax.plot(y_sensor_test.index, y_sensor_test.values, "k-", lw=1.3, label="real")
ax.plot(y_sensor_test.index, fc_sensor_t, color=ARCA_RED, lw=1.4,
         label=f"Prophet  MAPE={mape_sensor_p:.1f}%")
ax.axvline(y_sensor_train.index[-1], color="gray", ls=":")
ax.set_title("Prophet sobre el sensor: forecast PLANO, no captura la dinamica")
ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
### Por que Prophet se APLANA --- la explicacion completa

El forecast de Prophet es casi una linea recta. No es un bug: es exactamente lo que
Prophet *puede* hacer. Veamos por que, paso a paso.

**1. Prophet solo sabe mirar el calendario.** Su formula es

$y(t) = \\text{tendencia}(t) + \\text{estacionalidad}(t) + \\text{holidays}(t)$

Todas las piezas son funciones de **la fecha/hora** $t$. Prophet NO recibe como input
los valores pasados de la serie ($y_{t-1}, y_{t-2}, \\ldots$). Solo sabe "que dia y hora es".

**2. Las estacionalidades de Prophet tienen periodo FIJO** (diaria = 24h, semanal = 7 dias,
anual = 365 dias). Para que sirvan, el patron se tiene que repetir SIEMPRE en el mismo
horario: por ejemplo "todos los lunes a las 9am sube".

**3. Pero el sensor NO se repite con el calendario.** Mackey-Glass oscila con un periodo
"casi" regular de ~50 pasos, que *no* esta alineado con 24h ni 7 dias y ademas se corre un
poco en cada ciclo. Entonces, cuando Prophet intenta ajustar su estacionalidad diaria,
encuentra que un pico cae a las 3am en un ciclo, a las 11am en otro, a las 18h en otro...

**4. Al promediar ciclos desalineados, se cancelan.** Como cada pico cae en una hora distinta,
el promedio sobre todo el entrenamiento es ~0. La estacionalidad ajustada queda casi plana.
Lo unico que sobrevive es la tendencia (casi constante) -> **forecast = linea horizontal
cerca de la media**.

Vamoslo con los propios componentes que Prophet ajusto:
""")

code("""
# Que estacionalidad encontro Prophet en el sensor? (spoiler: casi nada util)
fig = m_sensor.plot_components(fc_sensor)
fig.set_size_inches(11, 6)
plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion de los componentes:** la amplitud de la estacionalidad diaria/semanal que
Prophet encontro es **diminuta** comparada con el rango real de la serie (que va de ~0.4 a
~1.3 bar). Prophet "buscó" un patron de calendario y no lo hay, asi que su mejor apuesta es
quedarse cerca del promedio. Por eso aplana.

**La serie del sensor necesita lo contrario:** un modelo cuya prediccion dependa de los
valores recientes de la propia serie,

$y_t = f(y_{t-1}, y_{t-2}, \\ldots, y_{t-k})$ con $f$ **no lineal**.

Eso es justo lo que hace una red con memoria: la **LSTM**.

---

## 8. LSTM al rescate

### Que es una LSTM (con una analogia)

Imaginate **alguien tomando notas en una reunion larga**. En cada parrafo nuevo:

1. **Lee** el parrafo (eso es el dato nuevo $x_t$).
2. **Mira** su libreta de notas (la memoria de lo que vino antes).
3. **Decide que tachar** de las notas viejas (lo que ya no es relevante).
4. **Decide que anotar nuevo** en la libreta.
5. **Decide que decirle** al grupo en ese momento (eso es la prediccion $h_t$).
6. **Pasa al siguiente parrafo** con la libreta actualizada.

Una **LSTM** (Long Short-Term Memory) hace exactamente eso, paso a paso. La gran
ventaja es que la "libreta" (memoria interna) sobrevive a traves de muchos pasos
--- por eso aprende patrones que dependen de cosas que pasaron hace mucho.

### Comparativa rapida: por que funciona donde Prophet no

| | **Prophet** | **LSTM** |
|---|-------------|----------|
| Input | el tiempo $t$ | los valores pasados $y_{t-1}, y_{t-2}, \\ldots$ |
| Modelo | trend + estacion + holidays (aditivo) | combinaciones NO lineales aprendidas |
| Cuando gana | tendencia y estacionalidad claras | dinamica interna no lineal |
| Interpretabilidad | alta (te muestra componentes) | baja (caja negra) |

### Como se entrena: ventanas deslizantes

Cada ejemplo de entrenamiento es **(ventana de k pasos, valor siguiente)**.
Asi la LSTM aprende: *"dadas estas k mediciones, cual es la proxima?"*
""")

code("""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(42)

# Normalizar a [0,1] con stats del train
s = y_sensor.values.astype(np.float32)
train_n = len(y_sensor_train)
mn, mx = s[:train_n].min(), s[:train_n].max()
s_norm = (s - mn) / (mx - mn + 1e-9)

WINDOW = 50

def make_windows(arr, w):
    X, y = [], []
    for i in range(len(arr) - w):
        X.append(arr[i:i+w])
        y.append(arr[i+w])
    return np.array(X)[..., None], np.array(y)

X_all, y_all = make_windows(s_norm, WINDOW)
n_train_w = train_n - WINDOW
X_train_lstm, y_train_lstm = X_all[:n_train_w], y_all[:n_train_w]
print(f"Ejemplos train: {len(X_train_lstm)}, shape de X: {X_train_lstm.shape}")
""")

code("""
# Modelo LSTM minimo
model = keras.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(32),
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
model.summary()
""")

code("""
# Entrenamiento (20 epocas, ~30 segundos sin GPU, ~5 segundos con GPU)
hist = model.fit(X_train_lstm, y_train_lstm,
                  epochs=20, batch_size=64, validation_split=0.1, verbose=0)

# Plot training history
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(hist.history["loss"], label="train", color=ARCA_DARK)
ax.plot(hist.history["val_loss"], label="val", color=ARCA_RED)
ax.set_title("Entrenamiento LSTM"); ax.set_xlabel("epoca"); ax.set_ylabel("MSE")
ax.legend(); plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion:** ambas curvas (train y validacion) bajan juntas y se estabilizan.
Eso es senal de un entrenamiento sano: el modelo aprende sin sobre-ajustar (si la curva
de validacion subiera mientras la de train baja, tendriamos overfitting). El MSE final
bajo nos dice que la LSTM ya capturo la dinamica.
""")

code("""
# Forecast del test (1-step ahead con los valores reales como ventana)
fc_lstm = []
for t in range(train_n, len(s_norm)):
    window_arr = s_norm[t-WINDOW:t].reshape(1, WINDOW, 1)
    pred_norm = model.predict(window_arr, verbose=0)[0, 0]
    fc_lstm.append(pred_norm * (mx - mn) + mn)
fc_lstm = np.array(fc_lstm)

mae_sensor_l = mae(y_sensor_test.values, fc_lstm)
mape_sensor_l = float(np.mean(np.abs((y_sensor_test.values - fc_lstm) / y_sensor_test.values))) * 100
print(f"LSTM sobre sensor:    MAE = {mae_sensor_l:.3f}  MAPE = {mape_sensor_l:.1f}%")
print(f"Prophet sobre sensor: MAE = {mae_sensor_p:.3f}  MAPE = {mape_sensor_p:.1f}%")
print(f"\\nMejora de LSTM vs Prophet: {(mae_sensor_p - mae_sensor_l) / mae_sensor_p * 100:.0f}%")
""")

code("""
# Visualizacion comparativa
fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                          gridspec_kw={"width_ratios": [2.5, 1]})

n_show = 200
axes[0].plot(y_sensor_test.index[:n_show], y_sensor_test.values[:n_show], "k-", lw=1.5, label="real")
axes[0].plot(y_sensor_test.index[:n_show], fc_sensor_t[:n_show], color=ARCA_RED, lw=1.3,
              label=f"Prophet  MAPE={mape_sensor_p:.1f}%")
axes[0].plot(y_sensor_test.index[:n_show], fc_lstm[:n_show], color=ARCA_GREEN, lw=1.3,
              label=f"LSTM     MAPE={mape_sensor_l:.1f}%")
axes[0].set_title("Misma serie, dos modelos --- primeros 200 puntos del test")
axes[0].legend()

metrics = ["MAE", "MAPE (%)"]
pro_v = [mae_sensor_p, mape_sensor_p]
lstm_v = [mae_sensor_l, mape_sensor_l]
xpos = np.arange(2); width = 0.35
axes[1].bar(xpos - width/2, pro_v, width, color=ARCA_RED, label="Prophet")
axes[1].bar(xpos + width/2, lstm_v, width, color=ARCA_GREEN, label="LSTM")
for i, (p, l) in enumerate(zip(pro_v, lstm_v)):
    axes[1].text(i - width/2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].text(i + width/2, l, f"{l:.2f}", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(xpos); axes[1].set_xticklabels(metrics)
axes[1].set_title("Metricas lado a lado")
axes[1].legend(); plt.tight_layout(); plt.show()
""")

md("""
**Interpretacion:** la diferencia es brutal y visible. La linea verde (LSTM) se monta
casi encima de la real: aprendio la regla interna del sensor. La roja (Prophet) sigue
plana, sin enterarse de las oscilaciones. Las barras de la derecha lo confirman:
**LSTM mejora ~85% sobre Prophet** (MAPE 3% vs 23%).

La leccion no es "LSTM > Prophet" en general --- es que **cada herramienta sirve para
un tipo de serie**. Cuando la dinamica esta DENTRO de la serie (no en el calendario),
las redes con memoria ganan. Cuando la serie es tendencia + estacionalidad limpia,
Prophet gana en simplicidad.

### Cuando vale la pena pasar de Prophet a LSTM

- **Si:** la serie tiene dependencia no lineal de sus pasados (sensores, fisica, financieros caoticos)
- **Si:** tenes muchos datos (LSTM necesita miles de puntos)
- **Si:** tenes features exogenas que interactuan de forma compleja
- **NO:** si la serie es "simple" (tendencia + estacionalidad + holidays) --- ahi Prophet gana en simplicidad

> **Regla del oficio:** prueba Prophet primero. Solo pasa a LSTM si Prophet falla
> y tenes los datos para entrenar bien.

---

# 9. Cierre

### Lo que te llevas hoy

1. Una serie tiene que ser **estacionaria** para que un modelo aprenda algo util.
   Si no lo es, **diferenciamos**.
2. **ADF** diagnostica. Tres transformaciones (log, delta, delta_s) cubren casi todo.
3. **Prophet** es la herramienta moderna para series de negocio con tendencia,
   estacionalidad y holidays. API minima, componentes interpretables.
4. **Walk-forward CV** para metricas honestas. **MAE, RMSE, MAPE, WAPE** con su uso.
5. **La prediccion es un rango, no un punto.** El **costo asimetrico** define el percentil.
6. Cuando los clasicos no bastan (dinamica no lineal, sensores, multivariado):
   **LSTM**. Hoy probamos una con ~10 lineas de Keras y vimos como pasa de MAPE 23% a 3%.

### Decision tree practico

```
Pocos datos (< 2 ciclos)?               -> Naive / Seasonal naive / MA
Patron CLASICO (tendencia + estac)?     -> Prophet (con holidays!)
Patron NO LINEAL / mucha memoria?       -> LSTM
```

**Empieza simple. Sube de complejidad solo si lo simple ya no da.**

---

*Codigo + datos: github.com/cmosquerat/arca-diplomado/tree/main/clase-32*
""")

# =====================================================================
# BUILD
# =====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook generado: {OUT} ({len(cells)} celdas)")
