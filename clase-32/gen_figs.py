"""
Figuras clase 32 - Series Temporales II
De la estacionariedad a la decision + cuando los clasicos no bastan (LSTM).

Estructura pedagogica:
  Bloque 0: Setup conceptual del problema (planeacion en general)
  Bloque 1: Por que estacionariedad (con series sinteticas A/B)
  Bloque 2: Diferenciacion explicada (intuicion + AirPassengers paso a paso)
  Bloque 3: Prophet en accion (AirPassengers - la serie clasica donde brilla)
  Bloque 4: Metricas honestas (walk-forward, MAE/RMSE/MAPE/WAPE)
  Bloque 5: Inferencia + costo asimetrico
  Bloque 6: Caso aplicado a Favorita (6 meses adelante)
  Bloque 7: Cuando Prophet falla (sensor industrial con dinamica no lineal)
  Bloque 8: LSTM al rescate (intro + aplicacion al sensor)
  Bloque 9: Cierre - decision tree
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.datasets import get_rdataset
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = ROOT

ARCA_RED    = "#C82B40"
ARCA_DARK   = "#6B1525"
ARCA_GRAY   = "#F5F5F5"
ARCA_GREEN  = "#16A34A"
ARCA_BLUE   = "#2563EB"
ARCA_ORANGE = "#EA580C"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 130,
})

np.random.seed(42)

# =============================================================================
#  DATOS
# =============================================================================
print("Cargando datos...")

# 1) AirPassengers (mensual, 1949-1960, tendencia + estacionalidad anual)
ap = get_rdataset("AirPassengers").data
ap_dates = pd.date_range("1949-01-01", periods=len(ap), freq="MS")
y_ap = pd.Series(ap["value"].values.astype(float), index=ap_dates, name="passengers")
print(f"  AirPassengers: {len(y_ap)} meses ({y_ap.index.min().date()} a {y_ap.index.max().date()})")

# Split para AirPassengers: train = primeros 10 anios, test = ultimos 24 meses
AP_HORIZON = 24
y_ap_train = y_ap.iloc[:-AP_HORIZON]
y_ap_test  = y_ap.iloc[-AP_HORIZON:]

# 2) Favorita Quito Q44 (diario -> semanal, para el caso aplicado del bloque 6)
df = pd.read_csv(os.path.join(ROOT, "quito44_beverages_daily.csv"),
                 parse_dates=["date"], index_col="date").asfreq("D")
df["unit_sales"] = df["unit_sales"].interpolate(method="time")
y_d = df["unit_sales"]
counts = y_d.resample("W").count()
y_w = y_d.resample("W").sum()
y_w = y_w[counts == 7]
y_w.name = "ventas_semanales"
print(f"  Favorita semanal: {len(y_w)} semanas ({y_w.index.min().date()} a {y_w.index.max().date()})")

# Para el caso Favorita: train hasta sep 2016, test 26 semanas (= 6 meses, incluye navidad/anio nuevo)
FAV_HORIZON = 26
FAV_TEST_START = pd.Timestamp("2016-09-04")
y_fav_train = y_w[y_w.index < FAV_TEST_START]
y_fav_test  = y_w[y_w.index >= FAV_TEST_START].iloc[:FAV_HORIZON]
print(f"  Favorita train: {len(y_fav_train)} sem, test: {len(y_fav_test)} sem")


# =============================================================================
#  BLOQUE 0 - Setup conceptual: el problema de la planeacion
# =============================================================================
def fig_problema_planeacion():
    """Ilustracion conceptual: dos consecuencias del error de pronostico."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    dias = np.arange(28)
    real = 100 + 15*np.sin(2*np.pi*dias/7) + np.random.randn(28)*5

    pred_low = real - 35
    stockout = real > pred_low + 5
    axes[0].fill_between(dias, pred_low, real, where=stockout,
                          color=ARCA_RED, alpha=0.35, label="ventas perdidas")
    axes[0].plot(dias, real, color="black", lw=1.6, label="demanda real")
    axes[0].plot(dias, pred_low, color=ARCA_BLUE, lw=1.6, ls="--",
                  label="despacho (subestima)")
    axes[0].fill_between(dias, 0, pred_low, color="lightgray", alpha=0.3)
    axes[0].set_title("Subestimar = STOCKOUT\nGondola vacia, ventas perdidas",
                       color=ARCA_DARK, fontsize=11)
    axes[0].set_xlabel("dia"); axes[0].set_ylabel("cajas")
    axes[0].legend(loc="lower left", fontsize=10)
    axes[0].set_ylim(0, real.max()*1.15)

    pred_high = real + 35
    sobrestock = pred_high > real + 5
    axes[1].fill_between(dias, real, pred_high, where=sobrestock,
                          color=ARCA_ORANGE, alpha=0.35, label="producto vencido / capital atado")
    axes[1].plot(dias, real, color="black", lw=1.6, label="demanda real")
    axes[1].plot(dias, pred_high, color=ARCA_BLUE, lw=1.6, ls="--",
                  label="despacho (sobrestima)")
    axes[1].fill_between(dias, 0, real, color="lightgray", alpha=0.3)
    axes[1].set_title("Sobrestimar = PRODUCTO VENCIDO\nCapital inmovilizado",
                       color=ARCA_DARK, fontsize=11)
    axes[1].set_xlabel("dia")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].set_ylim(0, real.max()*1.45)

    fig.suptitle("El costo del error de pronostico es ASIMETRICO --- vamos a usar esto al final",
                  color=ARCA_DARK, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_problema_planeacion.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_problema_planeacion.png")
    plt.close(fig)


# =============================================================================
#  BLOQUE 1 - Por que estacionariedad
# =============================================================================
def _make_two_series():
    n = 300; rng = np.random.RandomState(11)
    a = np.zeros(n); a[0] = 10
    for t in range(1, n):
        a[t] = 0.6*a[t-1] + 4 + rng.randn()*0.8
    b = 10 + np.cumsum(rng.randn(n)*0.6) + 0.05*np.arange(n)
    return a, b

def fig_dos_series_descubrimiento():
    a, b = _make_two_series()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(a, color=ARCA_BLUE, lw=1.3)
    axes[0].set_title("Serie A", color=ARCA_DARK, fontsize=14)
    axes[0].set_xlabel("tiempo"); axes[0].set_ylabel("valor")
    axes[0].set_ylim(0, max(a.max(), b.max())*1.05)
    axes[1].plot(b, color=ARCA_RED, lw=1.3)
    axes[1].set_title("Serie B", color=ARCA_DARK, fontsize=14)
    axes[1].set_xlabel("tiempo")
    axes[1].set_ylim(0, max(a.max(), b.max())*1.05)
    fig.suptitle("Si tuvieras que predecir el valor de manana, en cual confiarias mas?",
                  color=ARCA_DARK, fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dos_series_descubrimiento.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_dos_series_descubrimiento.png")
    plt.close(fig)

def fig_dos_series_respuesta():
    a, b = _make_two_series()
    n = len(a); half = n // 2
    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                              gridspec_kw={"width_ratios": [2, 1]})

    axes[0,0].plot(np.arange(half), a[:half], color=ARCA_BLUE, lw=1.3, label="mitad 1")
    axes[0,0].plot(np.arange(half, n), a[half:], color=ARCA_ORANGE, lw=1.3, label="mitad 2")
    axes[0,0].axvline(half, color="gray", ls=":")
    axes[0,0].set_title("Serie A - distribucion estable  -->  ESTACIONARIA",
                         color=ARCA_DARK, fontsize=11)
    axes[0,0].set_xlabel("tiempo"); axes[0,0].legend(loc="upper right", fontsize=9)

    bins_a = np.linspace(a.min()-1, a.max()+1, 25)
    axes[0,1].hist(a[:half], bins=bins_a, color=ARCA_BLUE, alpha=0.6,
                    orientation="horizontal",
                    label=f"mitad 1\nmean={a[:half].mean():.1f}\nstd={a[:half].std():.1f}")
    axes[0,1].hist(a[half:], bins=bins_a, color=ARCA_ORANGE, alpha=0.6,
                    orientation="horizontal",
                    label=f"mitad 2\nmean={a[half:].mean():.1f}\nstd={a[half:].std():.1f}")
    axes[0,1].set_title("Distribuciones casi iguales", color=ARCA_DARK, fontsize=11)
    axes[0,1].set_xlabel("frecuencia"); axes[0,1].legend(loc="upper right", fontsize=8)

    axes[1,0].plot(np.arange(half), b[:half], color=ARCA_BLUE, lw=1.3, label="mitad 1")
    axes[1,0].plot(np.arange(half, n), b[half:], color=ARCA_ORANGE, lw=1.3, label="mitad 2")
    axes[1,0].axvline(half, color="gray", ls=":")
    axes[1,0].set_title("Serie B - nivel se desboca  -->  NO estacionaria",
                         color=ARCA_DARK, fontsize=11)
    axes[1,0].set_xlabel("tiempo"); axes[1,0].legend(loc="upper left", fontsize=9)

    bins_b = np.linspace(b.min()-1, b.max()+1, 25)
    axes[1,1].hist(b[:half], bins=bins_b, color=ARCA_BLUE, alpha=0.6,
                    orientation="horizontal",
                    label=f"mitad 1\nmean={b[:half].mean():.1f}\nstd={b[:half].std():.1f}")
    axes[1,1].hist(b[half:], bins=bins_b, color=ARCA_ORANGE, alpha=0.6,
                    orientation="horizontal",
                    label=f"mitad 2\nmean={b[half:].mean():.1f}\nstd={b[half:].std():.1f}")
    axes[1,1].set_title("Distribuciones muy distintas", color=ARCA_DARK, fontsize=11)
    axes[1,1].set_xlabel("frecuencia"); axes[1,1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Lo que aprendio el modelo en la mitad 1, sigue siendo cierto en la mitad 2?",
                  color=ARCA_DARK, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dos_series_respuesta.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_dos_series_respuesta.png")
    plt.close(fig)

def fig_modelo_falla():
    """Experimento: el mismo modelo simple aplicado a las 2 series, mide error en mitad 2."""
    a, b = _make_two_series()
    n = len(a); half = n // 2
    pred_a = np.full(n - half, a[half-10:half].mean())
    pred_b = np.full(n - half, b[half-10:half].mean())
    err_a = float(np.mean(np.abs(a[half:] - pred_a)))
    err_b = float(np.mean(np.abs(b[half:] - pred_b)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(np.arange(half), a[:half], color="black", lw=1.0, alpha=0.6, label="entrenamiento")
    axes[0].plot(np.arange(half, n), a[half:], color="black", lw=1.4, label="real (futuro)")
    axes[0].plot(np.arange(half, n), pred_a, color=ARCA_BLUE, lw=2,
                  label=f"prediccion = media de mitad 1")
    axes[0].axvspan(half, n, color=ARCA_GREEN, alpha=0.10)
    axes[0].axvline(half, color="gray", ls=":")
    axes[0].set_title(f"Serie A estacionaria  -  MAE futuro = {err_a:.2f}\n"
                       f"el modelo se mantiene cerca de la realidad",
                       color=ARCA_DARK, fontsize=11)
    axes[0].set_xlabel("tiempo"); axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(np.arange(half), b[:half], color="black", lw=1.0, alpha=0.6, label="entrenamiento")
    axes[1].plot(np.arange(half, n), b[half:], color="black", lw=1.4, label="real (futuro)")
    axes[1].plot(np.arange(half, n), pred_b, color=ARCA_BLUE, lw=2,
                  label=f"prediccion = media de mitad 1")
    axes[1].axvspan(half, n, color=ARCA_RED, alpha=0.12)
    axes[1].axvline(half, color="gray", ls=":")
    axes[1].set_title(f"Serie B NO estacionaria  -  MAE futuro = {err_b:.2f}\n"
                       f"el modelo se queda anclado al pasado",
                       color=ARCA_DARK, fontsize=11)
    axes[1].set_xlabel("tiempo"); axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle("Por que importa estacionariedad: si la distribucion cambia,\n"
                  "lo que el modelo aprendio ayer ya no aplica hoy",
                  color=ARCA_DARK, fontsize=12, y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_modelo_falla.png"), dpi=140, bbox_inches="tight")
    print(f"  fig_modelo_falla.png  errA={err_a:.2f}  errB={err_b:.2f}")
    plt.close(fig)
    return err_a, err_b


# =============================================================================
#  BLOQUE 2 - Diferenciacion (explicada mejor)
# =============================================================================
def fig_dif_intuicion():
    """
    La idea central de diferenciar: en lugar de predecir el VALOR, predecimos el CAMBIO.
    Visual didactico: temperatura cada dia (no estacionaria) vs delta dia a dia (estacionaria).
    """
    n = 200; rng = np.random.RandomState(3)
    t = np.arange(n)
    temp = 30 - 0.05*t + 3*np.sin(2*np.pi*t/30) + rng.randn(n)*0.8
    diff = np.diff(temp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(t, temp, color=ARCA_RED, lw=1.4)
    axes[0].set_title("VALOR de la temperatura cada dia\n"
                       "Tiene tendencia, nivel cambia  -->  NO estacionaria",
                       color=ARCA_DARK, fontsize=11)
    axes[0].set_xlabel("dia"); axes[0].set_ylabel("temperatura (C)")

    axes[1].plot(t[1:], diff, color=ARCA_BLUE, lw=1.0)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("CAMBIO dia a dia (delta)\n"
                       "Oscila alrededor de 0  -->  ESTACIONARIA",
                       color=ARCA_DARK, fontsize=11)
    axes[1].set_xlabel("dia"); axes[1].set_ylabel("delta (C)")

    fig.suptitle("Diferenciar = en vez de modelar el VALOR, modelamos el CAMBIO",
                  color=ARCA_DARK, fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dif_intuicion.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_dif_intuicion.png")
    plt.close(fig)

def fig_dif_airpass():
    """
    Aplicar diferenciacion paso a paso sobre AirPassengers.
    Cada paso ataca una patologia distinta y baja el ADF p.
    """
    y = y_ap.copy()
    y_log = np.log(y)
    y_d1 = y_log.diff().dropna()
    y_d1_d12 = y_d1.diff(12).dropna()

    p_o = adfuller(y.dropna())[1]
    p_l = adfuller(y_log.dropna())[1]
    p_d = adfuller(y_d1)[1]
    p_ds = adfuller(y_d1_d12)[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    def _panel(ax, s, ttl, color, hline0=False):
        ax.plot(s.index, s.values, color=color, lw=1.1)
        ax.set_title(ttl, color=ARCA_DARK, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if hline0:
            ax.axhline(0, color="black", lw=0.5)

    _panel(axes[0,0], y,
           f"(1) Original  -  ADF p = {p_o:.3f}\nNivel sube + estacionalidad + varianza creciente",
           ARCA_DARK)
    _panel(axes[0,1], y_log,
           f"(2) log(y)  -  ADF p = {p_l:.3f}\nlog estabiliza la VARIANZA",
           ARCA_BLUE)
    _panel(axes[1,0], y_d1,
           f"(3) delta log(y)  -  ADF p = {p_d:.3g}\ndelta quita la TENDENCIA",
           ARCA_GREEN, hline0=True)
    _panel(axes[1,1], y_d1_d12,
           f"(4) delta_12 delta log(y)  -  ADF p = {p_ds:.3g}\ndelta_12 quita la ESTACIONALIDAD",
           ARCA_RED, hline0=True)

    fig.suptitle("Diferenciacion como cirugia --- cada paso ataca una patologia distinta\n"
                  "(aplicada a AirPassengers, dataset clasico)",
                  color=ARCA_DARK, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dif_airpass.png"),
                dpi=140, bbox_inches="tight")
    print(f"  fig_dif_airpass.png  p: orig={p_o:.3f} log={p_l:.3f} d1={p_d:.3g} d1d12={p_ds:.3g}")
    plt.close(fig)
    return p_o, p_l, p_d, p_ds


# =============================================================================
#  HELPERS: modelos
# =============================================================================
def fit_naive(y_tr, h):
    return np.full(h, y_tr.iloc[-1])

def fit_seasonal_naive(y_tr, h, season):
    last = y_tr.iloc[-season:].values
    return np.tile(last, (h // season) + 1)[:h]

def fit_ma(y_tr, h, window):
    return np.full(h, y_tr.iloc[-window:].mean())

_PROPHET_CACHE = {}
def fit_prophet(y_tr, h, freq, country=None, key=""):
    """Prophet con holidays opcionales. freq='MS' para mensual, 'W' para semanal."""
    from prophet import Prophet
    import holidays as hol

    hdays = None
    if country:
        years = list(range(y_tr.index.year.min(), y_tr.index.year.max() + 5))
        hd = hol.country_holidays(country, years=years)
        hdays = pd.DataFrame({
            "holiday": list(hd.values()),
            "ds": pd.to_datetime(list(hd.keys())),
            "lower_window": 0, "upper_window": 1,
        })

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, holidays=hdays,
                interval_width=0.80, changepoint_prior_scale=0.05)
    m.fit(pd.DataFrame({"ds": y_tr.index, "y": y_tr.values}))
    future = m.make_future_dataframe(periods=h, freq=freq)
    fc = m.predict(future)
    _PROPHET_CACHE[key] = (m, fc)
    fc_tail = fc.iloc[-h:]
    return (m, fc, fc_tail["yhat"].values,
            fc_tail["yhat_lower"].values, fc_tail["yhat_upper"].values)


# =============================================================================
#  BLOQUE 3 - Prophet sobre AirPassengers
# =============================================================================
def fig_airpass_intro():
    """La serie clasica con marcas de train/test."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(y_ap_train.index, y_ap_train.values, color=ARCA_DARK, lw=1.4,
             label=f"train ({len(y_ap_train)} meses)")
    ax.plot(y_ap_test.index, y_ap_test.values, color="lightgray", lw=1.4,
             label=f"test ({len(y_ap_test)} meses, no lo ve el modelo)")
    ax.axvspan(y_ap_test.index[0], y_ap_test.index[-1], color=ARCA_RED, alpha=0.10)
    ax.axvline(y_ap_test.index[0], color=ARCA_RED, lw=1.5, linestyle="--")
    ax.set_title("AirPassengers --- pasajeros aereos mensuales 1949-1960\n"
                  "(serie canonica con tendencia + estacionalidad + varianza creciente)",
                  color=ARCA_DARK)
    ax.set_xlabel("anio"); ax.set_ylabel("miles de pasajeros")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_airpass_intro.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_airpass_intro.png")
    plt.close(fig)

def fig_prophet_forecast_airpass():
    """Prophet forecast sobre AirPassengers - SIN holidays (no aporta a esta serie)."""
    m, fc, fc_t, lo, hi = fit_prophet(y_ap_train, AP_HORIZON, freq="MS",
                                       country=None, key="airpass")
    mae = float(np.mean(np.abs(y_ap_test.values - fc_t)))
    mape = float(np.mean(np.abs((y_ap_test.values - fc_t) / y_ap_test.values))) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_ap_train.index, y_ap_train.values, color="black", lw=1, alpha=0.7,
             label="train")
    ax.plot(y_ap_test.index, y_ap_test.values, color="black", lw=2, marker="o",
             markersize=4, label="real (test)")
    ax.plot(y_ap_test.index, fc_t, color=ARCA_GREEN, lw=2, marker="s",
             markersize=4, label=f"Prophet  MAE={mae:.0f}  MAPE={mape:.1f}%")
    ax.fill_between(y_ap_test.index, lo, hi, color=ARCA_GREEN, alpha=0.20,
                     label="intervalo 80%")
    ax.axvline(y_ap_train.index[-1], color="gray", ls=":")
    ax.set_title("Prophet sobre AirPassengers --- forecast de 24 meses",
                  color=ARCA_DARK)
    ax.set_xlabel("anio"); ax.set_ylabel("miles de pasajeros")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_prophet_forecast_airpass.png"),
                dpi=140, bbox_inches="tight")
    print(f"  fig_prophet_forecast_airpass.png  MAE={mae:.0f}  MAPE={mape:.1f}%")
    plt.close(fig)
    return fc_t, lo, hi, mae, mape

def fig_prophet_components_airpass():
    """Que aprendio Prophet: trend + yearly seasonality."""
    m, fc = _PROPHET_CACHE["airpass"]
    fig = m.plot_components(fc)
    fig.set_size_inches(11, 6)
    fig.suptitle("Lo que Prophet aprendio --- componentes interpretables",
                  color=ARCA_DARK, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_prophet_components_airpass.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_prophet_components_airpass.png")
    plt.close(fig)


# =============================================================================
#  BLOQUE 4 - Metricas y walk-forward
# =============================================================================
def fig_walkforward():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_total = 36
    folds = [(0, 24, 24, 30),
             (0, 30, 30, 36),
             (0, 18, 18, 24)]
    folds = [(0, 18, 18, 24),
             (0, 24, 24, 30),
             (0, 30, 30, 36)]
    for i, (ts, te, vs, ve) in enumerate(folds):
        yy = -i
        ax.barh(yy, te-ts, left=ts, color=ARCA_BLUE, alpha=0.85, height=0.6,
                 label="train (todo el pasado)" if i==0 else None)
        ax.barh(yy, ve-vs, left=vs, color=ARCA_RED, alpha=0.95, height=0.6,
                 label="test (horizonte del forecast)" if i==0 else None)
        ax.text(-1, yy, f"fold {i+1}", va="center", ha="right",
                 fontsize=10, color=ARCA_DARK)
    ax.set_xlim(-3, n_total+1); ax.set_yticks([])
    ax.set_xlabel("tiempo")
    ax.set_title("Walk-forward CV  -  train siempre crece, test siempre futuro\n"
                  "(asi simulas como usarias el modelo en produccion)",
                  color=ARCA_DARK)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_walkforward.png"), dpi=140, bbox_inches="tight")
    print("  fig_walkforward.png")
    plt.close(fig)

def fig_metricas_cuando():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    n = 50
    rng = np.random.RandomState(1)
    err_a = rng.randn(n)*5
    err_b = np.zeros(n); err_b[10] = 50; err_b[35] = -50
    mae_a = float(np.mean(np.abs(err_a))); rmse_a = float(np.sqrt(np.mean(err_a**2)))
    mae_b = float(np.mean(np.abs(err_b))); rmse_b = float(np.sqrt(np.mean(err_b**2)))
    axes[0].plot(err_a, color=ARCA_BLUE, lw=1.4, label=f"A: MAE={mae_a:.1f} RMSE={rmse_a:.1f}")
    axes[0].plot(err_b, color=ARCA_RED,    lw=1.4, label=f"B: MAE={mae_b:.1f} RMSE={rmse_b:.1f}")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_title("MAE vs RMSE\nRMSE castiga outliers", color=ARCA_DARK, fontsize=11)
    axes[0].set_xlabel("dia"); axes[0].set_ylabel("error")
    axes[0].legend(loc="upper right", fontsize=9)

    y_true_s = np.array([10., 12, 11, 13, 14, 12, 10])
    y_pred_s = np.array([11., 13, 10, 14, 12, 13, 11])
    y_true_b = np.array([1000., 1200, 1100, 1300, 1400, 1200, 1000])
    y_pred_b = np.array([1100., 1300, 1000, 1400, 1200, 1300, 1100])
    mape_s = float(np.mean(np.abs((y_true_s - y_pred_s)/y_true_s))*100)
    mape_b = float(np.mean(np.abs((y_true_b - y_pred_b)/y_true_b))*100)
    x = np.arange(7)
    ax_l = axes[1]
    ax_l.plot(x, y_true_s, "o-", color=ARCA_BLUE, label="SKU chico")
    ax_l.plot(x, y_pred_s, "x--", color=ARCA_BLUE, alpha=0.5)
    ax_r = ax_l.twinx()
    ax_r.plot(x, y_true_b, "o-", color=ARCA_RED, label="SKU grande")
    ax_r.plot(x, y_pred_b, "x--", color=ARCA_RED, alpha=0.5)
    ax_r.grid(False)
    ax_l.set_title(f"MAPE compara escalas\nchico MAPE={mape_s:.1f}%  grande MAPE={mape_b:.1f}%",
                    color=ARCA_DARK, fontsize=11)
    ax_l.set_xlabel("dia")
    ax_l.legend(loc="upper left", fontsize=9); ax_r.legend(loc="lower right", fontsize=9)

    y_true = np.array([100., 50, 10, 5, 1, 0.5])
    y_pred = y_true + 1
    mape_per = np.abs((y_true - y_pred)/y_true)*100
    axes[2].bar(range(len(y_true)), mape_per, color=ARCA_RED)
    axes[2].set_xticks(range(len(y_true)))
    axes[2].set_xticklabels([f"{v:g}" for v in y_true])
    axes[2].set_xlabel("valor real"); axes[2].set_ylabel("MAPE (%) con error=+1")
    axes[2].set_title("MAPE explota cerca de 0\n--> usa WAPE en portafolios",
                       color=ARCA_DARK, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_metricas_cuando.png"), dpi=140, bbox_inches="tight")
    print("  fig_metricas_cuando.png")
    plt.close(fig)

def fig_comparacion_airpass(fc_prophet):
    """Comparar Prophet con baselines simples sobre AirPassengers."""
    pred_naive   = fit_naive(y_ap_train, AP_HORIZON)
    pred_snaive  = fit_seasonal_naive(y_ap_train, AP_HORIZON, season=12)
    pred_ma      = fit_ma(y_ap_train, AP_HORIZON, window=12)

    y_true = y_ap_test.values
    def mae(p): return float(np.mean(np.abs(y_true - p)))
    def wape(p): return float(np.sum(np.abs(y_true - p))/np.sum(y_true)*100)

    rows = [
        ("Naive (ultimo valor)",   pred_naive,   "#94A3B8"),
        ("Seasonal naive (12 m)",  pred_snaive,  ARCA_BLUE),
        ("Media movil 12 meses",   pred_ma,      "#7C3AED"),
        ("Prophet",                fc_prophet,   ARCA_GREEN),
    ]
    maes = [mae(p) for _, p, _ in rows]
    wapes = [wape(p) for _, p, _ in rows]
    baseline = maes[0]
    lifts = [(baseline - m_)/baseline*100 for m_ in maes]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                              gridspec_kw={"width_ratios":[2.4, 1]})

    hist_tail = y_ap_train.iloc[-24:]
    axes[0].plot(hist_tail.index, hist_tail.values, color="black", lw=1.0, alpha=0.6,
                  label="historico")
    axes[0].plot(y_ap_test.index, y_true, color="black", lw=2,
                  marker="o", markersize=5, label="real")
    for name, p, c in rows:
        axes[0].plot(y_ap_test.index, p, color=c, lw=1.6, alpha=0.85,
                      label=name, marker="x", markersize=4)
    axes[0].axvline(y_ap_train.index[-1], color="gray", ls=":")
    axes[0].set_title("Cuatro modelos sobre los mismos 24 meses (AirPassengers)",
                       color=ARCA_DARK, fontsize=12)
    axes[0].set_xlabel("anio"); axes[0].set_ylabel("miles de pasajeros")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    yy = np.arange(len(rows))[::-1]
    colors = [c for _, _, c in rows]
    bars = ax2.barh(yy, lifts, color=colors, edgecolor=ARCA_DARK, alpha=0.85)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f"{n}\nMAE={m_:.0f}  WAPE={w_:.1f}%"
                          for (n, _, _), m_, w_ in zip(rows, maes, wapes)],
                         fontsize=9)
    ax2.axvline(0, color="black", lw=0.6)
    ax2.set_xlabel("lift sobre baseline naive (%)")
    ax2.set_title("Mejora % vs naive (mas alto = mejor)",
                   color=ARCA_DARK, fontsize=12)
    for bar, lift in zip(bars, lifts):
        w_ = bar.get_width()
        ax2.text(w_ + (1 if w_ >= 0 else -1), bar.get_y() + bar.get_height()/2,
                  f"{lift:+.0f}%", va="center",
                  ha="left" if w_ >= 0 else "right", fontsize=9, weight="bold",
                  color=ARCA_DARK)
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_comparacion_airpass.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_comparacion_airpass.png")
    for (n, _, _), m_, w_, l_ in zip(rows, maes, wapes, lifts):
        print(f"      {n:30s}  MAE={m_:7.0f}  WAPE={w_:5.1f}%  lift={l_:+5.1f}%")
    plt.close(fig)
    return rows, maes, wapes, lifts


# =============================================================================
#  BLOQUE 5 - Inferencia + costo asimetrico
# =============================================================================
def fig_inferencia_intervalos_airpass(fc_central, lo, hi):
    """Visual del rango: prediccion no es punto, es banda."""
    fig, ax = plt.subplots(figsize=(11, 5))
    hist_tail = y_ap_train.iloc[-24:]
    ax.plot(hist_tail.index, hist_tail.values, color="black", lw=1.0, alpha=0.7,
             label="historico")
    ax.plot(y_ap_test.index, y_ap_test.values, color="black", lw=2, marker="o",
             markersize=5, label="real (no la vemos al planear)")
    ax.plot(y_ap_test.index, fc_central, color=ARCA_GREEN, lw=2, marker="s",
             markersize=5, label="punto central (P50)")
    ax.fill_between(y_ap_test.index, lo, hi, color=ARCA_GREEN, alpha=0.22,
                     label="intervalo 80%  (P10 a P90)")
    ax.axvline(y_ap_train.index[-1], color="gray", ls=":", lw=1)
    ax.set_title("La prediccion NO es un numero --- es un RANGO",
                  color=ARCA_DARK, fontsize=12)
    ax.set_xlabel("anio"); ax.set_ylabel("miles de pasajeros")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_inferencia_intervalos.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_inferencia_intervalos.png")
    plt.close(fig)

def fig_costo_asimetrico(fc_central, lo, hi, ylabel="cajas a despachar"):
    """Concepto: percentiles y tabla de tradeoffs."""
    from scipy.stats import norm
    sigma = (hi - lo) / (2 * 1.28)
    p50 = fc_central
    p70 = fc_central + norm.ppf(0.70) * sigma
    p80 = fc_central + norm.ppf(0.80) * sigma
    p90 = fc_central + norm.ppf(0.90) * sigma

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    x = np.arange(len(fc_central))
    axes[0].plot(x, fc_central + sigma*0, color="black", lw=0, label="real")  # placeholder
    axes[0].plot(x, p50, color=ARCA_GREEN, lw=1.8, ls="--",
                  marker="s", markersize=4, label="P50  (50% chance de stockout)")
    axes[0].plot(x, p70, color=ARCA_BLUE, lw=1.8,
                  marker="^", markersize=5, label="P70  (30% stockout)")
    axes[0].plot(x, p80, color=ARCA_ORANGE, lw=1.8,
                  marker="D", markersize=5, label="P80  (20% stockout)")
    axes[0].plot(x, p90, color=ARCA_RED, lw=1.8,
                  marker="v", markersize=5, label="P90  (10% stockout)")
    axes[0].set_title("Cual percentil entregar?  El costo asimetrico decide",
                       color=ARCA_DARK, fontsize=12)
    axes[0].set_xlabel("paso del horizonte"); axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper left", fontsize=9)

    ax = axes[1]; ax.axis("off")
    table_data = [
        ["Percentil", "Riesgo\nstockout", "Capital\natado", "Cuando usarlo"],
        ["P50",  "50%", "minimo",  "costos simetricos\n(raro)"],
        ["P70",  "30%", "+10%",    "default razonable"],
        ["P80",  "20%", "+20%",    "stockout 3x mas\ncaro que vencido"],
        ["P90",  "10%", "+35%",    "stockout es\ncatastrofico"],
    ]
    tbl = ax.table(cellText=table_data, loc="center", cellLoc="center",
                    colWidths=[0.16, 0.20, 0.18, 0.30])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(4):
        tbl[(0, j)].set_facecolor(ARCA_DARK); tbl[(0, j)].set_text_props(color="white", weight="bold")
    colors_rows = [ARCA_GREEN, ARCA_BLUE, ARCA_ORANGE, ARCA_RED]
    for i in range(1, 5):
        tbl[(i, 0)].set_facecolor(colors_rows[i-1]); tbl[(i, 0)].set_text_props(color="white", weight="bold")
    ax.set_title("Como elegir el percentil", color=ARCA_DARK, fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_costo_asimetrico.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_costo_asimetrico.png")
    plt.close(fig)
    return p50, p70, p80, p90


# =============================================================================
#  BLOQUE 6 - Caso aplicado a Favorita (6 meses adelante)
# =============================================================================
def fig_favorita_problema():
    """La serie completa de Favorita, marcando el horizonte de prediccion."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(y_fav_train.index, y_fav_train.values, color=ARCA_DARK, lw=1, alpha=0.9,
             label="historico (train)")
    ax.plot(y_fav_test.index, y_fav_test.values, color="lightgray", lw=1,
             label="6 meses siguientes (no los ve el modelo)")
    ax.axvspan(y_fav_test.index[0], y_fav_test.index[-1], color=ARCA_RED, alpha=0.10)
    ax.axvline(y_fav_test.index[0], color=ARCA_RED, ls="--", lw=1.5)
    ax.text(y_fav_test.index[0], y_w.max()*0.97,
             f"  horizonte: 6 meses\n  ({FAV_HORIZON} semanas)",
             color=ARCA_RED, fontsize=11, va="top", weight="bold")
    ax.set_title("Caso aplicado --- Favorita Quito Q44 bebidas\n"
                  "El planeador necesita el numero del proximo trimestre + bimestre",
                  color=ARCA_DARK)
    ax.set_xlabel("fecha"); ax.set_ylabel("ventas semanales (cajas)")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_favorita_problema.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_favorita_problema.png")
    plt.close(fig)

def fig_favorita_prophet():
    """Prophet sobre Favorita con holidays Ecuador, forecast 6 meses."""
    m, fc, fc_t, lo, hi = fit_prophet(y_fav_train, FAV_HORIZON, freq="W",
                                       country="EC", key="favorita")
    mae = float(np.mean(np.abs(y_fav_test.values - fc_t)))
    mape = float(np.mean(np.abs((y_fav_test.values - fc_t) / y_fav_test.values))) * 100
    wape = float(np.sum(np.abs(y_fav_test.values - fc_t)) / np.sum(y_fav_test.values)) * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    hist_tail = y_fav_train.iloc[-52:]
    ax.plot(hist_tail.index, hist_tail.values, color=ARCA_DARK, lw=1, alpha=0.8,
             label="historico (1 ano previo)")
    ax.plot(y_fav_test.index, y_fav_test.values, color="black", lw=1.5,
             label="real (test 26 sem)", marker="o", markersize=4)
    ax.plot(y_fav_test.index, fc_t, color=ARCA_GREEN, lw=2,
             label=f"Prophet (con holidays EC)  MAE={mae:.0f}  MAPE={mape:.1f}%  WAPE={wape:.1f}%",
             marker="s", markersize=4)
    ax.fill_between(y_fav_test.index, lo, hi, color=ARCA_GREEN, alpha=0.20,
                     label="intervalo 80%")
    ax.axvline(y_fav_train.index[-1], color="gray", ls=":")
    ax.set_title("Prophet sobre Favorita --- forecast 6 meses (con holidays Ecuador)",
                  color=ARCA_DARK)
    ax.set_xlabel("fecha"); ax.set_ylabel("ventas semanales (cajas)")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_favorita_prophet.png"),
                dpi=140, bbox_inches="tight")
    print(f"  fig_favorita_prophet.png  MAE={mae:.0f}  MAPE={mape:.1f}%  WAPE={wape:.1f}%")
    plt.close(fig)
    return fc_t, lo, hi, mae, mape, wape

def fig_favorita_components():
    m, fc = _PROPHET_CACHE["favorita"]
    fig = m.plot_components(fc)
    fig.set_size_inches(11, 7)
    fig.suptitle("Lo que Prophet aprendio en Favorita --- trend + estacionalidad + holidays",
                  color=ARCA_DARK, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_favorita_components.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_favorita_components.png")
    plt.close(fig)


# =============================================================================
#  BLOQUE 7 - Cuando Prophet falla (Mackey-Glass como "sensor industrial")
# =============================================================================
def mackey_glass(n=2000, tau=17, gamma=0.1, beta=0.2, p=10, dt=1.0, x0=1.2, burn=200):
    """
    Mackey-Glass: ecuacion diferencial con retardo, caotica para tau=17.
    Es el benchmark canonico para demostrar RNN/LSTM sobre series no lineales.
    Aqui la usamos como "sensor industrial con dinamica interna".
    """
    n_total = n + burn
    x = np.full(n_total, x0)
    for t in range(1, n_total):
        x_tau = x[t-tau] if t > tau else x0
        x[t] = x[t-1] + dt * (beta * x_tau / (1 + x_tau**p) - gamma * x[t-1])
    return x[burn:]

# Generar la serie del sensor industrial (sintetica pero deterministica)
print("\nGenerando serie 'sensor industrial' (Mackey-Glass)...")
SENSOR_N = 1500
sensor_raw = mackey_glass(n=SENSOR_N, tau=17)
# Anclar la serie a un calendario para que Prophet la pueda comer
sensor_dates = pd.date_range("2020-01-01", periods=SENSOR_N, freq="h")
y_sensor = pd.Series(sensor_raw, index=sensor_dates, name="presion")
print(f"  Serie sensor: {len(y_sensor)} puntos horarios, rango [{y_sensor.min():.2f}, {y_sensor.max():.2f}]")

# Split: 70% train, 30% test
SENSOR_HORIZON = 300   # 300 puntos (12.5 dias horarios) de test
y_sensor_train = y_sensor.iloc[:-SENSOR_HORIZON]
y_sensor_test  = y_sensor.iloc[-SENSOR_HORIZON:]

def fig_sensor_intro():
    """Presenta la serie del sensor industrial."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(y_sensor_train.index, y_sensor_train.values, color=ARCA_DARK, lw=0.6,
             label=f"historico ({len(y_sensor_train)} mediciones horarias)")
    ax.plot(y_sensor_test.index, y_sensor_test.values, color="lightgray", lw=0.7,
             label=f"futuro ({len(y_sensor_test)} mediciones, no lo ve el modelo)")
    ax.axvspan(y_sensor_test.index[0], y_sensor_test.index[-1], color=ARCA_RED, alpha=0.10)
    ax.axvline(y_sensor_test.index[0], color=ARCA_RED, ls="--", lw=1.5)
    ax.set_title("Nuevo problema: sensor de presion en una caldera industrial\n"
                  "(serie horaria, dinamica interna NO lineal)",
                  color=ARCA_DARK)
    ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sensor_intro.png"), dpi=140, bbox_inches="tight")
    print("  fig_sensor_intro.png")
    plt.close(fig)

def fig_sensor_zoom():
    """Zoom para ver el patron oscilatorio caotico."""
    sub = y_sensor.iloc[200:600]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(sub.index, sub.values, color=ARCA_DARK, lw=1)
    ax.set_title("Zoom: el valor de hoy depende NO LINEALMENTE de las mediciones pasadas\n"
                  "(no hay tendencia clara, no hay estacionalidad fija)",
                  color=ARCA_DARK, fontsize=11)
    ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sensor_zoom.png"), dpi=140, bbox_inches="tight")
    print("  fig_sensor_zoom.png")
    plt.close(fig)

_PROPHET_SENSOR = None
def fig_sensor_prophet_falla():
    """Prophet sobre el sensor: tendencia + estacionalidad no capturan el caos."""
    global _PROPHET_SENSOR
    from prophet import Prophet
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=True, interval_width=0.80,
                changepoint_prior_scale=0.05)
    m.fit(pd.DataFrame({"ds": y_sensor_train.index, "y": y_sensor_train.values}))
    future = m.make_future_dataframe(periods=SENSOR_HORIZON, freq="h")
    fc = m.predict(future)
    fc_t = fc["yhat"].iloc[-SENSOR_HORIZON:].values
    lo = fc["yhat_lower"].iloc[-SENSOR_HORIZON:].values
    hi = fc["yhat_upper"].iloc[-SENSOR_HORIZON:].values
    _PROPHET_SENSOR = (m, fc, fc_t, lo, hi)

    mae = float(np.mean(np.abs(y_sensor_test.values - fc_t)))
    rmse = float(np.sqrt(np.mean((y_sensor_test.values - fc_t)**2)))
    mape = float(np.mean(np.abs((y_sensor_test.values - fc_t) / y_sensor_test.values))) * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    hist_tail = y_sensor_train.iloc[-300:]
    ax.plot(hist_tail.index, hist_tail.values, color=ARCA_DARK, lw=0.8, alpha=0.7,
             label="historico")
    ax.plot(y_sensor_test.index, y_sensor_test.values, color="black", lw=1.4,
             label="real")
    ax.plot(y_sensor_test.index, fc_t, color=ARCA_RED, lw=1.5,
             label=f"Prophet  MAE={mae:.3f}  MAPE={mape:.1f}%")
    ax.fill_between(y_sensor_test.index, lo, hi, color=ARCA_RED, alpha=0.15,
                     label="intervalo 80%")
    ax.axvline(y_sensor_train.index[-1], color="gray", ls=":")
    ax.set_title("Prophet sobre el sensor: forecast PLANO, no captura la dinamica",
                  color=ARCA_DARK)
    ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sensor_prophet_falla.png"),
                dpi=140, bbox_inches="tight")
    print(f"  fig_sensor_prophet_falla.png  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    plt.close(fig)
    return fc_t, mae, mape

def fig_por_que_prophet_falla():
    """Diagrama conceptual: Prophet asume aditivo tendencia + estacionalidad."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sub = y_sensor.iloc[200:500]
    axes[0].plot(sub.index, sub.values, color=ARCA_DARK, lw=1)
    axes[0].set_title("La realidad:\nvalor_t depende NO LINEALMENTE de pasados (lags 17)",
                       color=ARCA_DARK, fontsize=11)
    axes[0].set_xlabel("fecha"); axes[0].set_ylabel("presion")

    # Diagrama conceptual
    ax = axes[1]; ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9, "Prophet asume:", ha="center", fontsize=13, weight="bold", color=ARCA_DARK)
    ax.text(5, 7.5, r"$y(t) \approx \mathrm{tendencia}(t) + \mathrm{estacion}(t) + \mathrm{holidays}(t)$",
             ha="center", fontsize=12, color=ARCA_BLUE)
    ax.text(5, 5.5, "Es una funcion del TIEMPO solamente.\n"
                     "No mira los valores pasados de la propia serie.",
             ha="center", fontsize=11, color=ARCA_DARK)
    ax.text(5, 3, "El sensor REQUIERE:", ha="center", fontsize=13, weight="bold", color=ARCA_DARK)
    ax.text(5, 1.7, r"$y(t) = f(y(t-1), y(t-2), \ldots, y(t-k))$",
             ha="center", fontsize=12, color=ARCA_RED)
    ax.text(5, 0.5, "Necesitamos un modelo con MEMORIA y no-linealidad.",
             ha="center", fontsize=11, weight="bold", color=ARCA_DARK)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_por_que_prophet_falla.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_por_que_prophet_falla.png")
    plt.close(fig)


# =============================================================================
#  BLOQUE 8 - LSTM al rescate
# =============================================================================
def fig_lstm_ventanas():
    """Visual de como se arman ventanas X -> y para entrenar la LSTM."""
    n = 20
    s = mackey_glass(n=n+40, tau=17)[40:]
    W = 6
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(n)
    ax.plot(x, s, "o-", color="lightgray", markersize=6, lw=1)

    # Resaltar ejemplo: ventana de tamaño W prediciendo el siguiente
    for i, (start, color) in enumerate([(2, ARCA_BLUE), (7, ARCA_GREEN), (12, ARCA_ORANGE)]):
        ax.plot(x[start:start+W], s[start:start+W], "o-", color=color, lw=2, markersize=8)
        ax.plot(x[start+W], s[start+W], marker="*", markersize=20, color=color)
        ax.annotate(f"ventana {i+1}", xy=(start+W/2-0.5, s[start:start+W].max()+0.1),
                     fontsize=9, color=color, weight="bold", ha="center")

    ax.set_title(f"Como entrenar LSTM: cada ventana de {W} pasos --> proximo valor\n"
                  "El modelo aprende: 'dadas estas {W} mediciones, cual es la siguiente?'".format(W=W),
                  color=ARCA_DARK)
    ax.set_xlabel("paso temporal"); ax.set_ylabel("valor")
    ax.legend(["serie", "ventana (input)", "objetivo (output)"], loc="upper right",
               fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_lstm_ventanas.png"), dpi=140, bbox_inches="tight")
    print("  fig_lstm_ventanas.png")
    plt.close(fig)

_LSTM_RESULTS = None
def fit_lstm_sensor(window=50, units=32, epochs=20):
    """Entrena una LSTM minima sobre el sensor. Devuelve forecast del test."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(42)

    # Normalizar a [0,1] con stats del train
    s = y_sensor.values.astype(np.float32)
    train_n = len(y_sensor_train)
    mn, mx = s[:train_n].min(), s[:train_n].max()
    s_norm = (s - mn) / (mx - mn + 1e-9)

    def make_windows(arr, w):
        X, y = [], []
        for i in range(len(arr) - w):
            X.append(arr[i:i+w])
            y.append(arr[i+w])
        return np.array(X)[..., None], np.array(y)

    X_all, y_all = make_windows(s_norm, window)
    # Las filas 0 .. (train_n - window - 1) corresponden a objetivos en train
    n_train_windows = train_n - window
    X_train, y_train_arr = X_all[:n_train_windows], y_all[:n_train_windows]

    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(units),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    hist = model.fit(X_train, y_train_arr, epochs=epochs, batch_size=64,
                      validation_split=0.1, verbose=0)

    # Forecast del test: rolling, usando los valores REALES (1-step ahead)
    # Esto es honesto: en produccion el sensor sigue midiendo, no necesitas
    # predecir multi-step ciegamente para evaluar el modelo.
    test_idx_start = train_n
    fc = []
    for t in range(test_idx_start, len(s_norm)):
        window_arr = s_norm[t-window:t].reshape(1, window, 1)
        pred_norm = model.predict(window_arr, verbose=0)[0, 0]
        fc.append(pred_norm * (mx - mn) + mn)
    fc = np.array(fc)
    return fc, hist

def fig_lstm_forecast():
    global _LSTM_RESULTS
    fc, hist = fit_lstm_sensor(window=50, units=32, epochs=20)
    mae = float(np.mean(np.abs(y_sensor_test.values - fc)))
    rmse = float(np.sqrt(np.mean((y_sensor_test.values - fc)**2)))
    mape = float(np.mean(np.abs((y_sensor_test.values - fc) / y_sensor_test.values))) * 100
    _LSTM_RESULTS = {"fc": fc, "mae": mae, "rmse": rmse, "mape": mape, "history": hist}

    fig, ax = plt.subplots(figsize=(13, 5))
    hist_tail = y_sensor_train.iloc[-300:]
    ax.plot(hist_tail.index, hist_tail.values, color=ARCA_DARK, lw=0.8, alpha=0.7,
             label="historico")
    ax.plot(y_sensor_test.index, y_sensor_test.values, color="black", lw=1.4,
             label="real")
    ax.plot(y_sensor_test.index, fc, color=ARCA_GREEN, lw=1.5,
             label=f"LSTM (50 lags)  MAE={mae:.3f}  MAPE={mape:.1f}%")
    ax.axvline(y_sensor_train.index[-1], color="gray", ls=":")
    ax.set_title("LSTM sobre el sensor: captura la dinamica no lineal",
                  color=ARCA_DARK)
    ax.set_xlabel("fecha"); ax.set_ylabel("presion (bar)")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_lstm_forecast.png"),
                dpi=140, bbox_inches="tight")
    print(f"  fig_lstm_forecast.png  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    plt.close(fig)
    return fc, mae, mape

def fig_lstm_vs_prophet(fc_prophet, mae_prophet, mape_prophet,
                        fc_lstm, mae_lstm, mape_lstm):
    """Comparacion lado a lado en la misma serie."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                              gridspec_kw={"width_ratios":[2.5, 1]})

    test_idx = y_sensor_test.index
    real = y_sensor_test.values
    sub_idx = test_idx[:200]   # primeros 200 puntos para no saturar
    sub_real = real[:200]; sub_pro = fc_prophet[:200]; sub_lstm = fc_lstm[:200]

    axes[0].plot(sub_idx, sub_real, color="black", lw=1.5, label="real")
    axes[0].plot(sub_idx, sub_pro, color=ARCA_RED, lw=1.3,
                  label=f"Prophet  MAE={mae_prophet:.3f}  MAPE={mape_prophet:.1f}%")
    axes[0].plot(sub_idx, sub_lstm, color=ARCA_GREEN, lw=1.3,
                  label=f"LSTM     MAE={mae_lstm:.3f}  MAPE={mape_lstm:.1f}%")
    axes[0].set_title("Misma serie, dos modelos --- primeros 200 puntos del test",
                       color=ARCA_DARK, fontsize=12)
    axes[0].set_xlabel("fecha"); axes[0].set_ylabel("presion")
    axes[0].legend(loc="upper left", fontsize=10)

    ax = axes[1]
    metrics = ["MAE", "RMSE", "MAPE (%)"]
    pro_vals = [mae_prophet, float(np.sqrt(np.mean((real-fc_prophet)**2))), mape_prophet]
    lstm_vals = [mae_lstm, float(np.sqrt(np.mean((real-fc_lstm)**2))), mape_lstm]
    x_pos = np.arange(len(metrics)); width = 0.35
    ax.bar(x_pos - width/2, pro_vals, width, color=ARCA_RED, label="Prophet")
    ax.bar(x_pos + width/2, lstm_vals, width, color=ARCA_GREEN, label="LSTM")
    for i, (p, l) in enumerate(zip(pro_vals, lstm_vals)):
        ax.text(i - width/2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, l, f"{l:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos); ax.set_xticklabels(metrics)
    ax.set_title("Metricas lado a lado", color=ARCA_DARK, fontsize=12)
    ax.legend()
    ax.grid(axis="x", alpha=0)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_lstm_vs_prophet.png"),
                dpi=140, bbox_inches="tight")
    print("  fig_lstm_vs_prophet.png")
    plt.close(fig)


# =============================================================================
#  BLOQUE 9 - Cierre: decision tree
# =============================================================================
def fig_decision_tree():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, fc, fg="white", fs=10):
        ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=fc,
                                     edgecolor=ARCA_DARK, lw=1.3))
        ax.text(x, y, text, ha="center", va="center", color=fg,
                 fontsize=fs, weight="bold")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.5))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label, fontsize=9,
                     color=ARCA_DARK, style="italic")

    box(5, 9.0, 6, 1.0, "Tengo una serie temporal --- que modelo uso?", ARCA_DARK, fs=12)

    box(2.5, 7.0, 3.0, 0.9, "Pocos datos\n(menos de 2 ciclos)?", ARCA_BLUE, fs=10)
    box(7.5, 7.0, 3.0, 0.9, "Patron CLASICO\n(tendencia + estacionalidad)?", ARCA_BLUE, fs=10)
    arrow(5, 8.5, 2.5, 7.5); arrow(5, 8.5, 7.5, 7.5)

    box(2.5, 5.0, 3.0, 0.9, "Naive / Seasonal naive\n/ Media movil", ARCA_GREEN, fs=10)
    box(7.5, 5.0, 3.0, 0.9, "Prophet\n(con holidays!)", ARCA_GREEN, fs=10)
    arrow(2.5, 6.5, 2.5, 5.5, "si"); arrow(7.5, 6.5, 7.5, 5.5, "si")

    box(5, 3.0, 6, 1.0, "Patron NO LINEAL / cambio de regimen /\nmuchos drivers externos?", ARCA_RED, fs=11)
    arrow(2.5, 4.5, 4, 3.5); arrow(7.5, 4.5, 6, 3.5)

    box(5, 1.2, 6, 1.0, "REDES NEURONALES  (clase 33)\nLSTM, Transformer, NeuralProphet, XGBoost+lags",
        ARCA_DARK, fs=11)
    arrow(5, 2.5, 5, 1.8, "si")

    ax.text(5, 0.2, "Empieza simple. Sube de complejidad solo si lo simple ya no da.",
             ha="center", color=ARCA_DARK, fontsize=11, style="italic")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_decision_tree.png"), dpi=140, bbox_inches="tight")
    print("  fig_decision_tree.png")
    plt.close(fig)


if __name__ == "__main__":
    print("\n=== Bloque 0 ===")
    fig_problema_planeacion()

    print("\n=== Bloque 1 - Estacionariedad ===")
    fig_dos_series_descubrimiento()
    fig_dos_series_respuesta()
    fig_modelo_falla()

    print("\n=== Bloque 2 - Diferenciacion ===")
    fig_dif_intuicion()
    fig_dif_airpass()

    print("\n=== Bloque 3 - Prophet sobre AirPassengers ===")
    fig_airpass_intro()
    fc_t, lo, hi, mae, mape = fig_prophet_forecast_airpass()
    fig_prophet_components_airpass()

    print("\n=== Bloque 4 - Metricas ===")
    fig_walkforward()
    fig_metricas_cuando()
    fig_comparacion_airpass(fc_t)

    print("\n=== Bloque 5 - Inferencia + costo asimetrico ===")
    fig_inferencia_intervalos_airpass(fc_t, lo, hi)
    fig_costo_asimetrico(fc_t, lo, hi, ylabel="miles de pasajeros")

    print("\n=== Bloque 6 - Aplicacion a Favorita ===")
    fig_favorita_problema()
    fav_fc, fav_lo, fav_hi, fav_mae, fav_mape, fav_wape = fig_favorita_prophet()
    fig_favorita_components()

    print("\n=== Bloque 7 - Cuando Prophet falla (sensor industrial) ===")
    fig_sensor_intro()
    fig_sensor_zoom()
    fc_pro_sensor, mae_pro_s, mape_pro_s = fig_sensor_prophet_falla()
    fig_por_que_prophet_falla()

    print("\n=== Bloque 8 - LSTM al rescate ===")
    fig_lstm_ventanas()
    fc_lstm_s, mae_lstm_s, mape_lstm_s = fig_lstm_forecast()
    fig_lstm_vs_prophet(fc_pro_sensor, mae_pro_s, mape_pro_s,
                         fc_lstm_s, mae_lstm_s, mape_lstm_s)

    print("\n=== Bloque 9 - Cierre ===")
    fig_decision_tree()

    print(f"\n--- Resumen ---")
    print(f"  AirPassengers Prophet MAE={mae:.0f} MAPE={mape:.1f}%")
    print(f"  Favorita Prophet MAE={fav_mae:.0f} MAPE={fav_mape:.1f}% WAPE={fav_wape:.1f}%")
    print(f"  Sensor Prophet MAE={mae_pro_s:.3f} MAPE={mape_pro_s:.1f}%")
    print(f"  Sensor LSTM    MAE={mae_lstm_s:.3f} MAPE={mape_lstm_s:.1f}%")
    print(f"  Mejora LSTM vs Prophet: {(mae_pro_s-mae_lstm_s)/mae_pro_s*100:.0f}%")
