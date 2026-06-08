"""
Figuras para clase 37 — Agentes con smolagents.

Solo genera figuras de DATOS REALES (las ventas de tiendas en Ecuador, agregadas
en make_data.py). Todo lo conceptual (estructura de smolagents, bucle, etc.) va en
TikZ dentro del .tex.

  fig_ventas_anual.png — total de unidades por año, por tienda/categoría.

Estilo: paleta Arca. No usa TensorFlow (deadlock en macOS — lección clase 33).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCA_RED, ARCA_DARK = "#C82B40", "#6B1525"
ARCA_BLUE, ARCA_ORANGE = "#2563EB", "#EA580C"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
})


def fig_ventas_anual():
    df = pd.read_csv(os.path.join(ROOT, "ventas_tiendas_ecuador.csv"))
    df["serie"] = df.tienda + " / " + df.categoria
    tabla = df.groupby(["anio", "serie"]).unidades.sum().unstack()
    colores = {"Quito-44 / bebidas": ARCA_RED, "Quito-44 / lacteos": ARCA_ORANGE,
               "Guayaquil-51 / bebidas": ARCA_BLUE}
    ax = tabla.plot(kind="bar", figsize=(11, 4),
                    color=[colores.get(c, "#999") for c in tabla.columns])
    ax.set_title("Ventas anuales por tienda y categoría (datos reales, Ecuador)")
    ax.set_xlabel("año"); ax.set_ylabel("unidades")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, fontsize=9)
    fig = ax.get_figure(); fig.tight_layout()
    out = os.path.join(ROOT, "fig_ventas_anual.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    print(f"  guardada fig_ventas_anual.png")


def fig_chart_demo():
    """Replica la salida de la herramienta GraficarVentas del notebook."""
    df = pd.read_csv(os.path.join(ROOT, "ventas_tiendas_ecuador.csv"))
    d = df[(df.tienda == "Quito-44") & (df.categoria == "bebidas")]
    serie = d.groupby("anio").unidades.sum()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(serie.index.astype(str), serie.values, color=ARCA_RED)
    ax.set_title("Ventas anuales — Quito-44 / bebidas")
    ax.set_ylabel("unidades")
    fig.tight_layout()
    out = os.path.join(ROOT, "fig_chart_demo.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    print("  guardada fig_chart_demo.png")


if __name__ == "__main__":
    print("Generando figuras de datos reales...")
    fig_ventas_anual()
    fig_chart_demo()
    print("Listo. (Nota: 2017 es parcial; los datos llegan hasta agosto.)")
