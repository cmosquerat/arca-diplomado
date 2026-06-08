"""
make_data.py — construye el dataset SIMPLE y REAL de la clase 37.

Toma los datos diarios reales de la clase 32 (ventas de tiendas en Ecuador,
dataset de Corporación Favorita) y los AGREGA por mes. El resultado es una tabla
plana y fácil de razonar:

    tienda, categoria, anio, mes, unidades

Se corre una sola vez para generar ventas_tiendas_ecuador.csv (que sí se versiona
en el repo y es lo que lee el notebook).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "clase-32")

FUENTES = {
    ("Quito-44",     "bebidas"): "quito44_beverages_daily.csv",
    ("Quito-44",     "lacteos"): "quito44_dairy_daily.csv",
    ("Guayaquil-51", "bebidas"): "guayaquil51_beverages_daily.csv",
}

filas = []
for (tienda, categoria), archivo in FUENTES.items():
    df = pd.read_csv(os.path.join(SRC, archivo), parse_dates=["date"])
    mensual = df.groupby([df.date.dt.year, df.date.dt.month]).unit_sales.sum()
    for (anio, mes), unidades in mensual.items():
        filas.append({"tienda": tienda, "categoria": categoria,
                      "anio": int(anio), "mes": int(mes),
                      "unidades": int(round(unidades))})

out = pd.DataFrame(filas).sort_values(["tienda", "categoria", "anio", "mes"])
destino = os.path.join(ROOT, "ventas_tiendas_ecuador.csv")
out.to_csv(destino, index=False)
print(f"Generado {destino}: {len(out)} filas")
print(out.head(8).to_string(index=False))
print("...")
print(f"tiendas: {sorted(out.tienda.unique())}")
print(f"categorias: {sorted(out.categoria.unique())}")
print(f"anios: {out.anio.min()}-{out.anio.max()}")
