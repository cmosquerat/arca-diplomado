"""Genera desde la fuente pública de Volve el dataset, las figuras y el notebook.

Uso (desde esta carpeta):
    python build_materials.py /tmp/volve_production_source.xlsx
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

ROOT = Path(__file__).parent
SOURCE_URL = "https://raw.githubusercontent.com/jcreyesh/Volve-Field/main/Volve%20production%20data.xlsx"
RED, DARK, BLUE, ORANGE, GRAY = "#C82B40", "#6B1525", "#2563EB", "#EA580C", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def prepare_data(source: Path) -> pd.DataFrame:
    raw = pd.read_excel(source, sheet_name="Daily Production Data")
    selected = raw[raw["WELL_TYPE"] == "OP"].rename(columns={
        "NPD_WELL_BORE_NAME": "pozo", "DATEPRD": "fecha", "ON_STREAM_HRS": "horas",
        "AVG_DOWNHOLE_PRESSURE": "p_fondo", "AVG_WHP_P": "p_cabeza",
        "AVG_WHT_P": "t_cabeza", "AVG_CHOKE_SIZE_P": "choke",
        "DP_CHOKE_SIZE": "dp_choke", "BORE_OIL_VOL": "oil",
    })
    cols = ["pozo", "fecha", "horas", "p_fondo", "p_cabeza", "t_cabeza", "choke", "dp_choke", "oil"]
    data = selected[cols].dropna().copy()
    data = data[(data.oil > 0) & (data.horas > 0)].reset_index(drop=True)
    data.to_csv(ROOT / "operacion_pozos_volve.csv", index=False)
    return data


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(data: pd.DataFrame):
    features = ["horas", "p_fondo", "p_cabeza", "t_cabeza", "choke", "dp_choke"]
    X, y = data[features], data.oil
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
    linear = LinearRegression().fit(Xtr, ytr)
    tree5 = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, ytr)
    forest = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)

    # Curva física: la recta deja un patrón de error visible.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    s = data.sample(1200, random_state=7)
    ax.scatter(s.p_cabeza, s.oil, s=11, alpha=.25, color=BLUE, label="días de operación")
    coef = np.polyfit(s.p_cabeza, s.oil, 1); xs = np.linspace(s.p_cabeza.min(), s.p_cabeza.max(), 120)
    ax.plot(xs, np.polyval(coef, xs), color=RED, lw=3, label="una recta intenta resumirlo")
    ax.set(title="La producción no responde con una sola recta", xlabel="Presión de cabeza", ylabel="Petróleo (Sm³/día)")
    ax.legend(frameon=False); save(fig, "fig_curva_real.png")

    Xm, ym = make_moons(n_samples=400, noise=.18, random_state=8)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, model, title in zip(axs, [LogisticRegression(), DecisionTreeRegressor(max_depth=5, random_state=8)], ["Una frontera recta", "Cortes que forman zonas"]):
        model.fit(Xm, ym)
        xx, yy = np.meshgrid(np.linspace(-1.7, 2.7, 250), np.linspace(-1.2, 1.6, 250))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-.1,.5,1.1], colors=["#B9D7F5", "#F5C1C8"], alpha=.75)
        ax.scatter(Xm[:,0], Xm[:,1], c=np.where(ym == 1, RED, BLUE), s=16, edgecolor="white", linewidth=.25)
        ax.set_title(title); ax.set_xlabel("variable 1")
    axs[0].set_ylabel("variable 2"); save(fig, "fig_lunas_fronteras.png")

    tree_visual = DecisionTreeRegressor(max_depth=3, random_state=42).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    plot_tree(tree_visual, feature_names=features, filled=True, rounded=True, impurity=False, proportion=True, ax=ax, fontsize=10)
    ax.set_title("Un árbol de regresión: preguntas que dividen la operación", color=DARK, pad=14); save(fig, "fig_arbol.png")

    depths = range(1, 19); train_scores=[]; test_scores=[]
    for d in depths:
        m=DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xtr,ytr)
        train_scores.append(m.score(Xtr,ytr)); test_scores.append(m.score(Xte,yte))
    fig, ax=plt.subplots(figsize=(8.5,4.5)); ax.plot(depths, train_scores, "o-", color=RED, label="entrenamiento"); ax.plot(depths,test_scores,"o-",color=BLUE,label="prueba")
    ax.axvline(5, ls="--",color=GRAY); ax.set(xticks=range(1,19), ylim=(0.45,1.03), xlabel="Profundidad máxima", ylabel="R²", title="Más profundidad no siempre generaliza mejor")
    ax.legend(frameon=False); save(fig,"fig_overfitting.png")

    models=[("Regresión lineal",linear,RED),("Árbol (prof. 5)",tree5,ORANGE),("Random Forest",forest,BLUE)]
    rows=[]
    for name,m,color in models:
        p=m.predict(Xte); rows.append((name,r2_score(yte,p),mean_absolute_error(yte,p),color))
    fig, axs=plt.subplots(1,2,figsize=(10,4.2)); names=[r[0] for r in rows]; colors=[r[3] for r in rows]
    axs[0].bar(names,[r[1] for r in rows],color=colors); axs[0].set_ylim(0,1.05); axs[0].set_ylabel("R² en prueba"); axs[0].set_title("Explicación de la variación")
    axs[1].bar(names,[r[2] for r in rows],color=colors); axs[1].set_ylabel("MAE (Sm³/día)"); axs[1].set_title("Error operativo promedio")
    for ax in axs: ax.tick_params(axis="x",rotation=13)
    save(fig,"fig_comparacion.png")

    pred=forest.predict(Xte); fig,ax=plt.subplots(figsize=(6.2,5.3)); ax.scatter(yte,pred,s=14,alpha=.45,color=BLUE); lim=[0,max(yte.max(),pred.max())]; ax.plot(lim,lim,"--",color=RED,lw=2)
    ax.set(xlabel="Petróleo real (Sm³/día)",ylabel="Petróleo predicho (Sm³/día)",title="Random Forest: predicho frente a real"); save(fig,"fig_pred_real_rf.png")

    imp=pd.Series(forest.feature_importances_,index=features).sort_values(); fig,ax=plt.subplots(figsize=(7,4)); ax.barh(imp.index,imp.values,color=BLUE); ax.set(xlabel="Importancia en el modelo",title="Qué señales usó más el bosque")
    save(fig,"fig_importancia.png")


def markdown(text): return {"cell_type":"markdown","metadata":{},"source":[line+"\n" for line in text.splitlines()]}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in text.splitlines()]}

def make_notebook():
    url = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase3-codex/operacion_pozos_volve.csv"
    cells=[
      markdown("# Módulo 3 · Clase 3\n## Cuando una recta no alcanza: no linealidad, árboles y Random Forest\n\n**Caso:** medidor virtual de flujo con datos de operación de Volve.\n\n> En Google Colab ejecuta las celdas en orden. El dataset se descarga desde GitHub."),
      markdown("---\n# 0 · Preparación y carga desde GitHub"),
      code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nURL = '"+url+"'\nvolve = pd.read_csv(URL)\nprint(volve.shape)\nvolve.head()"),
      markdown("---\n# 1 · El problema\nQueremos estimar el volumen diario de petróleo (`oil`) desde horas en línea, presiones, temperatura y choke. Es **regresión**: el objetivo es un número."),
      code("volve.info()\nvolve.describe().T.round(2)"),
      code("fig, ax = plt.subplots(figsize=(8,4))\nax.scatter(volve['p_cabeza'], volve['oil'], s=8, alpha=.25)\nax.set(xlabel='Presión de cabeza', ylabel='Petróleo (Sm³/día)', title='¿Una recta basta?')\nplt.show()"),
      markdown("---\n# 2 · Baseline: una regresión lineal\nSiempre comenzamos con una referencia simple. Si un modelo más complejo no la supera en prueba, no lo conservamos."),
      code("from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_absolute_error, r2_score\n\nfeatures = ['horas','p_fondo','p_cabeza','t_cabeza','choke','dp_choke']\nX, y = volve[features], volve['oil']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n\nlineal = LinearRegression().fit(X_train, y_train)\np_lineal = lineal.predict(X_test)\nprint(f'R² prueba: {r2_score(y_test, p_lineal):.3f}')\nprint(f'MAE prueba: {mean_absolute_error(y_test, p_lineal):.0f} Sm³/día')"),
      markdown("---\n# 3 · Árbol de decisión\nUn árbol aprende reglas: por ejemplo, *si `p_cabeza` ≤ un corte, usa una predicción; si no, usa otra*. No necesita escalar las variables."),
      code("from sklearn.tree import DecisionTreeRegressor, plot_tree\n\narbol = DecisionTreeRegressor(max_depth=5, random_state=42)\narbol.fit(X_train, y_train)\np_arbol = arbol.predict(X_test)\nprint(f'R² prueba: {r2_score(y_test, p_arbol):.3f}')\nprint(f'MAE prueba: {mean_absolute_error(y_test, p_arbol):.0f} Sm³/día')\n\nplt.figure(figsize=(18,8))\nplot_tree(arbol, feature_names=features, filled=True, rounded=True, impurity=False, fontsize=7)\nplt.show()"),
      markdown("## Actividad 1\nPrueba `max_depth` = 2, 5, 10 y `None`. Para cada valor, compara el R² de entrenamiento con el de prueba. ¿Cuál evidencia sobreajuste?"),
      code("resultados = []\nfor profundidad in [2, 5, 10, None]:\n    m = DecisionTreeRegressor(max_depth=profundidad, random_state=42).fit(X_train, y_train)\n    resultados.append([profundidad, m.score(X_train,y_train), m.score(X_test,y_test), mean_absolute_error(y_test,m.predict(X_test))])\npd.DataFrame(resultados, columns=['max_depth','R² train','R² prueba','MAE prueba']).round(3)"),
      markdown("---\n# 4 · Random Forest\nUn único árbol es sensible a los datos que le tocaron. Random Forest entrena muchos árboles con muestras y variables distintas, y **promedia** sus predicciones."),
      code("from sklearn.ensemble import RandomForestRegressor\n\nbosque = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)\nbosque.fit(X_train, y_train)\np_bosque = bosque.predict(X_test)\n\ncomparacion = pd.DataFrame({\n 'modelo':['Regresión lineal','Árbol profundidad 5','Random Forest'],\n 'R² prueba':[r2_score(y_test,p_lineal), r2_score(y_test,p_arbol), r2_score(y_test,p_bosque)],\n 'MAE prueba':[mean_absolute_error(y_test,p_lineal), mean_absolute_error(y_test,p_arbol), mean_absolute_error(y_test,p_bosque)]\n})\ncomparacion.round(3)"),
      code("importances = pd.Series(bosque.feature_importances_, index=features).sort_values()\nimportances.plot.barh(figsize=(7,4), title='Importancia en Random Forest')\nplt.xlabel('Importancia en el modelo'); plt.show()"),
      markdown("## Actividad 2 · Decisión de ingeniería\nCambia `n_estimators` (50, 200, 500) y `min_samples_leaf` (1, 5, 20). Elige una configuración y justifica tu decisión con el desempeño **en prueba**, tiempo y facilidad de explicar el resultado."),
      markdown("---\n# Cierre\n- Una recta es una referencia; no una ley de la naturaleza.\n- El árbol aprende cortes e interacciones no lineales.\n- Profundidad excesiva memoriza.\n- Random Forest suele generalizar mejor, pero su importancia no demuestra causalidad.\n- La siguiente clase validará por pozo y ajustará hiperparámetros sin leakage."),
    ]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.x"},"colab":{"name":"Clase3_No_Linealidad_Arboles_RandomForest.ipynb","provenance":[]}},"nbformat":4,"nbformat_minor":5}
    (ROOT / "Clase3_No_Linealidad_Arboles_RandomForest.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Indica la ruta al archivo Volve production data.xlsx")
    data = prepare_data(Path(sys.argv[1]))
    make_figures(data)
    make_notebook()
    print(f"Listo: {len(data):,} filas; dataset, figuras y notebook generados en {ROOT}")
