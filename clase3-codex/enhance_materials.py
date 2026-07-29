"""Figuras adicionales y notebook de la Clase 3.

Trabaja únicamente con los CSV versionados en esta carpeta.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
RED, DARK, BLUE, GREEN, ORANGE = "#C82B40", "#6B1525", "#2563EB", "#16A34A", "#EA580C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def nonlinear_gallery():
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.2))
    x = np.linspace(0, 10, 180)
    specs = [
        (axs[0, 0], 1.1 * x + 1, "Lineal", "El mismo paso produce el mismo cambio"),
        (axs[0, 1], 9 * (1 - np.exp(-x / 2.2)), "Saturación", "Crece y luego llega a una meseta"),
        (axs[1, 0], np.where(x < 5, 2, 8), "Umbral", "Al cruzar un punto cambia el régimen"),
        (axs[1, 1], 10 * np.exp(-x / 2.7), "Declive", "Cae rápido y después más lentamente"),
    ]
    for ax, y, title, subtitle in specs:
        ax.plot(x, y, lw=3, color=BLUE if title == "Lineal" else RED)
        ax.fill_between(x, y, alpha=.08, color=BLUE if title == "Lineal" else RED)
        ax.set_title(title, loc="left")
        ax.text(.02, .90, subtitle, transform=ax.transAxes, fontsize=9, color=DARK)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(alpha=.15)
    save(fig, "fig_galeria_nolineal.png")


def classification_shapes():
    sets = [
        (*make_moons(n_samples=500, noise=.16, random_state=12), "Lunas"),
        (*make_circles(n_samples=500, noise=.10, factor=.42, random_state=12), "Círculos"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
    for row, (X, y, label) in enumerate(sets):
        models = [
            (None, f"{label}: problema"),
            (LogisticRegression(), "Logística: frontera recta"),
            (DecisionTreeClassifier(max_depth=5, random_state=42), "Árbol: regiones"),
        ]
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min()-.45, X[:, 0].max()+.45, 300),
            np.linspace(X[:, 1].min()-.45, X[:, 1].max()+.45, 300),
        )
        for col, (model, title) in enumerate(models):
            ax = axs[row, col]
            if model is not None:
                model.fit(X, y)
                zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                ax.contourf(xx, yy, zz, levels=[-.1, .5, 1.1],
                            colors=["#CFE5FA", "#F8CCD2"], alpha=.72)
                score = model.score(X, y)
                title += f"\naccuracy = {score:.2f}"
            ax.scatter(X[:, 0], X[:, 1], c=np.where(y, RED, BLUE),
                       s=11, edgecolor="white", linewidth=.2)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig_lunas_circulos.png")


def split_candidates():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(35, 10, 50), rng.normal(72, 10, 50)]
    y = np.r_[np.zeros(50), np.ones(50)]
    cuts = [40, 55, 70]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, cut in zip(axs, cuts):
        ax.scatter(x[y == 0], rng.normal(0, .035, 50), color=BLUE, s=22, label="clase 0")
        ax.scatter(x[y == 1], rng.normal(.18, .035, 50), color=RED, s=22, label="clase 1")
        ax.axvline(cut, color=ORANGE, lw=3, ls="--")
        left = y[x <= cut]; right = y[x > cut]
        mixed = sum(min(z.sum(), len(z)-z.sum()) for z in (left, right) if len(z))
        ax.set_title(f"Corte = {cut}\n{int(mixed)} mezclados")
        ax.set_xlabel("valor de una variable"); ax.set_yticks([])
    axs[0].legend(frameon=False, loc="upper left")
    save(fig, "fig_candidatos_corte.png")


def force_results():
    d = pd.read_csv(ROOT / "litologia_force2020.csv")
    d["y"] = (d["LITH"] == "Sandstone").astype(int)
    feats = ["GR", "RHOB", "NPHI", "DTC", "RDEP"]
    Xtr, Xte, ytr, yte = train_test_split(
        d[feats], d.y, test_size=.25, random_state=42, stratify=d.y
    )
    models = {
        "Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Árbol": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    preds = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds[name] = model.predict(Xte)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (name, pred) in zip(axs, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            yte, pred, display_labels=["Lutita", "Arena"], cmap="Blues",
            colorbar=False, ax=ax
        )
        ax.set_title(f"{name}\naccuracy={accuracy_score(yte, pred):.3f}")
    save(fig, "fig_confusiones_modelos.png")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook():
    volve_url = (
        "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/refs/heads/"
        "agent/modulo3-clase3-codex/clase3-codex/operacion_pozos_volve.csv"
    )
    force_url = (
        "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/refs/heads/"
        "agent/modulo3-clase3-codex/clase3-codex/litologia_force2020.csv"
    )
    cells = [
        md("# Módulo 3 · Clase 3\n## Cuando una recta no alcanza: árboles y Random Forest\n\n"
           "[Abrir este notebook en Google Colab](https://colab.research.google.com/github/"
           "cmosquerat/arca-diplomado/blob/agent/modulo3-clase3-codex/clase3-codex/"
           "Clase3_No_Linealidad_Arboles_RandomForest.ipynb)\n\n"
           "**Dos problemas:** estimar producción (regresión) y reconocer arenisca "
           "(clasificación). El material se carga directamente desde GitHub."),
        md("---\n# 0 · Preparación"),
        code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.metrics import mean_absolute_error, r2_score, "
             "accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay\n"),
        md("---\n# 1 · Ver la no linealidad\n\nUna recta obliga a que el efecto sea "
           "constante. Lunas y círculos muestran fronteras que una logística no puede doblar."),
        code("from sklearn.datasets import make_moons, make_circles\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "from sklearn.tree import DecisionTreeClassifier\n\n"
             "X_moon, y_moon = make_moons(n_samples=500, noise=.16, random_state=12)\n"
             "X_circle, y_circle = make_circles(n_samples=500, noise=.10, factor=.42, random_state=12)\n"),
        code("fig, axs = plt.subplots(1,2,figsize=(10,4))\n"
             "for ax, X, y, title in [(axs[0],X_moon,y_moon,'Lunas'),"
             "(axs[1],X_circle,y_circle,'Círculos')]:\n"
             "    ax.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm',s=16)\n"
             "    ax.set_title(title)\nplt.show()\n"),
        code("for nombre, modelo in [('Logística', LogisticRegression()),"
             "('Árbol', DecisionTreeClassifier(max_depth=5,random_state=42))]:\n"
             "    modelo.fit(X_moon,y_moon)\n"
             "    print(nombre, 'accuracy lunas:', round(modelo.score(X_moon,y_moon),3))\n"),
        md("---\n# 2 · Cómo aprende un árbol\n\nEl árbol prueba variables y cortes. "
           "Escoge la pregunta que deja grupos más homogéneos y repite el proceso en cada rama."),
        code("from sklearn.tree import plot_tree\n"
             "arbol_demo = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_moon,y_moon)\n"
             "plt.figure(figsize=(14,6))\n"
             "plot_tree(arbol_demo,filled=True,rounded=True,impurity=False,proportion=True)\n"
             "plt.show()\n"),
        md("## Actividad corta\n\nLee una ruta completa: comienza en la raíz, "
           "elige verdadero/falso y termina en una hoja. ¿Qué clase predice y con qué proporción?"),
        md("---\n# 3 · Problema de regresión: medidor virtual de flujo"),
        code(f"URL_VOLVE = '{volve_url}'\nvolve = pd.read_csv(URL_VOLVE)\n"
             "print(volve.shape)\nvolve.head()\n"),
        md("## Diccionario de variables\n\n"
           "|Variable|Significado|\n|---|---|\n|`horas`|horas en operación ese día|\n"
           "|`p_fondo`|presión de fondo|\n|`p_cabeza`|presión de cabeza|\n"
           "|`t_cabeza`|temperatura en cabeza|\n|`choke`|apertura del choke|\n"
           "|`dp_choke`|caída de presión en el choke|\n|`oil`|producción medida; objetivo|"),
        code("features = ['horas','p_fondo','p_cabeza','t_cabeza','choke','dp_choke']\n"
             "X, y = volve[features], volve['oil']\n"
             "Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=42)\n"),
        code("from sklearn.linear_model import LinearRegression\n"
             "lineal=LinearRegression().fit(Xtr,ytr)\n"
             "p=lineal.predict(Xte)\n"
             "print('Lineal R²:',round(r2_score(yte,p),3),'MAE:',round(mean_absolute_error(yte,p)))\n"),
        code("from sklearn.tree import DecisionTreeRegressor\n"
             "arbol=DecisionTreeRegressor(max_depth=5,random_state=42).fit(Xtr,ytr)\n"
             "p=arbol.predict(Xte)\n"
             "print('Árbol R²:',round(r2_score(yte,p),3),'MAE:',round(mean_absolute_error(yte,p)))\n"),
        md("## Profundidad y sobreajuste\n\nPrueba 2, 5, 10 y sin límite. "
           "Un desempeño perfecto en entrenamiento puede ser una señal de memoria."),
        code("filas=[]\nfor depth in [2,5,10,None]:\n"
             "    m=DecisionTreeRegressor(max_depth=depth,random_state=42).fit(Xtr,ytr)\n"
             "    filas.append([depth,m.score(Xtr,ytr),m.score(Xte,yte),"
             "mean_absolute_error(yte,m.predict(Xte))])\n"
             "pd.DataFrame(filas,columns=['profundidad','R² train','R² test','MAE test']).round(3)\n"),
        md("---\n# 4 · Random Forest para regresión"),
        code("from sklearn.ensemble import RandomForestRegressor\n"
             "bosque=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1).fit(Xtr,ytr)\n"
             "p_rf=bosque.predict(Xte)\n"
             "print('Bosque R²:',round(r2_score(yte,p_rf),3),'MAE:',round(mean_absolute_error(yte,p_rf)))\n"),
        code("pd.Series(bosque.feature_importances_,index=features).sort_values().plot.barh("
             "title='Importancia de variables')\nplt.show()\n"),
        md("---\n# 5 · Segundo dataset: clasificación de litología FORCE 2020\n\n"
           "Volvemos al problema de la Clase 2: arenisca o lutita. Ahora permitimos "
           "que el modelo aprenda fronteras no lineales."),
        code(f"URL_FORCE = '{force_url}'\nlito = pd.read_csv(URL_FORCE)\n"
             "lito['y']=(lito['LITH']=='Sandstone').astype(int)\n"
             "lito[['GR','RHOB','NPHI','DTC','RDEP','LITH']].head()\n"),
        code("features_lito=['GR','RHOB','NPHI','DTC','RDEP']\n"
             "X=lito[features_lito]; y=lito['y']\n"
             "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)\n"),
        code("from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
             "from sklearn.ensemble import RandomForestClassifier\n"
             "modelos={\n"
             " 'Logística':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),\n"
             " 'Árbol':DecisionTreeClassifier(max_depth=8,random_state=42),\n"
             " 'Bosque':RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)}\n"
             "resultados=[]\nfor nombre,m in modelos.items():\n"
             "    m.fit(Xtr,ytr); pred=m.predict(Xte)\n"
             "    tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()\n"
             "    resultados.append([nombre,accuracy_score(yte,pred),recall_score(yte,pred),fn*10+fp])\n"
             "pd.DataFrame(resultados,columns=['modelo','accuracy','recall arena','costo']).round(3)\n"),
        md("## Actividad final\n\n1. ¿Qué modelo reduce más el costo de la Clase 2?\n"
           "2. Cambia `max_depth` del árbol.\n3. Explica por qué el bosque puede ganar "
           "precisión y a la vez perder explicabilidad."),
        md("---\n# Cierre\n\n- Un árbol aprende preguntas y cortes, no una fórmula curva.\n"
           "- En regresión, una hoja predice un promedio.\n"
           "- En clasificación, una hoja vota y produce probabilidades.\n"
           "- La profundidad controla la tensión entre aprender y memorizar.\n"
           "- Random Forest estabiliza muchos árboles distintos."),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "colab": {"name": "Clase3_No_Linealidad_Arboles_RandomForest.ipynb", "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (ROOT / "Clase3_No_Linealidad_Arboles_RandomForest.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    nonlinear_gallery()
    classification_shapes()
    split_candidates()
    force_results()
    notebook()
    print("Figuras y notebook ampliado generados.")
