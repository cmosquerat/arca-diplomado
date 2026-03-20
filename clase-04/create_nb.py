import json

def md(text):
    lines = text.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(text):
    lines = text.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}

cells = []

# ── HEADER ──
cells.append(md("""# Clase 4: Diccionarios y Funciones

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**  
Arca Continental Ecuador · UDLA · 2026

---

**Instructor:** Carlos Enrique Mosquera Trujillo  
**Contacto:** cmosquerat@unal.edu.co

---

### Contenido de esta sesión

| # | Tema | Duración |
|---|------|----------|
| 1 | Diccionarios: concepto, acceso, métodos, iteración | 20 min |
| 2 | Lista de diccionarios (datos tabulares) | 10 min |
| 3 | Ejercicios de diccionarios | 12 min |
| 4 | Funciones: `def`, parámetros, `return`, scope | 25 min |
| 5 | Funciones que procesan colecciones | 15 min |
| 6 | Ejercicios de funciones | 15 min |
| 7 | Proyecto Integrador: Sistema de Gestión de Rutas | 20 min |"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 1: DICCIONARIOS
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 1. Diccionarios

### El problema de las listas paralelas

En la Clase 3 usamos listas separadas para clientes, pedidos y productos.  
Si alguien reordena una lista, todo se rompe. Los diccionarios resuelven esto."""))

cells.append(code("""# Listas paralelas — frágiles
clientes = ["Tienda Sol", "Mini ABC"]
pedidos  = [30, 15]
productos = ["Coca-Cola", "Sprite"]

# Diccionario — todo junto
cliente = {
    "nombre": "Tienda Sol",
    "pedido": 30,
    "producto": "Coca-Cola"
}
print(cliente)"""))

cells.append(md("""### Creación y acceso

Un diccionario guarda pares **clave: valor**. Se accede por clave, no por índice.

```
    Clave          Valor
  ─────────    ──────────────
  "nombre"  →  "Tienda Sol"
  "pedido"  →  30
  "producto" → "Coca-Cola"
  "zona"    →  "Norte"
```"""))

cells.append(code("""cliente = {
    "nombre": "Tienda Sol",
    "pedido": 30,
    "producto": "Coca-Cola",
    "zona": "Norte"
}

print("Nombre:", cliente["nombre"])
print("Pedido:", cliente["pedido"])"""))

cells.append(md("""### Acceso seguro con `.get()`"""))

cells.append(code("""producto = {
    "nombre": "Coca-Cola 600ml",
    "precio": 1.50,
    "stock": 200
}

# .get() no da error si la clave no existe
print(producto.get("precio"))
print(producto.get("color", "N/A"))  # Default si no existe

# Esto daría KeyError:
# print(producto["color"])"""))

cells.append(md("""### Modificar, agregar y eliminar"""))

cells.append(code("""producto = {
    "nombre": "Coca-Cola 600ml",
    "precio": 1.50,
    "stock": 200
}

# Modificar un valor existente
producto["precio"] = 1.65

# Agregar una clave nueva
producto["categoria"] = "Bebidas"

# Eliminar una clave
del producto["stock"]

for k, v in producto.items():
    print(f"  {k}: {v}")"""))

cells.append(md("""### Métodos esenciales e iteración

| Método | Devuelve |
|--------|----------|
| `.keys()` | Todas las claves |
| `.values()` | Todos los valores |
| `.items()` | Pares (clave, valor) |
| `.get(k, d)` | Valor de `k`, o `d` si no existe |
| `.pop(k)` | Elimina y devuelve valor de `k` |"""))

cells.append(code("""producto = {
    "nombre": "Sprite 500ml",
    "precio": 1.25,
    "stock": 150
}

print("Claves:", list(producto.keys()))
print("Valores:", list(producto.values()))
print()

# .items() es la forma más común de iterar
for clave, valor in producto.items():
    print(f"  {clave}: {valor}")"""))

# ── LISTA DE DICTS ──
cells.append(md("""### Lista de diccionarios — Datos tabulares

Una lista de diccionarios es como una tabla:  
cada diccionario es una **fila**, cada clave es una **columna**.

Esto es la base conceptual de cómo funcionan las tablas en Pandas."""))

cells.append(code("""clientes = [
    {"nombre": "Tienda Sol", "pedido": 30, "zona": "Norte"},
    {"nombre": "Mini ABC",   "pedido": 15, "zona": "Sur"},
    {"nombre": "Super 10",   "pedido": 50, "zona": "Norte"},
]

for c in clientes:
    print(c["nombre"], "->", c["pedido"], "cajas")"""))

# ── EJERCICIO 1 ──
cells.append(md("""### Práctica: Ficha de Producto ⏱ 5 min"""))

cells.append(code("""# 1. Creen un diccionario "producto" con claves:
#    "nombre", "precio", "stock", "proveedor"


# 2. Impriman el nombre y el precio con f-string


# 3. Actualicen el stock a 180


# 4. Agreguen la clave "descuento" con valor 0.10


# 5. Iteren con .items() e impriman cada par clave-valor
"""))

# ── EJERCICIO 2 ──
cells.append(md("""### Práctica: Reporte de Clientes ⏱ 7 min

Filtren solo los clientes activos, calculen ingreso y reporten."""))

cells.append(code("""clientes = [
    {"nombre": "Tienda Sol", "pedido": 30, "zona": "Norte", "activo": True},
    {"nombre": "Mini ABC",   "pedido": 15, "zona": "Sur",   "activo": False},
    {"nombre": "Super 10",   "pedido": 50, "zona": "Norte", "activo": True},
    {"nombre": "Bodega LP",  "pedido": 20, "zona": "Centro","activo": True},
    {"nombre": "Market 7",   "pedido": 40, "zona": "Sur",   "activo": True},
]
precio_caja = 12.50

# 1. Recorran la lista y solo procesen clientes activos
# 2. Para cada activo, calculen el ingreso (pedido * precio_caja)
# 3. Acumulen el ingreso total y cuenten los clientes activos
# 4. Impriman cada cliente procesado y el resumen final
"""))

cells.append(md("""<details><summary><b>Ver solución — Reporte de Clientes</b></summary>

```python
total_ingreso = 0
activos = 0

for c in clientes:
    if c["activo"]:
        ingreso = c["pedido"] * precio_caja
        total_ingreso += ingreso
        activos += 1
        print(f"{c['nombre']} ({c['zona']}): ${ingreso:.2f}")

print(f"\\nClientes activos: {activos}")
print(f"Ingreso total: ${total_ingreso:.2f}")
```

</details>"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 2: FUNCIONES
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 2. Funciones

### El problema: el código se repite"""))

cells.append(code("""# Sin funciones — copiar y pegar
ingreso1 = 30 * 12.50
print("Tienda Sol:", ingreso1)

ingreso2 = 50 * 12.50
print("Super 10:", ingreso2)

ingreso3 = 40 * 12.50
print("Market 7:", ingreso3)"""))

cells.append(code("""# Con función — se escribe una vez
def calcular_ingreso(nombre, cajas):
    ingreso = cajas * 12.50
    print(f"{nombre}: ${ingreso:.2f}")
    return ingreso

calcular_ingreso("Tienda Sol", 30)
calcular_ingreso("Super 10", 50)
calcular_ingreso("Market 7", 40)"""))

cells.append(md("""### Anatomía de una función

```
def  calcular_ingreso  (cajas, precio):     ← definición
 │        │                  │
 │     nombre            parámetros
 │
 └─ palabra clave

    ingreso = cajas * precio               ← cuerpo (indentado)
    return ingreso                          ← salida (opcional)
```

- `def` inicia la definición
- Los **parámetros** reciben datos al llamar
- `return` devuelve el resultado. Sin él, la función devuelve `None`"""))

cells.append(code("""def calcular_ingreso(cajas, precio):
    ingreso = cajas * precio
    return ingreso

resultado = calcular_ingreso(30, 12.50)
print("Ingreso:", resultado)"""))

cells.append(md("""### `return` vs `print` — No son lo mismo

- `print()` muestra texto en pantalla — no se puede reutilizar
- `return` entrega un valor que el programa puede guardar, operar o pasar a otra función"""))

cells.append(code("""# Solo imprime — no se puede reutilizar
def mostrar_total(cajas, precio):
    print(cajas * precio)

resultado = mostrar_total(30, 12.50)
print("Guardado:", resultado)  # None!"""))

cells.append(code("""# Retorna — el valor se puede usar después
def calcular_total(cajas, precio):
    return cajas * precio

resultado = calcular_total(30, 12.50)
print("Guardado:", resultado)
print("Con IVA:", resultado * 1.12)"""))

cells.append(md("""### Valores por defecto"""))

cells.append(code("""def calcular_ingreso(cajas, precio=12.50):
    return cajas * precio

print(calcular_ingreso(30))         # Usa default 12.50
print(calcular_ingreso(30, 15.00))  # Usa 15.00"""))

cells.append(md("""### Alcance (scope) de variables

Las variables creadas **dentro** de una función son **locales**: solo existen ahí."""))

cells.append(code("""def duplicar(x):
    resultado = x * 2
    return resultado

print(duplicar(5))

# Descomentar la siguiente línea da NameError:
# print(resultado)  # No existe fuera de la función"""))

cells.append(md("""### Funciones que procesan colecciones

Las funciones pueden recibir listas o diccionarios como parámetros.  
Se pueden retornar **múltiples valores** separados por coma."""))

cells.append(code("""def resumen_ventas(ventas, meta):
    total = 0
    sobre_meta = 0
    for v in ventas:
        total += v
        if v >= meta:
            sobre_meta += 1
    promedio = total / len(ventas)
    return total, sobre_meta, promedio

datos = [150, 80, 200, 45, 310, 60]
t, s, p = resumen_ventas(datos, 100)
print(f"Total: {t}")
print(f"Sobre meta: {s}")
print(f"Promedio: {p:.1f}")"""))

cells.append(md("""### Funciones + Diccionarios

Las funciones pueden recibir y retornar diccionarios."""))

cells.append(code("""def procesar_pedido(cliente, precio_caja):
    nombre = cliente["nombre"]
    cajas  = cliente["pedido"]
    ingreso = cajas * precio_caja
    estado = "OK" if cliente["activo"] else "INACTIVO"
    return {
        "nombre": nombre,
        "ingreso": ingreso,
        "estado": estado
    }

c = {"nombre": "Tienda Sol", "pedido": 30, "activo": True}
resultado = procesar_pedido(c, 12.50)
for k, v in resultado.items():
    print(f"  {k}: {v}")"""))

# ── EJERCICIO 3 ──
cells.append(md("""### Práctica: Funciones de Análisis ⏱ 7 min

Creen funciones reutilizables para análisis de datos."""))

cells.append(code("""ventas = [150, 80, 200, 45, 310, 60]

# 1. Función calcular_promedio(lista):
#    Recibe una lista de números y retorna el promedio


# 2. Función clasificar(valor, umbral):
#    Retorna "Alto" si valor >= umbral, "Bajo" en otro caso


# 3. Función contar_por_clase(lista, umbral):
#    Retorna cuántos son "Alto" y cuántos "Bajo"
#    (Usen clasificar() adentro — una función puede llamar a otra)


# Prueben:
# print(calcular_promedio(ventas))
# print(clasificar(200, 100))
# print(contar_por_clase(ventas, 100))
"""))

cells.append(md("""<details><summary><b>Ver solución — Funciones de Análisis</b></summary>

```python
def calcular_promedio(lista):
    return sum(lista) / len(lista)

def clasificar(valor, umbral):
    if valor >= umbral:
        return "Alto"
    return "Bajo"

def contar_por_clase(lista, umbral):
    altos = 0
    bajos = 0
    for v in lista:
        if clasificar(v, umbral) == "Alto":
            altos += 1
        else:
            bajos += 1
    return altos, bajos

print(calcular_promedio(ventas))      # 140.83...
print(clasificar(200, 100))           # Alto
print(contar_por_clase(ventas, 100))  # (3, 3)
```

</details>"""))

# ── EJERCICIO 4 ──
cells.append(md("""### Práctica: Pipeline de Procesamiento ⏱ 8 min

Construyan un mini-pipeline con funciones que procesan una lista de diccionarios."""))

cells.append(code("""empleados = [
    {"nombre": "Ana",    "ventas": 15200, "zona": "Norte"},
    {"nombre": "Luis",   "ventas": 8900,  "zona": "Sur"},
    {"nombre": "Maria",  "ventas": 12400, "zona": "Norte"},
    {"nombre": "Carlos", "ventas": 6100,  "zona": "Centro"},
    {"nombre": "Sofia",  "ventas": 18300, "zona": "Sur"},
]

# 1. filtrar_zona(empleados, zona): retorna lista con solo
#    los empleados de esa zona


# 2. total_ventas(lista): retorna la suma de "ventas"
#    de todos los dicts en la lista


# 3. Combínenlas: filtren zona "Norte", calculen su total,
#    y compárenlo con el total general
"""))

cells.append(md("""<details><summary><b>Ver solución — Pipeline</b></summary>

```python
def filtrar_zona(empleados, zona):
    resultado = []
    for e in empleados:
        if e["zona"] == zona:
            resultado.append(e)
    return resultado

def total_ventas(lista):
    total = 0
    for e in lista:
        total += e["ventas"]
    return total

norte = filtrar_zona(empleados, "Norte")
print(f"Empleados Norte: {len(norte)}")
print(f"Ventas Norte: ${total_ventas(norte):,.2f}")
print(f"Ventas Total: ${total_ventas(empleados):,.2f}")
```

</details>"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 3: PROYECTO INTEGRADOR
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 3. Proyecto Integrador: Sistema de Gestión de Rutas ⏱ 20 min

Un camión sale del centro de distribución con carga limitada.  
Debe visitar 6 clientes y decidir en cada parada si puede entregar.

### Funciones a crear:

1. `validar_producto(prod, catalogo)`: retorna `True`/`False`
2. `procesar_parada(parada, carga, precio, catalogo)`: retorna un dict con resultado (entregado/rechazado, ingreso, carga restante)
3. `generar_reporte(resultados)`: imprime resumen (entregas, rechazos, ingreso total, carga final)

### Lógica por parada:
- Producto no en catálogo → "NO DISPONIBLE"
- Carga insuficiente → "SIN STOCK"
- OK → entregar, restar carga, sumar ingreso"""))

cells.append(code("""# Datos de la ruta
paradas = [
    {"cliente": "Tienda Sol", "cajas": 30, "producto": "Coca-Cola", "zona": "Norte"},
    {"cliente": "Mini ABC",   "cajas": 15, "producto": "Sprite",    "zona": "Sur"},
    {"cliente": "Super 10",   "cajas": 50, "producto": "Fanta",     "zona": "Norte"},
    {"cliente": "Bodega LP",  "cajas": 20, "producto": "Coca-Cola", "zona": "Centro"},
    {"cliente": "Market 7",   "cajas": 40, "producto": "Dasani",    "zona": "Sur"},
    {"cliente": "Don Pepe",   "cajas": 10, "producto": "Sprite",    "zona": "Norte"},
]
catalogo = ["Coca-Cola", "Sprite", "Dasani"]
carga_inicial = 120
precio_caja = 12.50"""))

cells.append(code("""# Escriban sus funciones aquí

# 1. validar_producto(prod, catalogo)


# 2. procesar_parada(parada, carga, precio, catalogo)


# 3. generar_reporte(resultados)


# Ejecuten el sistema:
# resultados = []
# carga = carga_inicial
# for p in paradas:
#     resultado = procesar_parada(p, carga, precio_caja, catalogo)
#     resultados.append(resultado)
#     carga = resultado["carga_restante"]
#
# generar_reporte(resultados)
"""))

cells.append(md("""---

**Clase 4 completada.**  
En la próxima clase veremos ciclos `while`, manejo de strings, `try/except` y lectura de archivos CSV."""))

# ── WRITE NOTEBOOK ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("/root/Arca/clase-04/Clase_04_Diccionarios_y_Funciones.ipynb", "w") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook creado exitosamente")
print(f"Celdas totales: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type']=='markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type']=='code')}")
