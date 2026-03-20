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
cells.append(md("""# Clase 3: Iteración y Lógica Aplicada

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**  
Arca Continental Ecuador · UDLA · 2026

---

**Instructor:** Carlos Enrique Mosquera Trujillo  
**Contacto:** cmosquerat@unal.edu.co

---

### Contenido de esta sesión

| # | Tema | Duración |
|---|------|----------|
| 1 | Repaso: Operadores lógicos (`and`, `or`, `not`) | 10 min |
| 2 | Repaso: Listas — Índices, slicing y métodos | 15 min |
| 3 | Ciclos `for` y `range()` | 15 min |
| 4 | Patrón acumulador y contadores | 15 min |
| 5 | `for` + `if` y el operador `in` | 15 min |
| 6 | Retos Prácticos (3 retos) | 20 min |
| 7 | Proyecto Integrador: Simulador de Ruta | 20 min |"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 1: REPASO LÓGICA BOOLEANA
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 1. Repaso: Operadores Lógicos

Recordemos cómo funcionan `and`, `or` y `not`:

| Operador | Resultado `True` cuando... | Ejemplo |
|----------|---------------------------|---------|
| `and` | **Ambas** condiciones son verdaderas | `stock > 0 and credito_ok` |
| `or` | **Al menos una** es verdadera | `es_urgente or es_preferente` |
| `not` | Invierte el valor | `not en_mantenimiento` |"""))

cells.append(code("""# Ejemplo rápido
hay_stock = True
tiene_credito = False
zona_bloqueada = False

print("and:", hay_stock and tiene_credito)
print("or: ", hay_stock or tiene_credito)
print("not:", not zona_bloqueada)"""))

cells.append(md("""### Práctica: Traducir Reglas a Código ⏱ 5 min

Traduzcan cada regla de negocio a una expresión booleana en Python."""))

cells.append(code("""# Variables de contexto
temperatura = 38
humedad = 72
en_mantenimiento = False
stock = 45
minimo = 50
es_preferente = True"""))

cells.append(code("""# 1. "Hay riesgo ambiental si la temperatura supera 35°C
#     o la humedad supera 80%."
riesgo = # Escribe tu expresión aquí
print("Riesgo:", riesgo)"""))

cells.append(code("""# 2. "La alarma suena si hay riesgo y el sistema
#     no está en mantenimiento."
alarma = # Escribe tu expresión aquí
print("Alarma:", alarma)"""))

cells.append(code("""# 3. "Se despacha si el stock alcanza o el cliente es preferente."
despacha = # Escribe tu expresión aquí
print("Despacha:", despacha)"""))

cells.append(code("""# 4. "Se bloquea si el stock no alcanza y no es preferente."
bloquea = # Escribe tu expresión aquí
print("Bloquea:", bloquea)"""))

cells.append(md("""<details><summary><b>Ver solución</b></summary>

```python
riesgo = temperatura > 35 or humedad > 80
alarma = riesgo and not en_mantenimiento
despacha = stock >= minimo or es_preferente
bloquea = stock < minimo and not es_preferente

print("Riesgo:", riesgo)       # True
print("Alarma:", alarma)       # True
print("Despacha:", despacha)   # True
print("Bloquea:", bloquea)     # False
```
Las reglas 3 y 4 son complementarias: una da `True` y la otra `False`.  
Cambien `es_preferente = False` y verifiquen que se invierte.

</details>"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 2: REPASO LISTAS
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 2. Repaso: Listas — Índices, Slicing y Métodos

Una lista guarda varios valores bajo un solo nombre.  
Cada valor tiene una **posición** (índice) que empieza en `0`.

```
Índice:      0           1          2          3
          ┌──────────┬──────────┬──────────┬──────────┐
          │"GYE Norte"│"GYE Sur" │  "Manta" │ "Cuenca" │
          └──────────┴──────────┴──────────┴──────────┘
Negativo:   -4          -3         -2         -1
```"""))

cells.append(code("""# Acceso por índice
rutas = ["GYE Norte", "GYE Sur", "Manta", "Cuenca"]

print(rutas[0])    # Primer elemento
print(rutas[2])    # Tercer elemento
print(rutas[-1])   # Último elemento
print(len(rutas))  # Cantidad de elementos"""))

cells.append(md("""### Slicing (rebanadas)

Con `lista[inicio:fin]` extraemos un subconjunto. El índice `fin` **no se incluye**."""))

cells.append(code("""precios = [1.50, 1.25, 0.90, 1.10, 0.80]

print("precios[1:3] =", precios[1:3])  # Índices 1 y 2
print("precios[:2]  =", precios[:2])   # Desde el inicio hasta el 2
print("precios[3:]  =", precios[3:])   # Desde el 3 hasta el final"""))

cells.append(md("""### Métodos esenciales

| Método | Qué hace |
|--------|----------|
| `.append(x)` | Agrega `x` al final |
| `.remove(x)` | Elimina la primera aparición de `x` |
| `.sort()` | Ordena de menor a mayor |
| `.pop(i)` | Elimina y devuelve el elemento en posición `i` |
| `lista[i] = v` | Reemplaza el valor en posición `i` |

Estos métodos **modifican** la lista original (in-place)."""))

cells.append(code("""rutas = ["GYE Norte", "Manta"]
rutas.append("Cuenca")
print("Append:", rutas)

rutas[0] = "Guayaquil N."
print("Reasig:", rutas)

rutas.sort()
print("Sort:  ", rutas)

rutas.pop(1)
print("Pop(1):", rutas)"""))

cells.append(md("""### Práctica: Selección de Turnos ⏱ 5 min

Una planta registra la producción por turno en la semana."""))

cells.append(code("""produccion = [820, 915, 780, 860, 950, 700, 610]
# Lunes=0  Mar=1  Mié=2  Jue=3  Vie=4  Sáb=5  Dom=6

# 1. Producción de martes a jueves (índices 1, 2, 3)


# 2. Primeros 3 días de la semana


# 3. Fin de semana (sábado y domingo)


# 4. ¿Qué devuelve produccion[-3:]? ¿Por qué?
"""))

cells.append(md("""### Práctica: Gestión de Inventario ⏱ 7 min

Apliquen los métodos para actualizar el inventario paso a paso."""))

cells.append(code("""bodega = ["Agua 500ml", "Cola 2L", "Jugo 1L", "Agua 1L"]

# 1. Agreguen "Te 500ml" al final con .append()


# 2. Cambien posición 1 a "Cola 2.5L" (reasignación)


# 3. Eliminen "Jugo 1L" con .remove()


# 4. Ordenen la bodega con .sort()


# 5. Retiren el último producto con .pop() e impriman cuál era


# 6. Impriman la bodega final y su cantidad con len()
"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 3: CICLOS FOR
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 3. Ciclos `for`

Ya sabemos acceder a elementos uno por uno con `lista[i]`.  
¿Pero qué pasa cuando la lista tiene muchos valores?"""))

cells.append(code("""# Acceso manual — no escala
precios = [1.50, 1.25, 0.90, 1.10, 0.80]
print("Producto 1:", precios[0])
print("Producto 2:", precios[1])
print("Producto 3:", precios[2])
# ... ¿y si fueran 500 productos?"""))

cells.append(code("""# Con ciclo for — funciona para 5 o 5000
precios = [1.50, 1.25, 0.90, 1.10, 0.80]
for precio in precios:
    print("Producto:", precio)"""))

cells.append(md("""### Anatomía del ciclo `for`

```
                    ┌──────────────────┐
                    │  Inicio del for  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
              ┌──── │ ¿Quedan elementos?│ ────┐
              │ Sí  └────────┬─────────┘  No │
              ▼              │                ▼
    ┌───────────────┐        │      ┌───────────────┐
    │ variable =    │        │      │ Fin del ciclo │
    │ siguiente     │        │      └───────────────┘
    └───────┬───────┘        │
            ▼                │
    ┌───────────────┐        │
    │ Ejecutar      │        │
    │ bloque        │────────┘
    └───────────────┘  (vuelve a preguntar)
```

**Sintaxis:** `for variable in secuencia:`  
- `variable` toma un valor distinto en cada vuelta.  
- Solo el código **indentado** se repite."""))

cells.append(code("""productos = ["Coca-Cola", "Sprite", "Dasani"]
for producto in productos:
    print("Procesando:", producto)
print("Listo")  # Esto se ejecuta UNA sola vez al final"""))

cells.append(md("""### `for` con `range()`

Cuando necesitamos contar o acceder por índice, usamos `range()`.

| Forma | Genera | Nota |
|-------|--------|------|
| `range(5)` | `0, 1, 2, 3, 4` | Empieza en 0 |
| `range(2, 6)` | `2, 3, 4, 5` | El 6 no se incluye |
| `range(0, 10, 3)` | `0, 3, 6, 9` | Salta de 3 en 3 |
| `range(5, 0, -1)` | `5, 4, 3, 2, 1` | Cuenta hacia atrás |"""))

cells.append(code("""# Ejemplo con range
print("range(5):", list(range(5)))
print("range(2,6):", list(range(2, 6)))
print("range(0,10,3):", list(range(0, 10, 3)))"""))

cells.append(code("""# Uso práctico: recorrer dos listas en paralelo
rutas    = ["GYE Norte", "Manta", "Cuenca"]
ingresos = [15200, 12400, 6300]

for i in range(len(rutas)):
    print(rutas[i], "->", ingresos[i])"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 4: ACUMULADOR Y CONTADORES
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 4. Patrón Acumulador y Contadores

### Patrón Acumulador
Para sumar sobre una lista: iniciar en cero, sumar en cada vuelta."""))

cells.append(code("""pedidos = [120, 45, 200, 30, 180, 95, 60, 75]
total = 0
for cantidad in pedidos:
    total += cantidad   # Equivale a total = total + cantidad
print("Total de cajas:", total)"""))

cells.append(md("""### Práctica: Costo Total de Envío ⏱ 5 min

Cinco camiones salen de bodega. Cada uno tiene un costo de combustible distinto.  
Calculen el costo total y el promedio."""))

cells.append(code("""costos_combustible = [85.50, 120.00, 67.30, 95.80, 110.25]

# 1. Crear variable total inicializada en 0

# 2. Recorrer la lista con for y acumular cada costo

# 3. Imprimir el costo total

# 4. Calcular e imprimir el promedio (total / len(costos_combustible))
# Resultado esperado: Total = 478.85, Promedio ≈ 95.77
"""))

cells.append(md("""### Contadores

Un **contador** suma `1` cada vez que algo ocurre.  
Un **acumulador** suma **valores**."""))

cells.append(code("""ventas = [150, 80, 200, 45, 310, 60, 95]
meta = 100

cumplieron = 0
no_cumplieron = 0

for venta in ventas:
    if venta >= meta:
        cumplieron += 1
    else:
        no_cumplieron += 1

print("Cumplieron:", cumplieron)
print("No cumplieron:", no_cumplieron)"""))

cells.append(md("""### Práctica: Clasificación de Temperaturas ⏱ 6 min

Una línea de producción registró temperaturas cada hora.  
Necesitan **dos contadores** y **un acumulador**."""))

cells.append(code("""temperaturas = [72, 85, 91, 68, 88, 95, 74, 90, 82, 78]
limite = 85

# 1. Inicializar contadores (normal, alerta) y acumulador (suma_temp)

# 2. Recorrer temperaturas: si > limite → alerta, sino → normal. Acumular siempre.

# 3. Calcular promedio

# 4. Imprimir: "Normal: X | Alerta: Y | Prom: Z"
# Resultado esperado: Normal: 6, Alerta: 4, Promedio: 82.3
"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 5: FOR+IF Y OPERADOR IN
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 5. `for` + `if` y el Operador `in`

### `for` + `if` — La combinación poderosa

El `for` recorre; el `if` filtra. Juntos procesan **y** deciden."""))

cells.append(code("""pedidos = [120, 45, 200, 30, 180, 95, 60, 75]
minimo_despacho = 100

for cantidad in pedidos:
    if cantidad >= minimo_despacho:
        estado = "DESPACHAR"
    else:
        estado = "REVISAR"
    print(cantidad, "->", estado)"""))

cells.append(md("""### Práctica: Reporte de Ingresos ⏱ 7 min

4 productos despachados hoy. Calculen ingreso por producto y acumulen el total."""))

cells.append(code("""productos = ["Coca-Cola 600ml", "Sprite 500ml", "Dasani 1L", "Fanta 400ml"]
unidades  = [1200, 800, 2000, 500]
precios   = [1.50, 1.25, 0.90, 1.10]

# 1. Calcular ingreso de cada producto: unidades[i] * precios[i]
# 2. Acumular el ingreso total del día
# 3. Marcar con "SOBRE 1000" los que superen $1000
"""))

cells.append(md("""<details><summary><b>Ver solución — Reporte de Ingresos</b></summary>

```python
total = 0
for i in range(len(productos)):
    ingreso = unidades[i] * precios[i]
    total += ingreso
    etiqueta = ""
    if ingreso >= 1000:
        etiqueta = "SOBRE 1000"
    print(productos[i], "->", ingreso, etiqueta)

print("Total del dia:", total)
```

</details>"""))

cells.append(md("""### El operador `in`

`in` responde "¿este valor está en la lista?" con `True` o `False`.

| Expresión | Resultado |
|-----------|-----------|
| `"pera" in ["manzana", "pera", "uva"]` | `True` |
| `"kiwi" in ["manzana", "pera", "uva"]` | `False` |
| `"kiwi" not in ["manzana", "pera", "uva"]` | `True` |"""))

cells.append(code("""# Ejemplo básico
frutas = ["manzana", "pera", "uva"]
print("pera" in frutas)
print("kiwi" in frutas)
print("kiwi" not in frutas)"""))

cells.append(code("""# Uso práctico con if
zonas = ["Norte", "Sur", "Centro"]
pedido = "Oriente"
if pedido in zonas:
    print("Zona válida")
else:
    print("Zona NO habilitada")"""))

cells.append(md("""### Práctica: Validador de Pedidos ⏱ 5 min

Un restaurante solo prepara ciertos platos.  
Verifiquen cuáles pedidos se pueden atender."""))

cells.append(code("""menu = ["hamburguesa", "pizza", "ensalada", "tacos"]
pedidos = ["pizza", "sushi", "tacos", "pasta", "ensalada"]

# 1. Recorrer pedidos con for
# 2. Usar "in" para verificar si el pedido está en el menu
# 3. Si está: "Preparando: ...". Si no: "No disponible: ..."
# 4. Contar cuántos se rechazaron
"""))

cells.append(md("""<details><summary><b>Ver solución — Validador de Pedidos</b></summary>

```python
rechazados = 0
for plato in pedidos:
    if plato in menu:
        print("Preparando:", plato)
    else:
        print("No disponible:", plato)
        rechazados += 1
print("Rechazados:", rechazados)
```

</details>"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 6: RETOS PRÁCTICOS
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 6. Retos Prácticos

### Reto 1: Evaluación de Notas ⏱ 6 min

Un profesor tiene las notas de 6 estudiantes y necesita clasificarlas.

**Reglas:** Nota ≥ 90 → "Sobresaliente" | Nota ≥ 70 → "Aprobado" | Nota < 70 → "Reprobado" """))

cells.append(code("""nombres = ["Ana", "Luis", "Maria", "Carlos", "Sofia", "Diego"]
notas   = [92, 67, 85, 55, 78, 43]

# Recorrer las listas, clasificar cada nota,
# imprimir nombre + nota + clasificación.
# Al final, contar cuántos reprobaron.
"""))

cells.append(md("""### Reto 2: Monitor de Calidad de Aire ⏱ 7 min

Un sensor mide PM2.5 (partículas finas en el aire, en μg/m³).

**Escala:** Valor ≤ 50 → "Normal" | 51–100 → "Precaución" | > 100 → "Alerta" """))

cells.append(code("""horas = ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00"]
pm25  = [42, 58, 67, 105, 88, 120]

# Imprimir cada hora con su valor y estado.
# Contar cuántas horas quedaron en "Alerta".
"""))

cells.append(md("""### Reto 3: Plan de Carga para Viaje Eléctrico ⏱ 10 min

Estimen autonomía por tramo y decidan cuándo recargar."""))

cells.append(code("""tramos_km = [42, 55, 38, 64, 29]
consumo_kwh_por_100km = [14, 16, 13, 18, 12]
bateria_inicial_kwh = 32
umbral_recarga = 8

# 1. Calcular consumo por tramo: km * consumo_100 / 100
# 2. Restar de la batería disponible
# 3. Si la batería cae por debajo del umbral, marcar "RECARGAR"
#    y simular recarga completa (bateria = bateria_inicial_kwh)
# 4. Contar cuántas recargas se recomendaron
"""))

# ══════════════════════════════════════════════════════════════
#  SECCIÓN 7: PROYECTO INTEGRADOR
# ══════════════════════════════════════════════════════════════
cells.append(md("""---
## 7. Proyecto Integrador: Simulador de Ruta de Distribución ⏱ 20 min

Un camión sale del centro de distribución con carga limitada.  
Debe visitar 6 clientes y decidir en cada parada si puede entregar.

### Reglas por parada:
1. ¿El producto está en el `catalogo`? (`in`)
2. Si no está → "NO DISPONIBLE"
3. Si está pero la `carga` no alcanza → "SIN STOCK"
4. Si se puede → entregar, restar de `carga`, sumar ingreso

### Reportar al final:
- Entregas exitosas (contador)
- Pedidos rechazados (contador)
- Ingreso total (acumulador)
- Cajas restantes en el camión"""))

cells.append(code("""# Datos de la ruta
clientes = ["Tienda Sol", "Mini ABC", "Super 10",
            "Bodega LP", "Market 7", "Don Pepe"]
pedidos  = [30, 15, 50, 20, 40, 10]
productos_pedidos = ["Coca-Cola", "Sprite", "Fanta",
                     "Coca-Cola", "Dasani", "Sprite"]

# Carga del camión
catalogo = ["Coca-Cola", "Sprite", "Dasani"]
carga = 80
precio_caja = 12.50

# Escriban su solución aquí
"""))

cells.append(md("""---

**Clase 3 completada.**  
En la próxima clase veremos funciones, diccionarios y manejo de archivos."""))

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

with open("/root/Arca/clase-03/Clase_03_Iteracion_y_Logica_Aplicada.ipynb", "w") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook creado exitosamente")
print(f"Celdas totales: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type']=='markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type']=='code')}")
