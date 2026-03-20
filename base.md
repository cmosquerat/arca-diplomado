# Curso Completo: Introducción a la Programación y Ciencias de la Computación
**Fuente:** [freeCodeCamp.org - Introduction to Programming and Computer Science](https://www.youtube.com/watch?v=zOjov-2OZ0E)

Este documento contiene un desglose exhaustivo de todos los conceptos de ciencias de la computación presentados en el curso, diseñados para aplicar a cualquier lenguaje de programación.

---

## 1. Introducción y Conceptos Básicos

### ¿Qué es la programación? [00:00:01]
* **Definición práctica:** Es el proceso de intentar que una computadora complete una tarea específica sin cometer errores.
* **Analogía del Lego:** Imagina que tienes un amigo muy poco inteligente que debe armar un set de Lego sin el manual. Debes darle instrucciones exactas (dónde y cómo colocar cada pieza). Si omites un solo detalle, el set se arruina. Las computadoras son igual de "tontas"; solo hacen exactamente lo que se les ordena.
* **El problema del idioma:** Las computadoras no hablan idiomas humanos, solo entienden **código máquina (sistema binario de ceros y unos)**.

### Lenguajes de Programación [00:04:26]
* Escribir en código máquina es casi imposible para un humano. Los lenguajes de programación actúan como un **intermediario o traductor** entre el inglés (o lógica humana) y el código máquina.
* **Tipos de lenguajes:**
  * Propósito general: Python, Java, C++.
  * Propósito específico: HTML/CSS (sitios web), Robot C (robótica).
* **Niveles de lenguajes:**
  * *Bajo nivel (ej. Assembly, C):* Más cercanos al código máquina (ceros y unos). Son más rápidos pero más difíciles de leer.
  * *Alto nivel (ej. Python, Java):* Más cercanos al lenguaje humano. Tienen mayor nivel de abstracción.

### El Entorno de Desarrollo (IDE) y la Sintaxis [00:06:22]
* **IDE (Integrated Development Environment):** Es el software donde se escribe, depura y compila el código (actúa como Microsoft Word pero para programadores). Ofrece autocompletado, detección de errores y un sistema jerárquico de archivos.
* **Sintaxis [00:08:28]:** Son las reglas gramaticales estrictas del lenguaje. Un error mínimo (como olvidar un punto y coma `;` en Java o JavaScript) hará que todo el programa falle. 

---

## 2. Operaciones Básicas y Variables

### La Consola y la función Print [00:11:51]
* **La Consola:** Es una interfaz de texto utilizada por los desarrolladores para ver qué está sucediendo "detrás de escena" en el programa. No está diseñada para que el usuario final la vea.
* **Sentencia Print (Imprimir):** Es la instrucción fundamental para enviar texto o resultados matemáticos a la consola. Ej: `print("Hello World")`.

### Matemáticas Básicas y Strings [00:15:06]
* La computadora puede realizar aritmética simple (`+`, `-`, `*`, `/`).
* **Operador Módulo (`%`):** Devuelve el *residuo* de una división. Es extremadamente útil, por ejemplo, para saber si un número es par o impar (si `numero % 2 == 0`, es par).
* **Concatenación:** Es el proceso de unir cadenas de texto (strings) con variables u otros textos usando el signo `+`. 
  * *Precaución:* El texto literal debe ir entre comillas (`"4"` es texto, `4` es un número entero). Sumar un número con un texto puede generar errores si no se maneja bien.

### Variables [00:20:47]
* **Definición:** Son espacios en la memoria del computador (comparadas con "cajas de cartón") que almacenan información que puede ser referenciada o modificada más adelante.
* **Tipos de datos primitivos:**
  1. `int` (Integer / Entero): Números enteros positivos o negativos (sin decimales).
  2. `boolean` (Booleano): Solo puede almacenar `True` (Verdadero) o `False` (Falso).
  3. `float` y `double`: Números con puntos decimales (Double tiene mayor precisión/capacidad que float).
  4. `String` (Cadena de texto): Palabras, frases o caracteres múltiples.
  5. `char` (Caracter): Un único símbolo o letra.
* **Manipulación de Variables [00:25:15]:** * Las variables pueden actualizarse dinámicamente (ej. sumar +1 al `score` de un jugador).
  * **Convención de nombres (Camel Case):** Regla de escritura donde no se capitaliza la primera palabra, pero sí las siguientes para facilitar la lectura. Ej: `playerScore` en lugar de `playerscore`.

---

## 3. Lógica y Flujo de Control

### Declaraciones Condicionales (If Statements) [00:31:57]
Permiten que el programa tome caminos distintos dependiendo de ciertas condiciones (evaluadas como booleanos: verdadero/falso).
* **If:** Si la condición entre paréntesis es verdadera, ejecuta el código entre llaves `{}`.
* **Else If:** Si el `If` anterior fue falso, evalúa una nueva condición. Puedes encadenar múltiples `Else If`.
* **Else:** El caso por defecto. Se ejecuta si ninguna de las condiciones anteriores fue verdadera.
* **Switch:** Una forma más limpia de escribir múltiples declaraciones `If/Else If` cuando se evalúa una sola variable contra muchos casos específicos (incluye un caso `default`).

### Arreglos (Arrays) [00:38:01]
* **Definición:** Estructuras utilizadas para almacenar listas de múltiples variables del mismo tipo bajo un solo nombre (ej. una lista de compras o un top 10 de puntajes).
* **Regla de Oro de la Indexación:** En ciencias de la computación, **siempre se empieza a contar desde el cero**. El primer elemento está en el índice 0, el segundo en el índice 1, etc.
* **Tamaño fijo:** Al crear un Array estándar, debes definir su tamaño. Si creas un array para 8 elementos, no puedes agregar un noveno elemento después sin causar un error (analogía de una estantería de libros con espacio limitado).
* **Arreglos 2D:** Arreglos dentro de arreglos (como una matriz matemática o una hoja de cálculo con filas y columnas).

### Bucles (Loops) [00:44:31]
Ejecutan un bloque de instrucciones de forma repetida.
* **Bucle For (`for`):** Ideal cuando sabes exactamente cuántas veces quieres repetir algo. Tiene 3 partes: el valor inicial, la condición de parada, y la operación de incremento (ej. de 0 a 10, sumando de a 1).
* **Bucle For-Each:** Diseñado específicamente para recorrer (iterar) automáticamente todos los elementos dentro de un Arreglo o Lista.
* **Bucle While (`while`):** Se repite continuamente *mientras* una condición siga siendo verdadera. Es común en videojuegos (el juego corre en un bucle infinito hasta que el jugador presiona "Salir").
* **Bucle Do-While:** Similar al `While`, pero garantiza que el código interior se ejecutará **al menos una vez** antes de comprobar la condición.

---

## 4. Resolución de Problemas y Funciones

### Errores (Bugs) [00:49:39]
Hay tres tipos principales de errores que todo programador enfrenta:
1. **Errores de Sintaxis:** Olvido de signos, mala ortografía en comandos. El IDE suele detectarlos y no deja ejecutar el código.
2. **Errores de Tiempo de Ejecución (Runtime):** El código está bien escrito, pero le pides al computador algo imposible (ej. un bucle infinito que consume toda la memoria y crashea el programa).
3. **Errores Lógicos:** Los más difíciles de rastrear. El programa corre perfectamente, pero da el resultado equivocado (ej. sumar en lugar de multiplicar por usar el símbolo incorrecto).

### Depuración (Debugging) [00:54:23]
* **Estrategia 1 (Print statements):** Colocar sentencias `print()` en varias partes del código para revisar el valor temporal de las variables y ver dónde se rompe la lógica.
* **Estrategia 2 (Breakpoints):** Herramienta del IDE para "pausar" el programa en una línea específica y evaluar el estado de todo el sistema en ese milisegundo.
* **Comentar código:** Convertir partes sospechosas de código en "comentarios" para que la máquina los ignore. Si el programa funciona al ignorar ese bloque, has encontrado al culpable.

### Funciones [01:00:30]
* **Definición:** Son bloques de código "empaquetados" bajo un nombre. Permiten reutilizar lógica sin tener que reescribirla. (El comando `print()` es una función nativa).
* **Argumentos/Parámetros:** Información que se le "pasa" a la función para que trabaje. (Ej. Una función `Sumar(a, b)` requiere los números `a` y `b` como argumentos).
* **Retorno (Return):** Las funciones pueden calcular datos y "devolverlos" al programa, o simplemente ejecutar acciones sin devolver nada (conocidas como funciones `Void`).

### Importar Librerías [01:09:57]
* No tienes que inventar la rueda. Existen "librerías" (colecciones de funciones temáticas hechas por otros, ej. librerías matemáticas o de análisis de datos).
* En lugar de cargar todo el peso de una librería gigante, es mejor especificar exactamente qué funciones necesitas (ej. `from math import factorial`).

---

## 5. Estructuras Avanzadas y Algoritmos

### ArrayLists y Diccionarios [01:22:00]
* **ArrayLists (Listas):** A diferencia de los Arrays clásicos, los ArrayLists son dinámicos. Empiezan con un tamaño (ej. 10), pero si agregas un 11vo elemento, el sistema automáticamente asigna más memoria y "crece".
* **Diccionarios (Hash Maps):** No usan índices numéricos (0, 1, 2). Guardan información en pares **Clave-Valor** (Key-Value pairs). Ej. Clave: "Manzana", Valor: "$1.00". Para buscar el valor, invocas la clave. Cada clave debe ser única e irrepetible.

### Algoritmos de Búsqueda [01:27:33]
* **Búsqueda Lineal (Linear Search):** Busca el elemento revisando desde el principio de la lista, uno por uno. Funciona en listas desordenadas, pero es ineficiente en grandes volúmenes de datos.
* **Búsqueda Binaria (Binary Search):** Requiere que la lista esté **ordenada**. Compara el objetivo con el elemento de la mitad de la lista. Si el objetivo es menor, descarta toda la mitad superior y repite el proceso en la mitad inferior. Es exponencialmente más rápido.

### Recursión [01:36:23]
* Es un concepto complejo donde **una función se llama a sí misma** desde su interior.
* Requiere un **Caso Base (Base Case)**: Una condición de salida imperativa. Si no existe, la función se llamará infinitamente creando un error de desbordamiento de pila (*Stack Overflow*).
* Funciona resolviendo problemas grandes rompiéndolos en el mismo problema a una escala menor hasta alcanzar el caso base.

---

## 6. Prácticas de Ingeniería y Futuro

### Pseudocódigo y Planificación [01:43:44]
* **Nunca comiences a programar directamente en el IDE.** Gran parte del trabajo es pensar en el problema lógico primero.
* **Pseudocódigo:** Escribir la lógica del programa en lenguaje humano sin preocuparse por la sintaxis estricta.
* **Técnicas:**
  1. *Diagramas de flujo:* Excelente para mapear visualmente funciones condicionales y bucles.
  2. *Escritura secuencial (Write-up):* Escribir cronológicamente qué debe hacer la aplicación paso a paso.
  3. *Planificación de funcionalidades:* Mapear una jerarquía de qué botones, opciones y sub-menús tendrá el programa final.

### Elección de un Lenguaje y Siguientes Pasos [01:50:44]
* **Para Web:** HTML (estructura), CSS (estilos), JavaScript (interactividad).
* **Para Software y propósito general:** Java, C++, Python. Python destaca hoy en día por su legibilidad y uso masivo en análisis de datos.
* **¿Cómo mejorar?**
  * La teoría no basta. Hay que enfrentar la "página en blanco" escribiendo proyectos reales.
  * Sitios web recomendados para practicar lógica y algoritmos: **CodingBat, Coderbyte, y HackerRank.**