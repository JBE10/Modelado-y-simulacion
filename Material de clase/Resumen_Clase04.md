# Resumen Clase 4: El Arte de la Aproximación Numérica

**Materia:** Modelado y Simulación
**Profesor:** Ing. Omar Cáceres
**Tema:** Integración Numérica (Reglas de Newton-Cotes, cotas de error matemático y limitaciones del hardware).

---

## 1. El Fundamento: ¿Por qué aproximar numéricamente?

Buscamos un valor numérico $\hat{I}$ que sea una excelente aproximación de una integral definida $I = \int_{a}^{b} f(x) dx$. Muchas funciones continuas y de gran importancia aplicativa (como la integral Gaussiana en estadística $\int e^{-x^2} dx$) **no tienen una primitiva elemental**, es decir, el cálculo analítico y exacto es imposible. 

La estrategia principal consiste en reemplazar la función $f(x)$ por un polinomio más simple (constante, lineal, cuadrático) cuya área bajo la curva se pueda calcular geométricamente sumando figuras simples como rectángulos, trapecios o parábolas.

### Lema del Valor Medio (Soporte Teórico)
Para acotar matemáticamente los errores necesitamos utilizar el Lema del Valor Medio para integrales. Si $f$ y $g$ son continuas en $[a, b]$, y $g(x) \geq 0$, entonces existe un valor intermedio $\xi \in [a, b]$ tal que:
$$ \int_{a}^{b} f(x)g(x) dx = f(\xi) \int_{a}^{b} g(x) dx $$
Esto nos va a permitir "sacar" la función $f(x)$ y sus derivadas parciales fuera de las integrales para formular el error empírico de las aproximaciones.

---

## 2. La Regla del Rectángulo (Punto Medio)

Aproxima el área bajo la curva utilizando múltiples rectángulos cuya altura se valúa en el punto medio de cada subintervalo. Es equivalente a aproximar $f(x)$ mediante un polinomio interpolador de **grado cero** (constante).

### Análisis de la Celda Unitaria (1 Subintervalo $[-h/2, h/2]$)
- **Aproximación:** $\int_{-h/2}^{h/2} f(x) dx \approx h \cdot f(0)$
- **Error Celda Unitaria:** $E_{r1} = \frac{h^3}{24} f''(\xi)$.
  *El error de una sola celda depende del cubo de su extensión temporal ($h^3$) y de qué tan plana es la curva ($f''$).*

### Fórmula Compuesta en el intervalo completo $[a, b]$
En vez de evaluar desde $[-h/2, h/2]$, particionamos la curva en $n$ subintervalos de ancho $h = (b-a)/n$.
- **Fórmula:** $I_r = h \sum_{k=0}^{n-1} f(x_{k+1/2})$
- **Error Global Teórico:** $\mathcal{O}(h^2) \longrightarrow E_{r} = \frac{(b-a)h^2}{24} f''(\xi)$

---

## 3. La Regla del Trapecio

En lugar de aproximar mediante escalones constantes, forma un trapecio conectando los valores en los extremos de un subintervalo usando un polinomio de **grado uno** (recta o secante).

### Análisis de la Celda Unitaria (1 Subintervalo $[0, h]$)
- **Aproximación:** $\int_{0}^{h} f(x) dx \approx \frac{h}{2}[f(0) + f(h)]$
- **Error Celda Unitaria:** $E_{t1} = -\frac{h^3}{12} f''(\xi)$

### Fórmula Compuesta en el intervalo completo $[a, b]$
Se suman iterativamente las áreas de los trapecios contiguos. Nótese que los puntos de enlace interiores se suman dos veces.
- **Fórmula:** $I_t = \frac{h}{2} \left[ f(x_0) + 2\sum_{k=1}^{n-1} f(x_k) + f(x_n) \right]$
- **Error Global Teórico:** $\mathcal{O}(h^2) \longrightarrow E_{t} = -\frac{(b-a)h^2}{12} f''(\xi)$

> [!tip] Rectángulo vs Trapecio
> Ambos métodos proveen una **Ganancia Cuadrática** global $\mathcal{O}(h^2)$ (reducir el bloque $h$ a la mitad minimiza el error total en aproximadamente 4 veces). Pese a la aparente sofisticación del Trapecio, el lado del error teórico global nos revela que el Rectángulo suele ser el doble de preciso debido a sus coeficientes de serie de Taylor ($1/24$ vs $1/12$).

---

## 4. La Regla de Simpson 1/3 (Parabólica)

Representa un salto exponencial en precisión. Interpola cada área no como una recta sino como una **parábola** (polinomio de **grado dos**) valiéndose de la conjunción de $3$ puntos adyacentes $(x_i, x_{i+1}, x_{i+2})$.

### Fórmula Compuesta en el intervalo completo $[a, b]$
Dado el uso de $3$ puntos por partición, **$n$ debe ser rigurosamente par**.
$I_{s} = \frac{h}{3} \left[ f(x_0) + 4\sum_{i \text{ impar}}f(x_i) + 2\sum_{i \text{ par}}f(x_i) + f(x_n) \right]$

### Análisis Avanzado de Error
Para su celda unitaria de tamaño $[-h, h]$, el error depende ya no de la 2da derivada, sino de la **cuarta derivada**.
- **Error Global Teórico:** $\mathcal{O}(h^4) \longrightarrow E_{simp} = -\frac{(b-a)h^4}{180} f^{(4)}(\xi)$
> [!important] Exactitud Extrema 
> $\mathcal{O}(h^4)$ implica que cortar $h$ a la mitad achica el error truncado a $\frac{1}{16}$ partes del tamaño original. Además, como el error de truncamiento depende pura y exclusivamente de $f^{(4)}(\xi)$, se ratifica que **Simpson es matemáticamente exacto para resolver polinomios de hasta grado 3**, al ser nula la cuarta derivada de los mismos.

---

## 5. La Regla de Simpson 3/8 (Cúbica)

Subdivisión extra que interpola 4 nodos usando un **polinomio cúbico** (grado 3).
- **Restricción Excluyente:** $n$ debe ser **múltiplo de 3**.
- **Fórmula de Segmentos:** $\approx \frac{3h}{8} [f(a)+3f(x_1)+3f(x_2)+f(b)]$
- **Error Global:** $\mathcal{O}(h^4) \longrightarrow E = -\frac{(b-a)h^4}{80} f^{(4)}(\xi)$

---

## 6. Elección de Especificaciones y Cotas de Error ($h_{max}$)

En el mundo profesional no adivinamos precisiones por ensayo y error. Si nuestro negocio exige una tolerancia máxima aceptable de desvío al medir (ejemplo $\tau = 10^{-5}$), procedemos al camino inverso buscando un parámetro matemático inquebrantable $h_{max}$.

Como no es trivial conocer $\xi$, usamos el tope máximo que adopta la respectiva derivada en ese intervalo. Definimos valores brutos como:
- $M_2 = \max |f^{(2)}(x)|$ (para Rectángulo o Trapecio)
- $M_4 = \max |f^{(4)}(x)|$ (para Simpson)

### Ejemplo de diseño estructural ($h_{max}$ para Simpson)
A partir de la inecuación de error absoluto global:
$$ \left| \frac{b-a}{180} M_4 \cdot h^4 \right| \leq \tau \implies h_{max} = \sqrt[4]{\frac{180 \cdot \tau}{(b-a) M_4}} $$

Si adoptamos un paso operativo de programación de iteraciones $h < h_{max}$, garantizamos sin excepción que el resultado cumpla con la cota de desviación $\tau$. 

---

## 7. La Realidad Informática: El Dilema del Cómputo Ciego

Se podría tender al ideal irreal de hacer $h$ tan increíblemente minúsculo argumentando que "las fórmulas indican que su decremento achica el error a cero".
En la integración intervienen computadoras operando datos **Float**.
- El **Error de Truncamiento** (la discrepancia analítica mostrada arriba por fórmulas matemáticas) se encoge junto con un $h$ menor.
- El **Error de Redondeo Finito** crece a medida que achicamos $h$ de golpe, porque requerimos de millones de iteraciones más, sumando en cada bucle una infinitesimal imprecisión de los bytes del CPU. 

Llega un valle ($h$ óptimo) donde achicarlo un ápice más invierte la gráfica e infesta de basuras sistémicas a los decímales de nuestro cómputo analítico ideal, destruyéndolo.

---

## 8. Sintésis Vectorial de NumPy

Las reglas compuestas vistas en Python:

```python
import numpy as np

# Datos funcionales
n = 100 # Subintervalos
h = (b - a) / n
x = np.linspace(a, b, n + 1)
y = f(x)  # Vectorización O(1) de NumPy

# ---- Regla del Trapecio ----
integral_t = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# ---- Regla de Simpson 1/3 (Requiere que 'n' sea par) ----
# y[1:-1:2] -> Indices impares: 1, 3, 5, ...
# y[2:-2:2] -> Indices pares interiores: 2, 4, 6, ...
integral_s13 = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

# ---- Regla de Simpson 3/8 (Requiere que 'n' sea múltiplo de 3) ----
integral_s38 = (3 * h / 8) * (y[0] + y[-1] 
               + 3 * np.sum(y[1:-1:3]) 
               + 3 * np.sum(y[2:-1:3]) 
               + 2 * np.sum(y[3:-2:3]))
```
