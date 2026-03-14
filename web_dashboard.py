import os
import math

from flask import Flask, render_template_string, request

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Numerical Lab - Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0b1120;
      --card: #111827;
      --soft: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --primary: #3b82f6;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #0f172a 0%, var(--bg) 55%);
      color: var(--text);
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 28px; }
    .header {
      display: flex; justify-content: space-between; align-items: end; margin-bottom: 20px;
      border-bottom: 1px solid var(--soft); padding-bottom: 14px;
    }
    .title { margin: 0; font-size: 28px; }
    .subtitle { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
    .pill { border: 1px solid #334155; border-radius: 999px; padding: 6px 10px; color: #bfdbfe; font-size: 12px; }
    .card {
      background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
      border: 1px solid var(--soft);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 6px 22px rgba(2, 6, 23, 0.35);
    }
    .h2 { margin: 0 0 14px; font-size: 20px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 12px; }
    @media (max-width: 900px) { .grid-3, .grid-2, .charts, .metrics { grid-template-columns: 1fr; } }
    label { display: block; margin-bottom: 6px; color: #cbd5e1; font-size: 13px; font-weight: 600; }
    input, select, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #334155;
      background: #0b1220;
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
      outline: none;
    }
    input:focus, select:focus { border-color: #60a5fa; }
    button {
      margin-top: 14px;
      background: linear-gradient(90deg, #2563eb, #3b82f6);
      border: none;
      font-weight: 700;
      cursor: pointer;
    }
    .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 8px; }
    .metric { border: 1px solid #334155; background: #0b1220; border-radius: 10px; padding: 10px; }
    .metric .k { color: #93c5fd; font-size: 11px; text-transform: uppercase; letter-spacing: .02em; }
    .metric .v { margin-top: 6px; font-size: 18px; font-weight: 600; }
    .ok { color: #86efac; font-weight: 700; }
    .error { color: #fca5a5; font-weight: 700; }
    .note { color: #cbd5e1; margin-top: 8px; line-height: 1.4; }
    .table-wrap { max-height: 360px; overflow: auto; border: 1px solid var(--soft); border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px; border-bottom: 1px solid var(--soft); text-align: right; }
    th { position: sticky; top: 0; background: #0b1220; color: #93c5fd; }
    .charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    .chart-box { border: 1px solid var(--soft); border-radius: 10px; padding: 12px; }
    .top-note { color: #93c5fd; font-size: 13px; margin-bottom: 8px; }
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1 class="title">Numerical Lab</h1>
      <p class="subtitle">Dashboard profesional en localhost. Calculo numerico ejecutado en Python.</p>
    </div>
    <div class="pill">Python Core + Flask UI</div>
  </div>

  <div class="card">
    <h2 class="h2">Configurar ejecucion</h2>
    <div class="top-note">Funciones soportadas: <code>sin</code>, <code>cos</code>, <code>tan</code>, <code>exp</code>, <code>log</code>, <code>sqrt</code>, <code>abs</code>, <code>cbrt</code>, constantes <code>pi</code>, <code>e</code>.</div>
    <form method="post">
      <div class="grid-3">
        <div>
          <label>Metodo</label>
          <select name="metodo" id="metodo">
            <option value="biseccion" {% if form.metodo == "biseccion" %}selected{% endif %}>Biseccion</option>
            <option value="punto_fijo" {% if form.metodo == "punto_fijo" %}selected{% endif %}>Punto Fijo</option>
          </select>
        </div>
        <div>
          <label>Tolerancia</label>
          <input name="tol" value="{{ form.tol }}" />
        </div>
        <div>
          <label>max_iter</label>
          <input name="max_iter" value="{{ form.max_iter }}" />
        </div>
      </div>

      <div class="grid-2">
        <div>
          <label>f(x)</label>
          <input name="expr_f" value="{{ form.expr_f }}" />
          <div class="hint">Obligatoria en biseccion y en punto fijo.</div>
        </div>
        <div>
          <label>g(x) (solo punto fijo)</label>
          <input name="expr_g" value="{{ form.expr_g }}" />
          <div class="hint">En punto fijo se itera <code>x(n+1)=g(x(n))</code>.</div>
        </div>
      </div>

      <div class="grid-2">
        <div id="bloque_ab">
          <div class="grid-2" style="margin-top:0">
            <div>
              <label>a</label>
              <input name="a" value="{{ form.a }}" />
            </div>
            <div>
              <label>b</label>
              <input name="b" value="{{ form.b }}" />
            </div>
          </div>
        </div>
        <div id="bloque_x0">
          <label>x0</label>
          <input name="x0" value="{{ form.x0 }}" />
        </div>
      </div>

      <button type="submit">Calcular</button>
      <div class="hint">Tip: podes usar <code>^</code> o <code>**</code> para potencias.</div>
    </form>
  </div>

  {% if error %}
    <div class="card"><div class="error">Error: {{ error }}</div></div>
  {% endif %}

  {% if result %}
    <div class="card">
      <h2 class="h2">Resultado</h2>
      <div class="metrics">
        <div class="metric"><div class="k">Metodo</div><div class="v">{{ result.metodo }}</div></div>
        <div class="metric"><div class="k">Raiz</div><div class="v">{{ "%.12f"|format(result.raiz) }}</div></div>
        <div class="metric"><div class="k">Iteraciones</div><div class="v">{{ result.iteraciones }}</div></div>
        <div class="metric"><div class="k">{{ residuo_label }}</div><div class="v">{{ "%.4e"|format(result.residuo) }}</div></div>
        <div class="metric"><div class="k">Convergencia</div><div class="v">{{ "SI" if result.convergio else "NO" }}</div></div>
      </div>
      <p class="{{ 'ok' if result.convergio else 'error' }}" style="margin-top:10px">
        {{ "Convergio segun tolerancia." if result.convergio else "No convergio en el maximo de iteraciones." }}
      </p>
      {% if result.justificacion %}
      <p class="note"><strong>Justificacion:</strong> {{ result.justificacion }}</p>
      {% endif %}
    </div>

    <div class="card">
      <h2 class="h2">Tabla de iteraciones</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              {% for col in columns %}
                <th>{{ col }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for row in result.historial %}
              <tr>
                {% for col in columns %}
                  <td>{{ row[col] }}</td>
                {% endfor %}
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card charts">
      <div class="chart-box">
        <h3 style="margin-top:0">Convergencia</h3>
        <canvas id="chartValor"></canvas>
      </div>
      <div class="chart-box">
        <h3 style="margin-top:0">Error por iteracion</h3>
        <canvas id="chartError"></canvas>
      </div>
    </div>
  {% endif %}
</div>

<script>
  function refreshInputs() {
    const metodo = document.getElementById("metodo").value;
    const ab = document.getElementById("bloque_ab");
    const x0 = document.getElementById("bloque_x0");
    if (metodo === "biseccion") {
      ab.style.display = "block";
      x0.style.display = "none";
    } else {
      ab.style.display = "none";
      x0.style.display = "block";
    }
  }
  document.getElementById("metodo").addEventListener("change", refreshInputs);
  refreshInputs();
</script>

{% if result %}
<script>
  const labels = {{ labels|tojson }};
  const serieValor = {{ serie_valor|tojson }};
  const serieError = {{ serie_error|tojson }};
  const labelValor = {{ label_valor|tojson }};
  new Chart(document.getElementById("chartValor"), {
    type: "line",
    data: { labels, datasets: [{ label: labelValor, data: serieValor, borderColor: "#60a5fa", tension: 0.15, fill: false }] },
    options: { responsive: true, maintainAspectRatio: true }
  });
  new Chart(document.getElementById("chartError"), {
    type: "line",
    data: { labels, datasets: [{ label: "error", data: serieError, borderColor: "#f87171", tension: 0.15, fill: false }] },
    options: { responsive: true, maintainAspectRatio: true }
  });
</script>
{% endif %}
</body>
</html>
"""


def cbrt_safe(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


ALLOWED_MATH = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "cbrt": cbrt_safe,
    "pi": math.pi,
    "e": math.e,
}


def normalize_expr(expr):
    return (expr or "").strip().replace("^", "**")


def build_function(expr):
    expr = normalize_expr(expr)
    if not expr:
        raise ValueError("La expresion esta vacia.")

    def f(x):
        return eval(expr, {"__builtins__": {}}, {"x": x, **ALLOWED_MATH})

    return f


def parse_float(name, value):
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"'{name}' debe ser numerico: {e}") from e


def parse_int(name, value):
    try:
        out = int(value)
    except Exception as e:
        raise ValueError(f"'{name}' debe ser entero: {e}") from e
    if out < 1:
        raise ValueError(f"'{name}' debe ser >= 1.")
    return out


def biseccion_python(expr_f, a, b, tol, max_iter):
    f = build_function(expr_f)
    historial = []

    try:
        fa = float(f(a))
        fb = float(f(b))
    except Exception as e:
        raise ValueError(f"No se pudo evaluar f en los extremos: {e}") from e

    if not math.isfinite(fa) or not math.isfinite(fb):
        raise ValueError("f(a) o f(b) no es finito (NaN/Inf).")
    if fa * fb > 0:
        raise ValueError("El intervalo [a,b] no encierra raiz: f(a) y f(b) tienen mismo signo.")

    for i in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        error = abs(b - a) / 2.0
        try:
            fc = float(f(c))
        except Exception as e:
            return {
                "metodo": "biseccion",
                "raiz": c,
                "iteraciones": i - 1,
                "residuo": abs(historial[-1]["fc"]) if historial else float("nan"),
                "convergio": False,
                "justificacion": f"Se detuvo por error al evaluar f(c) en iter {i}: {e}",
                "historial": historial,
            }
        if not math.isfinite(fc):
            return {
                "metodo": "biseccion",
                "raiz": c,
                "iteraciones": i - 1,
                "residuo": abs(historial[-1]["fc"]) if historial else float("nan"),
                "convergio": False,
                "justificacion": f"Se detuvo por valor no finito en iter {i}.",
                "historial": historial,
            }

        historial.append({"iter": i, "a": a, "b": b, "c": c, "fc": fc, "error": error})

        if abs(fc) < tol or error < tol:
            return {
                "metodo": "biseccion",
                "raiz": c,
                "iteraciones": i,
                "residuo": abs(fc),
                "convergio": True,
                "justificacion": f"Convergio en iter {i}: se cumplio |f(c)| < tol o error < tol.",
                "historial": historial,
            }

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raiz = 0.5 * (a + b)
    residuo = abs(float(f(raiz)))
    requerido = math.ceil(math.log2(abs((b - a) / tol))) if tol > 0 and b != a else max_iter
    return {
        "metodo": "biseccion",
        "raiz": raiz,
        "iteraciones": max_iter,
        "residuo": residuo,
        "convergio": False,
        "justificacion": f"No convergio: max_iter={max_iter}. Se estiman ~{requerido} iteraciones para esta tolerancia.",
        "historial": historial,
    }


def punto_fijo_python(expr_f, expr_g, x0, tol, max_iter):
    f = build_function(expr_f)
    g = build_function(expr_g)
    historial = []
    xn = float(x0)

    for i in range(1, max_iter + 1):
        try:
            x_next = float(g(xn))
        except Exception as e:
            return {
                "metodo": "punto_fijo",
                "raiz": xn,
                "iteraciones": i - 1,
                "residuo": historial[-1]["residual"] if historial else float("nan"),
                "convergio": False,
                "justificacion": f"Se detuvo en iter {i}: no se pudo evaluar g(x_n): {e}",
                "historial": historial,
            }
        if not math.isfinite(x_next):
            return {
                "metodo": "punto_fijo",
                "raiz": xn,
                "iteraciones": i - 1,
                "residuo": historial[-1]["residual"] if historial else float("nan"),
                "convergio": False,
                "justificacion": f"Se detuvo en iter {i}: g(x_n) dio NaN/Inf.",
                "historial": historial,
            }

        error = abs(x_next - xn)
        try:
            residual = abs(float(f(x_next)))
        except Exception as e:
            return {
                "metodo": "punto_fijo",
                "raiz": xn,
                "iteraciones": i - 1,
                "residuo": historial[-1]["residual"] if historial else float("nan"),
                "convergio": False,
                "justificacion": f"Se detuvo en iter {i}: no se pudo evaluar f(x_n+1): {e}",
                "historial": historial,
            }

        historial.append({"iter": i, "x_n": xn, "x_next": x_next, "error": error, "residual": residual})

        if error < tol:
            return {
                "metodo": "punto_fijo",
                "raiz": x_next,
                "iteraciones": i,
                "residuo": residual,
                "convergio": True,
                "justificacion": f"Convergio en iter {i}: se cumplio |x_n+1 - x_n| < tol.",
                "historial": historial,
            }

        xn = x_next

    raiz = xn
    residuo = abs(float(f(raiz)))
    return {
        "metodo": "punto_fijo",
        "raiz": raiz,
        "iteraciones": max_iter,
        "residuo": residuo,
        "convergio": False,
        "justificacion": "No convergio: se alcanzo max_iter. Revisa g(x), x0 o aumenta iteraciones.",
        "historial": historial,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form = {
        "metodo": "biseccion",
        "expr_f": "x^3 - x - 2",
        "expr_g": "cbrt(x + 2)",
        "a": "1",
        "b": "2",
        "x0": "1.5",
        "tol": "1e-7",
        "max_iter": "100",
    }
    result = None
    error = None
    columns = []
    labels = []
    serie_valor = []
    serie_error = []
    label_valor = "valor"
    residuo_label = "|f(raiz)|"

    if request.method == "POST":
        try:
            form.update({k: request.form.get(k, form.get(k, "")) for k in form})
            form["expr_f"] = normalize_expr(form["expr_f"])
            form["expr_g"] = normalize_expr(form["expr_g"])

            metodo = form["metodo"]
            tol = parse_float("tol", form["tol"])
            max_iter = parse_int("max_iter", form["max_iter"])
            if tol <= 0:
                raise ValueError("La tolerancia debe ser mayor a 0.")

            if metodo == "biseccion":
                if not form["expr_f"]:
                    error = "Para biseccion debes ingresar f(x)."
                else:
                    a = parse_float("a", form["a"])
                    b = parse_float("b", form["b"])
                    result = biseccion_python(form["expr_f"], a, b, tol, max_iter)
            else:
                if not form["expr_g"] or not form["expr_f"]:
                    error = "Para punto fijo debes ingresar f(x) y g(x)."
                else:
                    x0 = parse_float("x0", form["x0"])
                    result = punto_fijo_python(form["expr_f"], form["expr_g"], x0, tol, max_iter)

            if result is not None:
                labels = [row["iter"] for row in result["historial"]]
                serie_error = [row["error"] for row in result["historial"]]
                if metodo == "biseccion":
                    columns = ["iter", "a", "b", "c", "fc", "error"]
                    serie_valor = [row["c"] for row in result["historial"]]
                    label_valor = "c"
                else:
                    columns = ["iter", "x_n", "x_next", "error", "residual"]
                    serie_valor = [row["x_next"] for row in result["historial"]]
                    label_valor = "x_n+1"
                    residuo_label = "|f(raiz)|"
        except Exception as e:
            error = str(e)

    return render_template_string(
        TEMPLATE,
        form=form,
        result=result,
        error=error,
        columns=columns,
        labels=labels,
        serie_valor=serie_valor,
        serie_error=serie_error,
        label_valor=label_valor,
        residuo_label=residuo_label,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)
