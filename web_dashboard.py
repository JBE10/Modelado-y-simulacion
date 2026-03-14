import json
import os
import subprocess
from pathlib import Path

from flask import Flask, render_template_string, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
API_BIN = BASE_DIR / "algoritmos_api"

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
      <p class="subtitle">Dashboard profesional en localhost. Calculo numerico ejecutado en C.</p>
    </div>
    <div class="pill">C Core + Flask UI</div>
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


def normalize_expr(expr):
    return (expr or "").strip().replace("**", "^")


def run_api(command):
    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, check=False)
    raw = completed.stdout.strip() or completed.stderr.strip()
    if not raw:
        return {"ok": False, "error": "No hubo salida del binario C"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Salida JSON invalida del binario C"}
    if completed.returncode != 0 and data.get("ok", False):
        return {"ok": False, "error": "El binario C finalizo con error"}
    return data


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
        form.update({k: request.form.get(k, form.get(k, "")) for k in form})
        form["expr_f"] = normalize_expr(form["expr_f"])
        form["expr_g"] = normalize_expr(form["expr_g"])

        if not API_BIN.exists():
            error = "No existe el binario algoritmos_api. Ejecuta: make api"
        else:
            metodo = form["metodo"]
            if metodo == "biseccion":
                if not form["expr_f"]:
                    error = "Para biseccion debes ingresar f(x)."
                else:
                    cmd = [
                        str(API_BIN),
                        "biseccion",
                        form["expr_f"],
                        form["a"],
                        form["b"],
                        form["tol"],
                        form["max_iter"],
                    ]
            else:
                if not form["expr_g"] or not form["expr_f"]:
                    error = "Para punto fijo debes ingresar f(x) y g(x)."
                else:
                    cmd = [
                        str(API_BIN),
                        "punto_fijo",
                        form["expr_g"],
                        form["x0"],
                        form["tol"],
                        form["max_iter"],
                        form["expr_f"],
                    ]

            if error is None:
                data = run_api(cmd)
                if not data.get("ok"):
                    error = data.get("error", "Error desconocido")
                else:
                    result = data
                    labels = [row["iter"] for row in result["historial"]]
                    serie_error = [row["error"] for row in result["historial"]]
                    if metodo == "biseccion":
                        columns = ["iter", "a", "b", "c", "fc", "error"]
                        serie_valor = [row["c"] for row in result["historial"]]
                        label_valor = "c"
                        residuo_label = "|f(raiz)|"
                    else:
                        columns = ["iter", "x_n", "x_next", "error", "residual"]
                        serie_valor = [row["x_next"] for row in result["historial"]]
                        label_valor = "x_n+1"
                        residuo_label = "|f(raiz)|"

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
