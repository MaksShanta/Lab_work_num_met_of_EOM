import http.server
import socketserver
import json
import math
import webbrowser
import threading

PORT = 8004

# 1. МАТЕМАТИЧНЕ ЯДРО

SAFE_MATH_ENV = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tg": math.tan,
    "atan": math.atan,
    "arctg": math.atan,
    "asin": math.asin,
    "acos": math.acos,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "lg": math.log10,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def make_function(expr_str):
    """Перетворює текстовий вираз у функцію f(x)."""
    cleaned = expr_str.strip().replace("^", "**")
    code = compile(cleaned, "<string>", "eval")

    def func(x):
        local_env = dict(SAFE_MATH_ENV)
        local_env["x"] = x
        return float(eval(code, {"__builtins__": {}}, local_env))

    return func


# КВАДРАТУРНІ ФОРМУЛИ

def left_rectangles(f, a, b, n):
    """Формула лівих прямокутників (порядок точності m = 1)."""
    h = (b - a) / n
    return h * sum(f(a + (i - 1) * h) for i in range(1, n + 1))


def right_rectangles(f, a, b, n):
    """Формула правих прямокутників (порядок точності m = 1)."""
    h = (b - a) / n
    return h * sum(f(a + i * h) for i in range(1, n + 1))


def mid_rectangles(f, a, b, n):
    """Формула середніх прямокутників (порядок точності m = 2)."""
    h = (b - a) / n
    return h * sum(f(a + (i - 0.5) * h) for i in range(1, n + 1))


def trapezoidal(f, a, b, n):
    """Формула трапецій (порядок точності m = 2)."""
    h = (b - a) / n
    endpoints = 0.5 * (f(a) + f(b))
    middle = sum(f(a + i * h) for i in range(1, n))
    return h * (endpoints + middle)


def simpson(f, a, b, n):
    """Формула Сімпсона / парабол (порядок точності m = 4)."""
    h = (b - a) / n
    endpoints = f(a) + f(b)
    sum_nodes = sum(f(a + i * h) for i in range(1, n))
    sum_midpoints = sum(f(a + (i - 0.5) * h) for i in range(1, n + 1))
    return (h / 6.0) * (endpoints + 2.0 * sum_nodes + 4.0 * sum_midpoints)


METHODS_CONFIG = [
    {"id": "left", "name": "Лівих прямокутників", "func": left_rectangles, "order": 1},
    {"id": "right", "name": "Правих прямокутників", "func": right_rectangles, "order": 1},
    {"id": "mid", "name": "Середніх прямокутників", "func": mid_rectangles, "order": 2},
    {"id": "trap", "name": "Трапецій", "func": trapezoidal, "order": 2},
    {"id": "simpson", "name": "Сімпсона (парабол)", "func": simpson, "order": 4},
]


def solve_with_precision(quad_func, f, a, b, eps, order, mode="runge", n_start=4, max_n=262144):
    """Пошук наближеного значення інтеграла із заданою точністю eps."""
    n = n_start
    i_prev = quad_func(f, a, b, n)
    denom = (2 ** order - 1) if mode == "runge" else 1.0

    while n <= max_n:
        n_next = n * 2
        i_curr = quad_func(f, a, b, n_next)
        est_error = abs(i_curr - i_prev) / denom

        if est_error <= eps:
            return {
                "value": i_curr,
                "n": n_next,
                "h": (b - a) / n_next,
                "est_error": est_error,
                "converged": True,
            }
        n = n_next
        i_prev = i_curr

    return {
        "value": i_prev,
        "n": n,
        "h": (b - a) / n,
        "est_error": abs(i_prev - quad_func(f, a, b, n // 2)) / denom,
        "converged": False,
    }


def process_calculation(data):
    f_str = data.get("f_str", "").strip()
    F_str = data.get("F_str", "").strip()
    a = float(data["a"])
    b = float(data["b"])
    eps = float(data["eps"])
    n_table = int(data.get("n_table", 10))

    if a >= b:
        return {"error": "Нижня межа інтегрування a має бути меншою за верхню межу b."}
    if eps <= 0:
        return {"error": "Точність ε має бути додатним числом."}
    if n_table < 2 or n_table > 200:
        return {"error": "Кількість розбиттів n для таблиці вузлів має бути від 2 до 200."}

    try:
        f = make_function(f_str)
        _ = f((a + b) / 2.0)
    except Exception as ex:
        return {"error": f"Помилка у записі підінтегральної функції f(x): {ex}"}

    i_exact = None
    if F_str:
        try:
            F = make_function(F_str)
            i_exact = F(b) - F(a)
        except Exception as ex:
            return {"error": f"Помилка у записі первісної F(x): {ex}"}

    # 1. Подвійний перерахунок при n1=8 та n2=10
    fixed_results = []
    for m in METHODS_CONFIG:
        i8 = m["func"](f, a, b, 8)
        i10 = m["func"](f, a, b, 10)
        fixed_results.append({
            "name": m["name"],
            "order": m["order"],
            "i8": i8,
            "i10": i10,
            "diff": abs(i10 - i8),
            "exact_err": abs(i10 - i_exact) if i_exact is not None else None,
        })

    # 2. Обчислення із заданою точністю eps (Подвійний перерахунок vs Рунге)
    precision_results = []
    for m in METHODS_CONFIG:
        res_dbl = solve_with_precision(m["func"], f, a, b, eps, m["order"], mode="double")
        res_rng = solve_with_precision(m["func"], f, a, b, eps, m["order"], mode="runge")
        precision_results.append({
            "name": m["name"],
            "order": m["order"],
            "dbl": {
                "n": res_dbl["n"],
                "h": res_dbl["h"],
                "value": res_dbl["value"],
                "est_error": res_dbl["est_error"],
                "exact_err": abs(res_dbl["value"] - i_exact) if i_exact is not None else None,
                "converged": res_dbl["converged"],
            },
            "rng": {
                "n": res_rng["n"],
                "h": res_rng["h"],
                "value": res_rng["value"],
                "est_error": res_rng["est_error"],
                "exact_err": abs(res_rng["value"] - i_exact) if i_exact is not None else None,
                "converged": res_rng["converged"],
            },
        })

    # 3. Таблиця вузлів для обраного n_table та точки для графіка
    h_tbl = (b - a) / n_table
    nodes_table = []
    for i in range(n_table + 1):
        xi = a + i * h_tbl
        x_mid = (xi - 0.5 * h_tbl) if i > 0 else None
        nodes_table.append({
            "i": i,
            "xi": xi,
            "f_xi": f(xi),
            "x_mid": x_mid,
            "f_xmid": f(x_mid) if x_mid is not None else None,
        })

    curve_points = []
    steps_plot = 200
    for k in range(steps_plot + 1):
        xk = a + (b - a) * k / steps_plot
        curve_points.append({"x": xk, "y": f(xk)})

    return {
        "a": a,
        "b": b,
        "eps": eps,
        "i_exact": i_exact,
        "fixed_results": fixed_results,
        "precision_results": precision_results,
        "nodes_table": nodes_table,
        "curve_points": curve_points,
        "h_tbl": h_tbl,
    }


# 2. HTML/CSS/JS ІНТЕРФЕЙС КАЛЬКУЛЯТОРА 

HTML_PAGE = """<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Лабораторна робота №4 — Чисельне інтегрування (Варіант 16)</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f6f9; color: #222; margin: 0; padding: 20px; }
  .container { max-width: 1180px; margin: auto; background: #fff; padding: 25px 35px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
  h1 { margin-top: 0; color: #1a365d; font-size: 24px; }
  h2 { color: #2b6cb0; font-size: 18px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; }
  .presets { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
  .btn-preset { background: #edf2f7; color: #2d3748; border: 1px solid #cbd5e0; padding: 7px 12px; border-radius: 5px; font-size: 13px; cursor: pointer; font-weight: 600; }
  .btn-preset:hover { background: #e2e8f0; }
  .btn-preset.active { background: #2b6cb0; color: #fff; border-color: #2b6cb0; }
  .controls { display: flex; flex-wrap: wrap; gap: 18px; background: #edf2f7; padding: 16px; border-radius: 8px; align-items: flex-end; }
  .control-group { display: flex; flex-direction: column; }
  label { font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #4a5568; }
  input[type="number"], input[type="text"], select { padding: 7px 10px; border: 1px solid #cbd5e0; border-radius: 5px; font-size: 14px; }
  button.btn-primary { padding: 9px 20px; border: none; border-radius: 5px; font-size: 14px; font-weight: 600; cursor: pointer; background: #2b6cb0; color: #fff; }
  button.btn-primary:hover { background: #2c5282; }
  .badges { display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
  .badge { background: #ebf8ff; border: 1px solid #bee3f8; padding: 10px 16px; border-radius: 6px; font-size: 14px; }
  .badge strong { color: #2b6cb0; }
  table.res-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13.5px; }
  table.res-table th, table.res-table td { border: 1px solid #cbd5e0; padding: 8px 10px; text-align: left; }
  table.res-table th { background: #edf2f7; color: #2d3748; }
  table.res-table tr:nth-child(even) { background: #f8fafc; }
  .visual-grid { display: flex; flex-wrap: wrap; gap: 25px; margin-top: 15px; }
  .chart-box { flex: 1.4; min-width: 480px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; background: #fafafa; }
  .nodes-box { flex: 1; min-width: 360px; max-height: 360px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
  .error-box { background: #fed7d7; color: #9b2c2c; padding: 12px; border-radius: 6px; margin-top: 15px; font-weight: 600; }
  code { font-family: Consolas, monospace; font-size: 13px; }
</style>
</head>
<body>
<div class="container">
  <h1>Лабораторна робота №4 — Чисельне інтегрування функцій (Варіант 16)</h1>
  <p>Квадратурні формули <b>лівих, правих, середніх прямокутників</b>, <b>трапецій</b> та <b>Сімпсона (парабол)</b>. Оцінка похибки методом <b>подвійного перерахунку</b> та методом <b>Рунге</b>.</p>

  <div class="presets">
    <button class="btn-preset active" id="p0" onclick="setPreset(0)">Варіант 16 (Основний): ∫₁² x·sin(x) dx</button>
    <button class="btn-preset" id="p1" onclick="setPreset(1)">Тест 1 (вар. 16): ∫₁.₆²·⁶ (x²+1)·sin(x) dx</button>
    <button class="btn-preset" id="p2" onclick="setPreset(2)">Тест 2 (вар. 16): ∫₁.₂¹·⁸ dx / √(x²+0.6)</button>
    <button class="btn-preset" id="p3" onclick="setPreset(3)">Тест 3 (вар. 16): ∫₁.₄² dx / √(3x²+1)</button>
  </div>

  <div class="controls">
    <div class="control-group">
      <label>Підінтегральна функція f(x):</label>
      <input type="text" id="f_str" value="x * sin(x)" style="width: 200px;">
    </div>
    <div class="control-group">
      <label>Первісна F(x) (необов'язково):</label>
      <input type="text" id="F_str" value="sin(x) - x * cos(x)" style="width: 210px;">
    </div>
    <div class="control-group">
      <label>Нижня межа a:</label>
      <input type="number" step="any" id="a" value="1" style="width: 80px;">
    </div>
    <div class="control-group">
      <label>Верхня межа b:</label>
      <input type="number" step="any" id="b" value="2" style="width: 80px;">
    </div>
    <div class="control-group">
      <label>Точність ε:</label>
      <input type="text" id="eps" value="1e-5" style="width: 90px;">
    </div>
    <div class="control-group">
      <label>n для таблиці/графіка:</label>
      <input type="number" id="n_table" value="10" min="2" max="100" style="width: 85px;">
    </div>
    <button class="btn-primary" onclick="calculate()">Обчислити</button>
  </div>

  <div id="errorContainer"></div>

  <div id="resultsSection" style="display:none;">
    <div class="badges" id="summaryBadges"></div>

    <h2>1. Подвійний перерахунок при фіксованих n₁ = 8 та n₂ = 10</h2>
    <table class="res-table">
      <thead>
        <tr>
          <th>Квадратурна формула</th>
          <th>Порядок</th>
          <th>I₈ (n₁ = 8)</th>
          <th>I₁₀ (n₂ = 10)</th>
          <th>|I₁₀ − I₈|</th>
          <th>Точна похибка |I₁₀ − I*|</th>
        </tr>
      </thead>
      <tbody id="fixedBody"></tbody>
    </table>

    <h2>2. Досягнення заданої точності ε (Подвійний перерахунок та Метод Рунге)</h2>
    <table class="res-table">
      <thead>
        <tr>
          <th rowspan="2">Квадратурна формула</th>
          <th colspan="4" style="text-align:center; background:#e2e8f0;">Подвійний перерахунок (|I₂ₙ − Iₙ| ≤ ε)</th>
          <th colspan="4" style="text-align:center; background:#dbeafe;">Метод Рунге (|I₂ₙ − Iₙ| / (2ᵐ − 1) ≤ ε)</th>
        </tr>
        <tr>
          <th>n</th>
          <th>Значення I</th>
          <th>Оцінка похибки</th>
          <th>Точна похибка</th>
          <th>n</th>
          <th>Значення I</th>
          <th>Оцінка Рунге</th>
          <th>Точна похибка</th>
        </tr>
      </thead>
      <tbody id="precisionBody"></tbody>
    </table>

    <h2>3. Геометрична ілюстрація та таблиця значень у вузлах (при заданому n)</h2>
    <div style="margin-bottom: 10px;">
      <label>Відобразити на графіку метод: </label>
      <select id="visMethod" onchange="redrawPlot()">
        <option value="mid">Середні прямокутники</option>
        <option value="left">Ліві прямокутники</option>
        <option value="right">Праві прямокутники</option>
        <option value="trap">Трапеції</option>
      </select>
    </div>
    <div class="visual-grid">
      <div class="chart-box">
        <canvas id="integralCanvas" width="600" height="330"></canvas>
      </div>
      <div class="nodes-box">
        <table class="res-table" style="margin-top:0;">
          <thead>
            <tr>
              <th>i</th>
              <th>xᵢ</th>
              <th>f(xᵢ)</th>
              <th>xᵢ₋½</th>
              <th>f(xᵢ₋½)</th>
            </tr>
          </thead>
          <tbody id="nodesBody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const PRESETS = [
  { f: "x * sin(x)", F: "sin(x) - x * cos(x)", a: 1, b: 2 },
  { f: "(x**2 + 1) * sin(x)", F: "(1 - x**2) * cos(x) + 2 * x * sin(x)", a: 1.6, b: 2.6 },
  { f: "1 / sqrt(x**2 + 0.6)", F: "ln(x + sqrt(x**2 + 0.6))", a: 1.2, b: 1.8 },
  { f: "1 / sqrt(3 * x**2 + 1)", F: "(1 / sqrt(3)) * ln(sqrt(3)*x + sqrt(3 * x**2 + 1))", a: 1.4, b: 2.0 }
];

let lastData = null;

function setPreset(idx) {
  for (let i = 0; i < 4; i++) {
    document.getElementById('p' + i).classList.toggle('active', i === idx);
  }
  const p = PRESETS[idx];
  document.getElementById('f_str').value = p.f;
  document.getElementById('F_str').value = p.F;
  document.getElementById('a').value = p.a;
  document.getElementById('b').value = p.b;
  calculate();
}

async function calculate() {
  const payload = {
    f_str: document.getElementById('f_str').value,
    F_str: document.getElementById('F_str').value,
    a: parseFloat(document.getElementById('a').value),
    b: parseFloat(document.getElementById('b').value),
    eps: parseFloat(document.getElementById('eps').value),
    n_table: parseInt(document.getElementById('n_table').value)
  };

  const resp = await fetch('/integrate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await resp.json();

  const errBox = document.getElementById('errorContainer');
  const resSec = document.getElementById('resultsSection');

  if (data.error) {
    errBox.innerHTML = `<div class="error-box">${data.error}</div>`;
    resSec.style.display = 'none';
    return;
  }
  errBox.innerHTML = '';
  resSec.style.display = 'block';
  lastData = data;

  const exactStr = data.i_exact !== null ? data.i_exact.toFixed(10) : 'не задана первісна F(x)';
  document.getElementById('summaryBadges').innerHTML = `
    <div class="badge">Проміжок інтегрування: <strong>[${data.a}, ${data.b}]</strong></div>
    <div class="badge">Задана точність ε: <strong>${data.eps}</strong></div>
    <div class="badge">Точне значення (Ньютон–Лейбніц): <strong>${exactStr}</strong></div>
  `;

  // Таблиця 1: n1=8, n2=10
  let fixedHtml = '';
  for (const r of data.fixed_results) {
    const exErr = r.exact_err !== null ? r.exact_err.toExponential(2) : '—';
    fixedHtml += `<tr>
      <td><b>${r.name}</b></td>
      <td>O(h<sup>${r.order}</sup>)</td>
      <td><code>${r.i8.toFixed(8)}</code></td>
      <td><code>${r.i10.toFixed(8)}</code></td>
      <td><code>${r.diff.toExponential(2)}</code></td>
      <td><code>${exErr}</code></td>
    </tr>`;
  }
  document.getElementById('fixedBody').innerHTML = fixedHtml;

  // Таблиця 2: Точність eps
  let precHtml = '';
  for (const r of data.precision_results) {
    const dEx = r.dbl.exact_err !== null ? r.dbl.exact_err.toExponential(2) : '—';
    const rEx = r.rng.exact_err !== null ? r.rng.exact_err.toExponential(2) : '—';
    precHtml += `<tr>
      <td><b>${r.name}</b> (m=${r.order})</td>
      <td><b>${r.dbl.n}</b></td>
      <td><code>${r.dbl.value.toFixed(8)}</code></td>
      <td><code>${r.dbl.est_error.toExponential(2)}</code></td>
      <td><code>${dEx}</code></td>
      <td style="background:#eff6ff;"><b>${r.rng.n}</b></td>
      <td style="background:#eff6ff;"><code>${r.rng.value.toFixed(8)}</code></td>
      <td style="background:#eff6ff;"><code>${r.rng.est_error.toExponential(2)}</code></td>
      <td style="background:#eff6ff;"><code>${rEx}</code></td>
    </tr>`;
  }
  document.getElementById('precisionBody').innerHTML = precHtml;

  // Таблиця 3: Вузли
  let nodesHtml = '';
  for (const row of data.nodes_table) {
    nodesHtml += `<tr>
      <td>${row.i}</td>
      <td>${row.xi.toFixed(4)}</td>
      <td>${row.f_xi.toFixed(6)}</td>
      <td>${row.x_mid !== null ? row.x_mid.toFixed(4) : '—'}</td>
      <td>${row.f_xmid !== null ? row.f_xmid.toFixed(6) : '—'}</td>
    </tr>`;
  }
  document.getElementById('nodesBody').innerHTML = nodesHtml;

  redrawPlot();
}

function redrawPlot() {
  if (!lastData) return;
  const canvas = document.getElementById('integralCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const method = document.getElementById('visMethod').value;
  const { a, b, curve_points, nodes_table } = lastData;

  const padL = 55, padR = 20, padT = 20, padB = 35;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;

  const allY = curve_points.map(p => p.y).concat([0]);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const spanY = (maxY - minY) || 1;

  const toX = x => padL + ((x - a) / (b - a)) * W;
  const toY = y => padT + H - ((y - minY) / spanY) * H;
  const yZero = toY(0);

  // Малювання фігур (прямокутники або трапеції)
  ctx.fillStyle = 'rgba(43, 108, 176, 0.18)';
  ctx.strokeStyle = 'rgba(43, 108, 176, 0.75)';
  ctx.lineWidth = 1.2;

  for (let i = 1; i < nodes_table.length; i++) {
    const prev = nodes_table[i - 1];
    const curr = nodes_table[i];
    const x0 = toX(prev.xi);
    const x1 = toX(curr.xi);

    ctx.beginPath();
    if (method === 'left') {
      const yTop = toY(prev.f_xi);
      ctx.rect(x0, Math.min(yTop, yZero), x1 - x0, Math.abs(yZero - yTop));
    } else if (method === 'right') {
      const yTop = toY(curr.f_xi);
      ctx.rect(x0, Math.min(yTop, yZero), x1 - x0, Math.abs(yZero - yTop));
    } else if (method === 'mid') {
      const yTop = toY(curr.f_xmid);
      ctx.rect(x0, Math.min(yTop, yZero), x1 - x0, Math.abs(yZero - yTop));
    } else if (method === 'trap') {
      ctx.moveTo(x0, yZero);
      ctx.lineTo(x0, toY(prev.f_xi));
      ctx.lineTo(x1, toY(curr.f_xi));
      ctx.lineTo(x1, yZero);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
  }

  // Осі координат
  ctx.strokeStyle = '#4a5568';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + H);
  ctx.moveTo(padL, yZero);
  ctx.lineTo(padL + W, yZero);
  ctx.stroke();

  // Крива функції f(x)
  ctx.strokeStyle = '#e53e3e';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  curve_points.forEach((pt, idx) => {
    const px = toX(pt.x);
    const py = toY(pt.y);
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Підписи
  ctx.fillStyle = '#2d3748';
  ctx.font = '12px sans-serif';
  ctx.fillText(maxY.toFixed(2), 8, padT + 10);
  ctx.fillText(minY.toFixed(2), 8, padT + H);
  ctx.fillText(`a = ${a}`, padL - 10, padT + H + 20);
  ctx.fillText(`b = ${b}`, padL + W - 25, padT + H + 20);
}

window.onload = () => setPreset(0);
</script>
</body>
</html>
"""


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/integrate":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            result = process_calculation(data)

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"Сервер ЛР №4 запущено: {url}")
    print("Для зупинки натисніть Ctrl+C у терміналі.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with socketserver.TCPServer(("127.0.0.1", PORT), RequestHandler) as httpd:
        httpd.serve_forever()