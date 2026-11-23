#Інструкція:
#Код запускається задопомогою команди: python lab3.py
#Зупиняжться за допомогою комбінації клавіш: `Ctrl+C` .

import http.server
import socketserver
import json
import math
import random
import webbrowser
import threading

PORT = 8003  

# 1. МАТЕМАТИЧНЕ ЯДРО 

def eval_func(expr_str, x):
    """Безпечне обчислення математичної функції f(x) для довільного рядка."""
    allowed = {
        "x": x,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "tg": math.tan,
        "atan": math.atan,
        "arctg": math.atan,
        "acos": math.acos,
        "arccos": math.acos,
        "asin": math.asin,
        "arcsin": math.asin,
        "exp": math.exp,
        "ln": math.log,
        "log": math.log,
        "lg": math.log10,
        "sqrt": math.sqrt,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }
    return float(eval(expr_str, {"__builtins__": {}}, allowed))


# Завдання №1: Вузли Чебишова та інтерполяційний многочлен Лагранжа
def get_chebyshev_nodes(alpha, beta, n):
    nodes = []
    for i in range(n + 1):
        xi = 0.5 * (beta + alpha) + 0.5 * (beta - alpha) * math.cos(((2 * i + 1) * math.pi) / (2 * (n + 1)))
        nodes.append(xi)
    return nodes


def lagrange_eval(x_val, x_nodes, y_nodes):
    n = len(x_nodes) - 1
    total = 0.0
    for k in range(n + 1):
        phi_k = 1.0
        for i in range(n + 1):
            if i != k:
                phi_k *= (x_val - x_nodes[i]) / (x_nodes[k] - x_nodes[i])
        total += y_nodes[k] * phi_k
    return total


# Завдання №2: Розділені різниці та многочлен Ньютона для нерівновіддалених вузлів
def divided_differences_table(x_nodes, y_nodes):
    n = len(x_nodes)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y_nodes[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])
    return table


def newton_divided_eval(x_val, x_nodes, div_table):
    n = len(x_nodes) - 1
    result = div_table[0][0]
    prod = 1.0
    for k in range(1, n + 1):
        prod *= (x_val - x_nodes[k - 1])
        result += div_table[0][k] * prod
    return result


# Завдання №3: Скінченні різниці та многочлени Ньютона вперед і назад
def finite_differences_table(y_nodes):
    n = len(y_nodes)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y_nodes[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table


def newton_forward_eval(x_val, x_nodes, fin_table):
    n = len(x_nodes) - 1
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[0]) / h
    result = fin_table[0][0]
    t_prod = 1.0
    fact = 1.0
    for k in range(1, n + 1):
        t_prod *= (t - (k - 1))
        fact *= k
        result += (fin_table[0][k] / fact) * t_prod
    return result


def newton_backward_eval(x_val, x_nodes, fin_table):
    n = len(x_nodes) - 1
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[n]) / h
    result = fin_table[n][0]
    t_prod = 1.0
    fact = 1.0
    for k in range(1, n + 1):
        t_prod *= (t + (k - 1))
        fact *= k
        result += (fin_table[n - k][k] / fact) * t_prod
    return result


def process_interpolation(data):
    expr = data["expr"]
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    n = int(data["n"])
    m = int(data["m"])
    seed = int(data.get("seed", 16))

    if beta <= alpha:
        return {"error": "Права межа β має бути строго більшою за ліву межу α."}
    if n < 2 or n > 25:
        return {"error": "Степінь полінома n має бути в межах від 2 до 25."}
    if m < 100 or m > 50000:
        return {"error": "Кількість контрольних точок m має бути від 100 до 50000."}

    try:
        _ = eval_func(expr, 0.5 * (alpha + beta))
    except Exception as e:
        return {"error": f"Помилка у записі функції f(x): {e}"}

    # ЗАВДАННЯ 1: Лагранж у вузлах Чебишова 
    x_cheb = get_chebyshev_nodes(alpha, beta, n)
    y_cheb = [eval_func(expr, xi) for xi in x_cheb]

    # Оцінка eps_n на густій сітці m точок
    eps_n = 0.0
    for j in range(m + 1):
        xj = alpha + j * (beta - alpha) / m
        err = abs(eval_func(expr, xj) - lagrange_eval(xj, x_cheb, y_cheb))
        if err > eps_n:
            eps_n = err

    # ЗАВДАННЯ 2: Ньютон на випадкових вузлах 
    rng = random.Random(seed)
    x_rand = sorted([rng.uniform(alpha, beta) for _ in range(n + 1)])
    y_rand = [eval_func(expr, xi) for xi in x_rand]
    div_table = divided_differences_table(x_rand, y_rand)

    eps_m_rand = 0.0
    for _ in range(m):
        xj = rng.uniform(alpha, beta)
        err = abs(eval_func(expr, xj) - newton_divided_eval(xj, x_rand, div_table))
        if err > eps_m_rand:
            eps_m_rand = err

    #ЗАВДАННЯ 3: Ньютон вперед та назад на рівномірній сітці
    h = (beta - alpha) / n
    x_unif = [alpha + i * h for i in range(n + 1)]
    y_unif = [eval_func(expr, xi) for xi in x_unif]
    fin_table = finite_differences_table(y_unif)

    h_m = (beta - alpha) / m
    eps_m_fwd = 0.0
    eps_m_bwd = 0.0
    for j in range(m + 1):
        xj = alpha + j * h_m
        f_val = eval_func(expr, xj)
        err_f = abs(f_val - newton_forward_eval(xj, x_unif, fin_table))
        err_b = abs(f_val - newton_backward_eval(xj, x_unif, fin_table))
        if err_f > eps_m_fwd:
            eps_m_fwd = err_f
        if err_b > eps_m_bwd:
            eps_m_bwd = err_b

    #Дані для побудови графіків (250 точок на [alpha, beta])
    plot_points = 250
    plot_data = []
    for k in range(plot_points + 1):
        xk = alpha + k * (beta - alpha) / plot_points
        fk = eval_func(expr, xk)
        lk = lagrange_eval(xk, x_cheb, y_cheb)
        nk = newton_divided_eval(xk, x_rand, div_table)
        nfk = newton_forward_eval(xk, x_unif, fin_table)
        nbk = newton_backward_eval(xk, x_unif, fin_table)
        plot_data.append({
            "x": xk,
            "f": fk,
            "L": lk,
            "N_rand": nk,
            "N_fwd": nfk,
            "N_bwd": nbk,
            "err_L": abs(fk - lk),
            "err_N_rand": abs(fk - nk),
            "err_N_fwd": abs(fk - nfk),
            "err_N_bwd": abs(fk - nbk)
        })

    return {
        "eps_n": eps_n,
        "eps_m_rand": eps_m_rand,
        "eps_m_fwd": eps_m_fwd,
        "eps_m_bwd": eps_m_bwd,
        "cheb_nodes": [{"i": i, "x": x_cheb[i], "y": y_cheb[i]} for i in range(n + 1)],
        "rand_nodes": [{"i": i, "x": x_rand[i], "y": y_rand[i]} for i in range(n + 1)],
        "unif_nodes": [{"i": i, "x": x_unif[i], "y": y_unif[i]} for i in range(n + 1)],
        "div_table": div_table,
        "fin_table": fin_table,
        "plot_data": plot_data
    }


# 2. HTML/CSS/JS ІНТЕРФЕЙС ОНЛАЙН-КАЛЬКУЛЯТОРА 

HTML_PAGE = """<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Лабораторна робота №3 — Інтерполяція функцій (Варіант 16)</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f6f9; color: #222; margin: 0; padding: 20px; }
  .container { max-width: 1180px; margin: auto; background: #fff; padding: 25px 35px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
  h1 { margin-top: 0; color: #1a365d; font-size: 24px; }
  h2 { color: #2b6cb0; font-size: 18px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; }
  .controls { display: flex; flex-wrap: wrap; gap: 18px; background: #edf2f7; padding: 15px; border-radius: 8px; align-items: flex-end; }
  .control-group { display: flex; flex-direction: column; }
  label { font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #4a5568; }
  input[type="number"], input[type="text"] { padding: 7px 10px; border: 1px solid #cbd5e0; border-radius: 5px; font-size: 14px; }
  input.func-input { width: 260px; font-family: Consolas, monospace; }
  input.num-input { width: 85px; }
  button { padding: 9px 18px; border: none; border-radius: 5px; font-size: 14px; font-weight: 600; cursor: pointer; transition: 0.2s; }
  .btn-primary { background: #2b6cb0; color: #fff; }
  .btn-primary:hover { background: #2c5282; }
  .btn-secondary { background: #718096; color: #fff; }
  .btn-secondary:hover { background: #4a5568; }
  table.res-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
  table.res-table th, table.res-table td { border: 1px solid #cbd5e0; padding: 8px 10px; text-align: left; }
  table.res-table th { background: #edf2f7; color: #2d3748; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }
  @media (max-width: 950px) { .charts-grid { grid-template-columns: 1fr; } }
  .chart-box { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; background: #fafafa; }
  .chart-title { font-weight: 600; font-size: 14px; color: #2d3748; margin-bottom: 8px; text-align: center; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; font-size: 12px; margin-top: 8px; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .color-dot { width: 12px; height: 4px; display: inline-block; }
  .tabs { display: flex; gap: 10px; margin-top: 15px; }
  .tab-btn { background: #e2e8f0; color: #2d3748; padding: 7px 14px; border-radius: 5px; font-size: 13px; }
  .tab-btn.active { background: #2b6cb0; color: #fff; }
  .table-scroll { max-height: 340px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 10px; }
  .error-box { background: #fed7d7; color: #9b2c2c; padding: 12px; border-radius: 6px; margin-top: 15px; font-weight: 600; }
  .summary-note { background: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 12px 15px; margin-top: 15px; font-size: 14px; border-radius: 0 6px 6px 0; }
</style>
</head>
<body>
<div class="container">
  <h1>Інтерполювання функцій алгебраїчними поліномами Лагранжа та Ньютона</h1>
  <p>Лабораторна робота №3 (Варіант 16): <b>f(x) = (1/3) · ln(e<sup>3x</sup> / (1 + e<sup>3x</sup>))</b> на відрізку <b>[2, 3]</b>.</p>

  <div class="controls">
    <div class="control-group">
      <label>Функція f(x):</label>
      <input type="text" id="expr" class="func-input" value="(1/3) * ln(exp(3*x) / (1 + exp(3*x)))">
    </div>
    <div class="control-group">
      <label>Ліва межа α:</label>
      <input type="number" id="alpha" class="num-input" step="any" value="2">
    </div>
    <div class="control-group">
      <label>Права межа β:</label>
      <input type="number" id="beta" class="num-input" step="any" value="3">
    </div>
    <div class="control-group">
      <label>Степінь n:</label>
      <input type="number" id="n" class="num-input" min="2" max="25" value="10">
    </div>
    <div class="control-group">
      <label>Контр. точок m:</label>
      <input type="number" id="m" class="num-input" min="100" max="50000" value="10000">
    </div>
    <div class="control-group">
      <label>Seed (випадкові):</label>
      <input type="number" id="seed" class="num-input" value="16">
    </div>
    <button class="btn-primary" onclick="calculate()">Побудувати поліноми</button>
    <button class="btn-secondary" onclick="resetVariant16()">Варіант 16</button>
  </div>

  <div id="errorContainer"></div>

  <div id="resultsSection" style="display:none;">
    <h2>1. Порівняння похибок інтерполяції (Завдання №1–3)</h2>
    <table class="res-table">
      <thead>
        <tr>
          <th>Завдання</th>
          <th>Метод та вибір вузлів</th>
          <th>Позначення</th>
          <th>Максимальна похибка на [α, β]</th>
        </tr>
      </thead>
      <tbody id="errorsTableBody"></tbody>
    </table>
    <div class="summary-note" id="conclusionText"></div>

    <h2>2. Графіки функції f(x), інтерполяційних многочленів та похибок</h2>
    <div class="charts-grid">
      <div class="chart-box">
        <div class="chart-title">Функція f(x) та інтерполяційні многочлени</div>
        <canvas id="funcChart" width="520" height="310"></canvas>
        <div class="legend">
          <div class="legend-item"><span class="color-dot" style="background:#111;height:3px;"></span>f(x)</div>
          <div class="legend-item"><span class="color-dot" style="background:#2b6cb0;"></span>L<sub>n</sub>(x) (Чебишов)</div>
          <div class="legend-item"><span class="color-dot" style="background:#dd6b20;"></span>N<sub>n</sub>(x) (Випадкові)</div>
          <div class="legend-item"><span class="color-dot" style="background:#2f855a;"></span>N<sub>n</sub><sup>(В)</sup>(x) / N<sub>n</sub><sup>(Н)</sup>(x)</div>
        </div>
      </div>
      <div class="chart-box">
        <div class="chart-title">Розподіл абсолютних похибок |f(x) - P<sub>n</sub>(x)| (лог. шкала)</div>
        <canvas id="errChart" width="520" height="310"></canvas>
        <div class="legend">
          <div class="legend-item"><span class="color-dot" style="background:#2b6cb0;"></span>|f - L<sub>n</sub>| (Чебишов)</div>
          <div class="legend-item"><span class="color-dot" style="background:#dd6b20;"></span>|f - N<sub>n</sub>| (Випадкові)</div>
          <div class="legend-item"><span class="color-dot" style="background:#2f855a;"></span>|f - N<sub>n</sub><sup>(В)</sup>| (Вперед)</div>
          <div class="legend-item"><span class="color-dot" style="background:#c53030;"></span>|f - N<sub>n</sub><sup>(Н)</sup>| (Назад)</div>
        </div>
      </div>
    </div>

    <h2>3. Вузли інтерполяції та таблиці різниць</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="showTab('tabCheb', this)">Завдання 1: Вузли Чебишова</button>
      <button class="tab-btn" onclick="showTab('tabDiv', this)">Завдання 2: Таблиця розділених різниць</button>
      <button class="tab-btn" onclick="showTab('tabFin', this)">Завдання 3: Таблиця скінченних різниць</button>
    </div>

    <div id="tabCheb" class="table-scroll"></div>
    <div id="tabDiv" class="table-scroll" style="display:none;"></div>
    <div id="tabFin" class="table-scroll" style="display:none;"></div>
  </div>
</div>

<script>
function resetVariant16() {
  document.getElementById('expr').value = '(1/3) * ln(exp(3*x) / (1 + exp(3*x)))';
  document.getElementById('alpha').value = 2;
  document.getElementById('beta').value = 3;
  document.getElementById('n').value = 10;
  document.getElementById('m').value = 10000;
  document.getElementById('seed').value = 16;
  calculate();
}

function showTab(tabId, btn) {
  ['tabCheb', 'tabDiv', 'tabFin'].forEach(id => {
    document.getElementById(id).style.display = (id === tabId) ? 'block' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

async function calculate() {
  const payload = {
    expr: document.getElementById('expr').value,
    alpha: parseFloat(document.getElementById('alpha').value),
    beta: parseFloat(document.getElementById('beta').value),
    n: parseInt(document.getElementById('n').value),
    m: parseInt(document.getElementById('m').value),
    seed: parseInt(document.getElementById('seed').value) || 16
  };

  const resp = await fetch('/solve', {
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

  // 1. Таблиця порівняння похибок
  document.getElementById('errorsTableBody').innerHTML = `
    <tr style="background:#ebf8ff;">
      <td><b>Завдання №1</b></td>
      <td>Многочлен Лагранжа (корені многочлена Чебишова)</td>
      <td><b>ε<sub>n</sub></b></td>
      <td><b>${data.eps_n.toExponential(6)}</b></td>
    </tr>
    <tr>
      <td><b>Завдання №2</b></td>
      <td>Многочлен Ньютона (випадкові вузли, розділені різниці)</td>
      <td><b>ε<sub>m</sub></b></td>
      <td><b>${data.eps_m_rand.toExponential(6)}</b></td>
    </tr>
    <tr>
      <td><b>Завдання №3 (а)</b></td>
      <td>Многочлен Ньютона вперед (рівномірна сітка, скінченні різниці)</td>
      <td><b>ε<sub>m</sub><sup>(В)</sup></b></td>
      <td><b>${data.eps_m_fwd.toExponential(6)}</b></td>
    </tr>
    <tr>
      <td><b>Завдання №3 (б)</b></td>
      <td>Многочлен Ньютона назад (рівномірна сітка, скінченні різниці)</td>
      <td><b>ε<sub>m</sub><sup>(Н)</sup></b></td>
      <td><b>${data.eps_m_bwd.toExponential(6)}</b></td>
    </tr>
  `;

  document.getElementById('conclusionText').innerHTML = `
    <b>Висновок:</b> Найвищу точність забезпечує інтерполяція Лагранжа у вузлах Чебишова (<b>ε<sub>n</sub> = ${data.eps_n.toExponential(2)}</b>), 
    оскільки ці вузли мінімізують залишковий член на відрізку. Похибки формул Ньютона вперед та назад на рівномірній сітці збігаються 
    (<b>ε<sub>m</sub><sup>(В)</sup> = ε<sub>m</sub><sup>(Н)</sup> = ${data.eps_m_fwd.toExponential(2)}</b>) внаслідок єдиності інтерполяційного многочлена степеня n.
  `;

  // 2. Вкладка 1: Вузли Чебишова
  let chebHtml = '<table class="res-table" style="margin-top:0;"><thead><tr><th>i</th><th>Вузол Чебишова x<sub>i</sub></th><th>Значення f(x<sub>i</sub>)</th></tr></thead><tbody>';
  for (const row of data.cheb_nodes) {
    chebHtml += `<tr><td>${row.i}</td><td>${row.x.toFixed(8)}</td><td>${row.y.toExponential(8)}</td></tr>`;
  }
  chebHtml += '</tbody></table>';
  document.getElementById('tabCheb').innerHTML = chebHtml;

  // 3. Вкладка 2: Таблиця розділених різниць
  const n = data.rand_nodes.length - 1;
  let divHtml = '<table class="res-table" style="margin-top:0;"><thead><tr><th>i</th><th>x<sub>i</sub></th><th>f(x<sub>i</sub>)</th>';
  for (let j = 1; j <= n; j++) divHtml += `<th>Порядок ${j}</th>`;
  divHtml += '</tr></thead><tbody>';
  for (let i = 0; i <= n; i++) {
    divHtml += `<tr><td>${i}</td><td>${data.rand_nodes[i].x.toFixed(5)}</td>`;
    for (let j = 0; j <= n; j++) {
      if (j <= n - i) divHtml += `<td>${data.div_table[i][j].toExponential(4)}</td>`;
      else divHtml += '<td></td>';
    }
    divHtml += '</tr>';
  }
  divHtml += '</tbody></table>';
  document.getElementById('tabDiv').innerHTML = divHtml;

  // 4. Вкладка 3: Таблиця скінченних різниць
  let finHtml = '<table class="res-table" style="margin-top:0;"><thead><tr><th>i</th><th>x<sub>i</sub></th><th>y<sub>i</sub></th>';
  for (let j = 1; j <= n; j++) finHtml += `<th>Δ<sup>${j}</sup>y<sub>i</sub></th>`;
  finHtml += '</tr></thead><tbody>';
  for (let i = 0; i <= n; i++) {
    finHtml += `<tr><td>${i}</td><td>${data.unif_nodes[i].x.toFixed(4)}</td>`;
    for (let j = 0; j <= n; j++) {
      if (j <= n - i) finHtml += `<td>${data.fin_table[i][j].toExponential(4)}</td>`;
      else finHtml += '<td></td>';
    }
    finHtml += '</tr>';
  }
  finHtml += '</tbody></table>';
  document.getElementById('tabFin').innerHTML = finHtml;

  // 5. Побудова обох графіків
  drawFuncChart(data);
  drawErrChart(data);
}

function drawFuncChart(data) {
  const canvas = document.getElementById('funcChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pts = data.plot_data;
  const padL = 65, padR = 20, padT = 20, padB = 35;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;

  const minX = pts[0].x, maxX = pts[pts.length - 1].x;
  let allY = [];
  pts.forEach(p => { allY.push(p.f, p.L, p.N_rand, p.N_fwd); });
  let minY = Math.min(...allY), maxY = Math.max(...allY);
  if (Math.abs(maxY - minY) < 1e-12) { minY -= 1; maxY += 1; }

  const toX = x => padL + ((x - minX) / (maxX - minX)) * W;
  const toY = y => padT + H - ((y - minY) / (maxY - minY)) * H;

  // Сітка та осі
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const gy = padT + (g / 4) * H;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(padL + W, gy); ctx.stroke();
    const valY = maxY - (g / 4) * (maxY - minY);
    ctx.fillStyle = '#4a5568'; ctx.font = '11px sans-serif';
    ctx.fillText(valY.toExponential(2), 4, gy + 4);
  }
  for (let g = 0; g <= 5; g++) {
    const gx = padL + (g / 5) * W;
    ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, padT + H); ctx.stroke();
    const valX = minX + (g / 5) * (maxX - minX);
    ctx.fillStyle = '#4a5568'; ctx.font = '11px sans-serif';
    ctx.fillText(valX.toFixed(2), gx - 10, padT + H + 18);
  }

  function plotCurve(key, color, width, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const px = toX(p.x), py = toY(p[key]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  plotCurve('f', '#111111', 2.5);
  plotCurve('L', '#2b6cb0', 1.8, [6, 4]);
  plotCurve('N_rand', '#dd6b20', 1.8, [4, 3]);
  plotCurve('N_fwd', '#2f855a', 1.8, [2, 3]);

  // Вузли Чебишова
  ctx.fillStyle = '#2b6cb0';
  data.cheb_nodes.forEach(pt => {
    ctx.beginPath();
    ctx.arc(toX(pt.x), toY(pt.y), 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawErrChart(data) {
  const canvas = document.getElementById('errChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pts = data.plot_data;
  const padL = 65, padR = 20, padT = 20, padB = 35;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;

  const minX = pts[0].x, maxX = pts[pts.length - 1].x;
  const floorVal = 1e-17;

  let allLogY = [];
  pts.forEach(p => {
    allLogY.push(
      Math.log10(Math.max(p.err_L, floorVal)),
      Math.log10(Math.max(p.err_N_rand, floorVal)),
      Math.log10(Math.max(p.err_N_fwd, floorVal)),
      Math.log10(Math.max(p.err_N_bwd, floorVal))
    );
  });
  const minLog = Math.floor(Math.min(...allLogY));
  const maxLog = Math.ceil(Math.max(...allLogY));

  const toX = x => padL + ((x - minX) / (maxX - minX)) * W;
  const toY = val => {
    const lv = Math.log10(Math.max(val, floorVal));
    return padT + H - ((lv - minLog) / (maxLog - minLog || 1)) * H;
  };

  // Сітка
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let lg = minLog; lg <= maxLog; lg += 2) {
    const gy = padT + H - ((lg - minLog) / (maxLog - minLog || 1)) * H;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(padL + W, gy); ctx.stroke();
    ctx.fillStyle = '#4a5568'; ctx.font = '11px sans-serif';
    ctx.fillText(`1e${lg}`, 8, gy + 4);
  }
  for (let g = 0; g <= 5; g++) {
    const gx = padL + (g / 5) * W;
    ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, padT + H); ctx.stroke();
    const valX = minX + (g / 5) * (maxX - minX);
    ctx.fillStyle = '#4a5568'; ctx.font = '11px sans-serif';
    ctx.fillText(valX.toFixed(2), gx - 10, padT + H + 18);
  }

  function plotErr(key, color, width, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const px = toX(p.x), py = toY(p[key]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  plotErr('err_L', '#2b6cb0', 2.0);
  plotErr('err_N_rand', '#dd6b20', 1.8);
  plotErr('err_N_fwd', '#2f855a', 2.0);
  plotErr('err_N_bwd', '#c53030', 1.6, [5, 4]);
}

window.onload = () => resetVariant16();
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
        if self.path == "/solve":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            result = process_interpolation(data)

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"Сервер Лабораторної роботи №3 запущено: {url}")
    print("Для зупинки натисніть Ctrl+C у терміналі.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with socketserver.TCPServer(("127.0.0.1", PORT), RequestHandler) as httpd:
        httpd.serve_forever()