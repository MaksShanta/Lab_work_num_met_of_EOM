#Інструкція:
#Код запускається задопомогою команди: python lab2.py
#Зупиняжться за допомогою комбінації клавіш: `Ctrl+C` .
import http.server
import socketserver
import json
import math
import webbrowser
import threading

PORT = 8000

# 1. МАТЕМАТИЧНЕ ЯДРО 

def determinant(A):
    """Обчислення визначника матриці методом Гауса з вибором головного елемента."""
    n = len(A)
    M = [row[:] for row in A]
    det = 1.0
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(M[r][i]) > abs(M[pivot][i]):
                pivot = r
        if abs(M[pivot][i]) < 1e-14:
            return 0.0
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
            det = -det
        det *= M[i][i]
        for r in range(i + 1, n):
            factor = M[r][i] / M[i][i]
            for c in range(i, n):
                M[r][c] -= factor * M[i][c]
    return det


def check_diagonal_dominance(A):
    """Перевірка строгої діагональної переваги по рядках."""
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True


def check_symmetry(A):
    """Перевірка симетричності матриці."""
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-12:
                return False
    return True


def solve_jacobi(A, f, eps, max_iter=10000):
    """Метод Якобі."""
    n = len(f)
    for i in range(n):
        if abs(A[i][i]) < 1e-14:
            return None, 0, "Нульовий елемент на головній діагоналі"

    x = [0.0] * n
    for k in range(1, max_iter + 1):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (f[i] - s) / A[i][i]

        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if math.isnan(diff) or math.isinf(diff):
            return None, k, "Метод розбігається (переповнення)"
        if diff < eps:
            return x_new, k, "Збігся"
        x = x_new

    return x, max_iter, "Досягнуто ліміт ітерацій (можлива розбіжність)"


def solve_seidel(A, f, eps, max_iter=10000):
    """Метод Зейделя."""
    n = len(f)
    for i in range(n):
        if abs(A[i][i]) < 1e-14:
            return None, 0, "Нульовий елемент на головній діагоналі"

    x = [0.0] * n
    for k in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (f[i] - s1 - s2) / A[i][i]

        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if math.isnan(diff) or math.isinf(diff):
            return None, k, "Метод розбігається (переповнення)"
        if diff < eps:
            return x, k, "Збігся"

    return x, max_iter, "Досягнуто ліміт ітерацій (можлива розбіжність)"


def solve_sor(A, f, eps, omega, max_iter=10000):
    """Метод верхньої релаксації (SOR)."""
    n = len(f)
    for i in range(n):
        if abs(A[i][i]) < 1e-14:
            return None, 0, "Нульовий елемент на головній діагоналі"

    x = [0.0] * n
    for k in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x_gs = (f[i] - s1 - s2) / A[i][i]
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs

        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if math.isnan(diff) or math.isinf(diff):
            return None, k, "Метод розбігається (переповнення)"
        if diff < eps:
            return x, k, "Збігся"

    return x, max_iter, "Досягнуто ліміт ітерацій (можлива розбіжність)"


def estimate_spectral_radius_jacobi(A, iterations=200):
    """Наближене обчислення спектрального радіуса матриці Якобі степеневим методом."""
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < 1e-14:
            return None
    # B_J = I - D^-1 * A
    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[i][j] = (1.0 if i == j else 0.0) - (A[i][j] / A[i][i])

    v = [1.0 / math.sqrt(n)] * n
    rho = 0.0
    for _ in range(iterations):
        w = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(val * val for val in w))
        if norm < 1e-15:
            return 0.0
        v = [val / norm for val in w]
        # Оцінка Релея
        Bv = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
        rho = abs(sum(v[i] * Bv[i] for i in range(n)))
    return rho


def process_calculation(data):
    A = data["A"]
    f = data["f"]
    x_exact = data["x_exact"]
    eps = float(data["eps"])
    omega = float(data["omega"])

    det_A = determinant(A)
    diag_dom = check_diagonal_dominance(A)
    symm = check_symmetry(A)

    if abs(det_A) < 1e-12:
        return {"error": "Визначник матриці дорівнює нулю (det(A) = 0). Система не має єдиного розв'язку."}

    x_jac, it_jac, st_jac = solve_jacobi(A, f, eps)
    x_sei, it_sei, st_sei = solve_seidel(A, f, eps)
    x_sor, it_sor, st_sor = solve_sor(A, f, eps, omega)

    def format_res(name, x_res, iters, status):
        err = None
        if x_res is not None and len(x_exact) == len(x_res):
            err = max(abs(x_res[i] - x_exact[i]) for i in range(len(x_res)))
        return {
            "method": name,
            "status": status,
            "iterations": iters,
            "x": x_res,
            "error": err
        }

    methods = [
        format_res("Метод Якобі", x_jac, it_jac, st_jac),
        format_res("Метод Зейделя", x_sei, it_sei, st_sei),
        format_res(f"Метод верхньої релаксації (ω = {omega:.2f})", x_sor, it_sor, st_sor),
    ]

    # Дослідження параметра omega на інтервалі (0, 2) з кроком 0.05
    sweep = []
    best_omega = None
    best_iters = None
    for step in range(1, 40):
        w = round(step * 0.05, 2)
        _, it_w, st_w = solve_sor(A, f, eps, w, max_iter=2000)
        if st_w == "Збігся":
            sweep.append({"omega": w, "iterations": it_w})
            if best_iters is None or it_w < best_iters:
                best_iters = it_w
                best_omega = w

    rho = estimate_spectral_radius_jacobi(A)
    omega_theor = None
    if symm and rho is not None and rho < 1.0:
        omega_theor = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))

    return {
        "det_A": det_A,
        "diag_dom": diag_dom,
        "symm": symm,
        "methods": methods,
        "sweep": sweep,
        "best_omega_exp": best_omega,
        "best_iters_exp": best_iters,
        "rho_jacobi": rho,
        "omega_theor": omega_theor
    }


# 2. HTML/CSS/JS ІНТЕРФЕЙС ОНЛАЙН-КАЛЬКУЛЯТОРА 

HTML_PAGE = """<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Лабораторна робота №2 — Онлайн-калькулятор СЛАР (Варіант 16)</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f6f9; color: #222; margin: 0; padding: 20px; }
  .container { max-width: 1100px; margin: auto; background: #fff; padding: 25px 35px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
  h1 { margin-top: 0; color: #1a365d; font-size: 24px; }
  h2 { color: #2b6cb0; font-size: 18px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 28px; }
  .controls { display: flex; flex-wrap: wrap; gap: 20px; background: #edf2f7; padding: 15px; border-radius: 8px; align-items: flex-end; }
  .control-group { display: flex; flex-direction: column; }
  label { font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #4a5568; }
  input[type="number"], input[type="text"] { padding: 7px 10px; border: 1px solid #cbd5e0; border-radius: 5px; font-size: 14px; width: 110px; }
  button { padding: 9px 18px; border: none; border-radius: 5px; font-size: 14px; font-weight: 600; cursor: pointer; transition: 0.2s; }
  .btn-primary { background: #2b6cb0; color: #fff; }
  .btn-primary:hover { background: #2c5282; }
  .btn-secondary { background: #718096; color: #fff; }
  .btn-secondary:hover { background: #4a5568; }
  .matrix-area { display: flex; flex-wrap: wrap; gap: 30px; margin-top: 15px; overflow-x: auto; }
  table.matrix-table { border-collapse: collapse; }
  table.matrix-table td, table.matrix-table th { padding: 4px; text-align: center; }
  table.matrix-table input { width: 68px; text-align: right; }
  .badges { display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
  .badge { background: #ebf8ff; border: 1px solid #bee3f8; padding: 10px 15px; border-radius: 6px; font-size: 14px; }
  .badge strong { color: #2b6cb0; }
  table.res-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
  table.res-table th, table.res-table td { border: 1px solid #cbd5e0; padding: 10px; text-align: left; }
  table.res-table th { background: #edf2f7; color: #2d3748; }
  .omega-layout { display: flex; flex-wrap: wrap; gap: 25px; margin-top: 15px; }
  .chart-box { flex: 2; min-width: 450px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #fafafa; }
  .sweep-table-box { flex: 1; min-width: 240px; max-height: 320px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
  .error-box { background: #fed7d7; color: #9b2c2c; padding: 12px; border-radius: 6px; margin-top: 15px; font-weight: 600; }
</style>
</head>
<body>
<div class="container">
  <h1>Онлайн-калькулятор розв'язання СЛАР ітераційними методами</h1>
  <p>Лабораторна робота №2 (Варіант 16): методи <b>Якобі</b>, <b>Зейделя</b> та <b>верхньої релаксації (SOR)</b>.</p>

  <div class="controls">
    <div class="control-group">
      <label>Порядок матриці m:</label>
      <input type="number" id="matrixSize" value="4" min="2" max="10" onchange="buildMatrixInputs()">
    </div>
    <div class="control-group">
      <label>Точність ε:</label>
      <input type="text" id="eps" value="1e-5">
    </div>
    <div class="control-group">
      <label>Параметр ω (0 &lt; ω &lt; 2):</label>
      <input type="number" id="omega" value="1.05" step="0.05" min="0.05" max="1.95">
    </div>
    <button class="btn-primary" onclick="calculate()">Розв'язати СЛАР</button>
    <button class="btn-secondary" onclick="resetVariant16()">Скинути до Варіанта 16</button>
  </div>

  <h2>1. Матриця коефіцієнтів A, вектор вільних членів f та точний розв'язок x*</h2>
  <div class="matrix-area" id="matrixContainer"></div>

  <div id="errorContainer"></div>

  <div id="resultsSection" style="display:none;">
    <h2>2. Властивості матриці та результати обчислень</h2>
    <div class="badges" id="matrixProps"></div>

    <table class="res-table">
      <thead>
        <tr>
          <th>Метод</th>
          <th>Статус</th>
          <th>Кількість ітерацій</th>
          <th>Наближений розв'язок x</th>
          <th>Похибка ||x - x*||∞</th>
        </tr>
      </thead>
      <tbody id="resultsBody"></tbody>
    </table>

    <h2>3. Дослідження вибору параметра ω у методі верхньої релаксації</h2>
    <div class="badges" id="omegaSummary"></div>
    <div class="omega-layout">
      <div class="chart-box">
        <canvas id="omegaChart" width="600" height="290"></canvas>
      </div>
      <div class="sweep-table-box">
        <table class="res-table" style="margin-top:0;">
          <thead><tr><th>Параметр ω</th><th>Кількість ітерацій</th></tr></thead>
          <tbody id="sweepBody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const V16_A = [
  [13,  1, -3, -1],
  [ 1, 20,  1, -4],
  [-3,  1, 13,  2],
  [-1, -4,  2, 20]
];
const V16_F = [10, 18, 13, 17];
const V16_EXACT = [1, 1, 1, 1];

function buildMatrixInputs(useV16 = false) {
  const m = parseInt(document.getElementById('matrixSize').value);
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  for (let j = 0; j < m; j++) html += `<th>x<sub>${j+1}</sub></th>`;
  html += '<th>&nbsp;</th><th>f<sub>i</sub></th><th>&nbsp;</th><th>x*<sub>i</sub></th></tr></thead><tbody>';

  for (let i = 0; i < m; i++) {
    html += `<tr><th>Рівн. ${i+1}</th>`;
    for (let j = 0; j < m; j++) {
      let val = (useV16 && m === 4) ? V16_A[i][j] : (i === j ? 10 : 1);
      html += `<td><input type="number" step="any" id="a_${i}_${j}" value="${val}"></td>`;
    }
    let fVal = (useV16 && m === 4) ? V16_F[i] : (10 + m - 1);
    let exVal = (useV16 && m === 4) ? V16_EXACT[i] : 1;
    html += `<td>=</td><td><input type="number" step="any" id="f_${i}" value="${fVal}"></td>`;
    html += `<td>&nbsp;</td><td><input type="number" step="any" id="ex_${i}" value="${exVal}"></td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('matrixContainer').innerHTML = html;
}

function resetVariant16() {
  document.getElementById('matrixSize').value = 4;
  document.getElementById('eps').value = '1e-5';
  document.getElementById('omega').value = 1.05;
  buildMatrixInputs(true);
  calculate();
}

async function calculate() {
  const m = parseInt(document.getElementById('matrixSize').value);
  const eps = parseFloat(document.getElementById('eps').value);
  const omega = parseFloat(document.getElementById('omega').value);

  if (isNaN(eps) || eps <= 0) { alert("Введіть коректну додатну точність ε!"); return; }
  if (isNaN(omega) || omega <= 0 || omega >= 2) { alert("Параметр ω має бути в межах (0, 2)!"); return; }

  const A = [], f = [], x_exact = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < m; j++) row.push(parseFloat(document.getElementById(`a_${i}_${j}`).value) || 0);
    A.push(row);
    f.push(parseFloat(document.getElementById(`f_${i}`).value) || 0);
    x_exact.push(parseFloat(document.getElementById(`ex_${i}`).value) || 0);
  }

  const resp = await fetch('/solve', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({A, f, x_exact, eps, omega})
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

  document.getElementById('matrixProps').innerHTML = `
    <div class="badge">Визначник det(A): <strong>${data.det_A.toFixed(2)}</strong></div>
    <div class="badge">Строга діагональна перевага: <strong>${data.diag_dom ? 'Так (збігається)' : 'Ні'}</strong></div>
    <div class="badge">Симетричність матриці: <strong>${data.symm ? 'Так' : 'Ні'}</strong></div>
  `;

  let rowsHtml = '';
  for (const item of data.methods) {
    const vecStr = item.x ? '(' + item.x.map(v => v.toFixed(6)).join(', ') + ')' : '—';
    const errStr = item.error !== null ? item.error.toExponential(2) : '—';
    rowsHtml += `<tr>
      <td><b>${item.method}</b></td>
      <td>${item.status}</td>
      <td><b>${item.iterations}</b></td>
      <td><code>${vecStr}</code></td>
      <td>${errStr}</td>
    </tr>`;
  }
  document.getElementById('resultsBody').innerHTML = rowsHtml;

  let summaryHtml = `<div class="badge">Оптимальне ω (експериментально): <strong>${data.best_omega_exp !== null ? data.best_omega_exp.toFixed(2) + ' (' + data.best_iters_exp + ' ітер.)' : '—'}</strong></div>`;
  if (data.omega_theor !== null) {
    summaryHtml += `<div class="badge">Теоретичне ω (формула Янга): <strong>${data.omega_theor.toFixed(4)}</strong> (ρ(B<sub>J</sub>) = ${data.rho_jacobi.toFixed(4)})</div>`;
  }
  document.getElementById('omegaSummary').innerHTML = summaryHtml;

  let sweepHtml = '';
  for (const s of data.sweep) {
    const highlight = (s.omega === data.best_omega_exp) ? ' style="background:#c6f6d5;font-weight:bold;"' : '';
    sweepHtml += `<tr${highlight}><td>${s.omega.toFixed(2)}</td><td>${s.iterations}</td></tr>`;
  }
  document.getElementById('sweepBody').innerHTML = sweepHtml;

  drawChart(data.sweep);
}

function drawChart(sweep) {
  const canvas = document.getElementById('omegaChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!sweep || sweep.length === 0) return;

  const padL = 50, padR = 20, padT = 25, padB = 40;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;

  const maxIt = Math.max(...sweep.map(d => d.iterations));
  const minIt = Math.min(...sweep.map(d => d.iterations));

  // Осі
  ctx.strokeStyle = '#718096';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + H);
  ctx.lineTo(padL + W, padT + H);
  ctx.stroke();

  // Підписи осей
  ctx.fillStyle = '#2d3748';
  ctx.font = '12px sans-serif';
  ctx.fillText('Ітерації', 8, padT - 8);
  ctx.fillText('Параметр ω', padL + W / 2 - 30, canvas.height - 8);
  ctx.fillText(maxIt, 12, padT + 10);
  ctx.fillText(minIt, 12, padT + H);

  // Лінія графіка
  ctx.strokeStyle = '#2b6cb0';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  sweep.forEach((pt, idx) => {
    const x = padL + ((pt.omega - 0.05) / 1.9) * W;
    const normY = (maxIt === minIt) ? 0.5 : (pt.iterations - minIt) / (maxIt - minIt);
    const y = padT + H - normY * H;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Точки та підписи осі X
  ctx.fillStyle = '#e53e3e';
  sweep.forEach((pt, idx) => {
    const x = padL + ((pt.omega - 0.05) / 1.9) * W;
    const normY = (maxIt === minIt) ? 0.5 : (pt.iterations - minIt) / (maxIt - minIt);
    const y = padT + H - normY * H;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();

    if (idx % 4 === 0) {
      ctx.fillStyle = '#4a5568';
      ctx.fillText(pt.omega.toFixed(2), x - 12, padT + H + 18);
      ctx.fillStyle = '#e53e3e';
    }
  });
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
            result = process_calculation(data)

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Приховує зайві логи HTTP-запитів у консолі


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"Сервер запущено: {url}")
    print("Для зупинки натисніть Ctrl+C у терміналі.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with socketserver.TCPServer(("127.0.0.1", PORT), RequestHandler) as httpd:
        httpd.serve_forever()