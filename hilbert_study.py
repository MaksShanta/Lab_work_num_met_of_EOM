import numpy as np
from gauss_solver import solve_gauss_with_pivot, matrix_condition_number

def generate_hilbert_system(m):
    """
    Генерує матрицю Гільберта a_ij = 1 / (i + j - 1),
    точний вектор x_i = i^2 та праву частину f = A * x.
    Індексація в формулах математична: від 1 до m.
    """
    i_indices = np.arange(1, m + 1).reshape(-1, 1)
    j_indices = np.arange(1, m + 1).reshape(1, -1)
    
    # a_ij = 1 / (i + j - 1)
    A = 1.0 / (i_indices + j_indices - 1.0)
    
    # x_i = i^2
    exact_x = (np.arange(1, m + 1, dtype=np.float64)) ** 2
    
    # f = A * x
    f = A @ exact_x
    
    return A, exact_x, f

def run_hilbert_experiments(m_values=[8, 10, 12, 14]):
    results = {}
    
    for m in m_values:
        A, exact_x, f = generate_hilbert_system(m)
        
        # Викликаємо Гаусс із порогом 0.0
        calc_x, det_A, is_consistent, msg = solve_gauss_with_pivot(A, f, eps=0.0)
        
        # Обчислюємо похибку завжди
        abs_err = np.max(np.abs(calc_x - exact_x))
        rel_err = abs_err / np.max(np.abs(exact_x))
            
        cond_A, _ = matrix_condition_number(A)
        
        results[m] = {
            "exact_x": exact_x,
            "calc_x": calc_x,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "cond_A": cond_A,
            "det_A": det_A
        }
        
    return results

if __name__ == "__main__":
    print(f"{'m':<5} | {'cond(A)':<15} | {'det(A)':<15} | {'Абс. похибка':<15} | {'Відн. похибка':<15}")
    print("-" * 75)
    res = run_hilbert_experiments()
    for m, data in res.items():
        print(f"{m:<5} | {data['cond_A']:<15.4e} | {data['det_A']:<15.4e} | {data['abs_err']:<15.4e} | {data['rel_err']:<15.4e}")
        