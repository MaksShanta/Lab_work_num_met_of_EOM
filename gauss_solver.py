import numpy as np

def solve_gauss_with_pivot(A_input, b_input, eps=0.0):
    """
    Розв'язання системи Ax = b методом Гаусса з вибором головного елемента по стовпцю.
    """
    A = np.array(A_input, dtype=np.float64, copy=True)
    b = np.array(b_input, dtype=np.float64, copy=True)
    
    m = len(b)
    det_sign = 1.0
    diag_elements = []

    #  ПРЯМИЙ ХІД 
    for k in range(m):
        pivot_row = k + np.argmax(np.abs(A[k:m, k]))
        max_val = np.abs(A[pivot_row, k])

        if max_val <= eps:
            diag_elements.append(0.0)
            continue

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            det_sign *= -1.0

        pivot = A[k, k]
        diag_elements.append(pivot)

        # Нормування
        A[k, k+1:] = A[k, k+1:] / pivot
        b[k] = b[k] / pivot
        A[k, k] = 1.0

        # Виключення
        for i in range(k + 1, m):
            factor = A[i, k]
            if factor != 0.0:
                A[i, k+1:] -= factor * A[k, k+1:]
                b[i] -= factor * b[k]
            A[i, k] = 0.0

    det_A = det_sign * np.prod(diag_elements)

    #  ОБЕРНЕНИЙ ХІД 
    x = np.zeros(m, dtype=np.float64)
    for i in range(m - 1, -1, -1):
        x[i] = b[i] - np.sum(A[i, i+1:] * x[i+1:])

    return x, det_A, True, "Розв'язок знайдено."


def inverse_matrix_gauss(A_input, eps=0.0):
    """
    Знаходження оберненої матриці розв'язанням A * X = E.
    """
    A = np.array(A_input, dtype=np.float64, copy=True)
    m = A.shape[0]
    inv_A = np.zeros((m, m), dtype=np.float64)
    
    for j in range(m):
        e_j = np.zeros(m, dtype=np.float64)
        e_j[j] = 1.0
        x_col, _, _, _ = solve_gauss_with_pivot(A, e_j, eps=eps)
        if x_col is None:
            return None, "Помилка знаходження оберненої матриці."
        inv_A[:, j] = x_col
        
    return inv_A, "Обернену матрицю знайдено."


def matrix_condition_number(A_input):
    """
    Обчислення числа обумовленості M_A = cond(A) = ||A|| * ||A^(-1)||.
    Якщо прямий пошук дає виродження, використовуємо псевдообернення за SVD.
    """
    A = np.array(A_input, dtype=np.float64)
    norm_A = np.max(np.sum(np.abs(A), axis=1))
    
    inv_A, _ = inverse_matrix_gauss(A, eps=0.0)
    
    if inv_A is not None and not np.any(np.isnan(inv_A)) and not np.any(np.isinf(inv_A)):
        norm_inv_A = np.max(np.sum(np.abs(inv_A), axis=1))
    else:
        # Для критично вироджених матриць Гільберта беремо псевдообернену
        inv_A = np.linalg.pinv(A)
        norm_inv_A = np.max(np.sum(np.abs(inv_A), axis=1))
        
    cond_A = norm_A * norm_inv_A
    return cond_A, inv_A