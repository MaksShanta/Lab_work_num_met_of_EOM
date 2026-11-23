#Інструкція:
#Код запускається задопомогою команди: python -m streamlit run app_lab1.py
#Зупиняжться за допомогою комбінації клавіш: `Ctrl+C` .
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gauss_solver import solve_gauss_with_pivot, inverse_matrix_gauss, matrix_condition_number
from hilbert_study import generate_hilbert_system, run_hilbert_experiments

st.set_page_config(page_title="Лабораторна робота №1 | Метод Гаусса", layout="wide")

st.title("🧮 Лабораторна робота №1: Метод Гаусса та погано обумовлені СЛАР")
st.markdown("**Студент 3 курсу** | Варіант 1.16")

tabs = st.tabs(["📌 Частина 1: Розв'язання СЛАР (Варіант 1.16 / Власний ввід)", 
               "📉 Частина 2: Дослідження матриці Гільберта"])

#  ВКЛАДКА 1: Розв'язання довільної СЛАР та варіант 1.16 
with tabs[0]:
    st.subheader("Розв'язання СЛАР, обчислення det(A) та A⁻¹")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        mode = st.radio("Оберіть режим вводу:", ["Варіант 1.16 (за замовчуванням)", "Ручний ввід матриці"])
        
        if mode == "Варіант 1.16 (за замовчуванням)":
            m_size = 3
            default_A = "2.0  1.0 -1.0\n1.0  1.0  1.0\n3.0 -1.0  1.0"
            default_b = "2.0\n6.0\n8.0"
        else:
            m_size = st.number_input("Розмірність системи (m):", min_value=2, max_value=10, value=3)
            default_A = "1.0 0.0\n0.0 1.0"
            default_b = "1.0\n1.0"
            
        A_text = st.text_area("Матриця A (елементи рядка через пробіл, новий рядок — новий рядок):", 
                              value=default_A, height=120)
        b_text = st.text_area("Вектор правої частини f (елементи через пробіл або новий рядок):", 
                              value=default_b, height=90)
        
        calc_btn = st.button("Обчислити СЛАР", type="primary")

    with col2:
        if calc_btn or mode == "Варіант 1.16 (за замовчуванням)":
            try:
                A = np.array([[float(num) for num in line.split()] for line in A_text.strip().split("\n") if line.strip()])
                b = np.array([float(num) for num in b_text.strip().replace("\n", " ").split() if num])
                
                if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
                    st.error("Помилка: Розмірність матриці A та вектора f не збігаються!")
                else:
                    x, det_A, is_consistent, msg = solve_gauss_with_pivot(A, b)
                    cond_A, inv_A = matrix_condition_number(A)
                    
                    st.success(f"**Статус:** {msg}")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Визначник det(A)", f"{det_A:.6f}")
                    with res_col2:
                        st.metric("Число обумовленості cond(A)", f"{cond_A:.6f}" if np.isfinite(cond_A) else "∞")
                        
                    if is_consistent and x is not None:
                        st.markdown("#### Знайдений вектор невідомих $x$:")
                        df_x = pd.DataFrame({"Змінна": [f"x_{i+1}" for i in range(len(x))], "Обчислене значення": x})
                        st.table(df_x)
                        
                        if inv_A is not None:
                            st.markdown("#### Обернена матриця $A^{-1}$:")
                            st.dataframe(pd.DataFrame(inv_A))
                    else:
                        st.warning("Розв'язок неможливо знайти (матриця вироджена або система несумісна).")
            except Exception as e:
                st.error(f"Помилка зчитування даних: {e}")

#  ВКЛАДКА 2: Дослідження матриці Гільберта 
with tabs[1]:
    st.subheader("Дослідження погано обумовлених систем на прикладі матриці Гільберта")
    st.markdown(r"""
    Компоненти матриці: $a_{ij} = \frac{1}{i + j - 1}$, точний розв'язок: $x_i = i^2$, вектор правої частини: $\bar{f} = A\bar{x}$.
    """)
    
    selected_m = st.multiselect("Виберіть розмірності m для дослідження:", 
                                options=[4, 6, 8, 10, 12, 14, 16, 17, 20], default=[8, 10, 12, 14])
    
    if st.button("Провести чисельний експеримент", type="primary"):
        results = run_hilbert_experiments(selected_m)
        
        table_data = []
        for m, data in results.items():
            table_data.append({
                "m": m,
                "cond(A)": f"{data['cond_A']:.4e}",
                "det(A)": f"{data['det_A']:.4e}",
                "Абсолютна похибка": f"{data['abs_err']:.4e}",
                "Відносна похибка": f"{data['rel_err']:.4e}"
            })
        
        st.markdown("### Зведена таблиця результатів")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Візуалізація відхилення
        st.markdown("### Візуалізація: Порівняння точного розв'язку та наближеного")
        
        fig, axes = plt.subplots(len(selected_m), 1, figsize=(10, 4 * len(selected_m)))
        if len(selected_m) == 1:
            axes = [axes]
            
        for ax, m in zip(axes, selected_m):
            data = results[m]
            idx = np.arange(1, m + 1)
            exact = data['exact_x']
            calc = data['calc_x']
            
            ax.plot(idx, exact, 'o-', label='Точний розв’язок $x_i = i^2$', color='black', linewidth=1.5)
            if calc is not None:
                ax.plot(idx, calc, 's--', label=f'Метод Гаусса (m={m})', color='crimson', markersize=5)
            ax.set_title(f"Розмірність m = {m} | cond(A) = {data['cond_A']:.2e} | Похибка = {data['abs_err']:.2e}")
            ax.set_xlabel("Номер компоненти i")
            ax.set_ylabel("Значення $x_i$")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            
        plt.tight_layout()
        st.pyplot(fig)