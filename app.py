import streamlit as st
import pandas as pd
import numpy as np
import pulp
from dataclasses import dataclass

# -----------------------------
# Data container
# -----------------------------
@dataclass
class OptimizationData:
    products: list[str]
    suppliers: list[str]
    minimums: np.ndarray
    demand: np.ndarray
    costs: np.ndarray


# -----------------------------
# CSV reader (faithful to Julia)
# -----------------------------
def read_optimization_data(file) -> OptimizationData:
    # Suporta CSV (ponto-e-vírgula ou vírgula) e arquivos Excel (.xls/.xlsx)
    name = getattr(file, "name", None)
    # garantir que o ponteiro do arquivo volte ao início para múltiplas tentativas
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass

    df = None
    try:
        if name and name.lower().endswith((".xls", ".xlsx")):
            try:
                df = pd.read_excel(file, dtype=str, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file, dtype=str)
        else:
            # primeiro tenta o separador ';' (comum em exports brasileiros)
            try:
                df = pd.read_csv(file, dtype=str, sep=';')
            except Exception:
                if hasattr(file, "seek"):
                    try:
                        file.seek(0)
                    except Exception:
                        pass
                df = pd.read_csv(file, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler o arquivo: {e}")

    def parse_clean(x):
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return float('nan')
        return float(s.replace(",", "."))

    # Normalizar cabeçalhos
    df.columns = df.columns.astype(str).str.strip()

    products = df.iloc[1:, 0].tolist()
    suppliers = df.columns[2:].tolist()

    demand = df.iloc[1:, 1].map(parse_clean).to_numpy()
    minimums = df.iloc[0, 2:].map(parse_clean).to_numpy()

    F = len(suppliers)
    P = len(products)

    costs = np.zeros((F, P))
    for i in range(F):
        for j in range(P):
            costs[i, j] = parse_clean(df.iloc[j + 1, i + 2])

    return OptimizationData(products, suppliers, minimums, demand, costs)


# -----------------------------
# Optimization model
# -----------------------------
def solve_model(opt: OptimizationData, max_suppliers: int):
    F = len(opt.suppliers)
    P = len(opt.products)

    model = pulp.LpProblem("Procurement", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", (range(F), range(P)), lowBound=0, cat="Integer"
    )
    y = pulp.LpVariable.dicts(
        "y", range(F), cat="Binary"
    )

    # Objective
    model += pulp.lpSum(
        opt.costs[i, j] * x[i][j]
        for i in range(F)
        for j in range(P)
    )

    # Demand
    for j in range(P):
        model += pulp.lpSum(x[i][j] for i in range(F)) >= opt.demand[j]

    # Supplier minimums
    for i in range(F):
        model += pulp.lpSum(
            opt.costs[i, j] * x[i][j] for j in range(P)
        ) >= opt.minimums[i] * y[i]

    # Linking constraint
    for i in range(F):
        for j in range(P):
            model += x[i][j] <= opt.demand[j] * y[i]

    # 🚚 Max number of suppliers (NEW)
    model += pulp.lpSum(y[i] for i in range(F)) <= max_suppliers

    # Tentar HiGHS (mais rápido quando disponível), caso contrário usar CBC como fallback
    try:
        solver = pulp.HiGHS(msg=False)
        model.solve(solver)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError("No optimal solution found")

    x_sol = np.array([[x[i][j].value() for j in range(P)] for i in range(F)])
    y_sol = np.array([y[i].value() for i in range(F)])
    cost = pulp.value(model.objective)

    return x_sol, y_sol, cost


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Procurement Optimization")

# Permitir upload de CSV e Excel (xls/xlsx) — útil para arquivos gerados no Brasil
file = st.file_uploader("Upload CSV ou Excel", type=["csv", "xls", "xlsx"])

if file:
    data = read_optimization_data(file)

    max_k = st.slider(
        "Maximum number of suppliers (deliveries)",
        min_value=1,
        max_value=len(data.suppliers),
        value=len(data.suppliers)
    )

    if st.button("Run optimization"):
        x, y, cost = solve_model(data, max_k)

        st.subheader("Total Cost (BRL)")
        st.metric("Optimal Cost", f"{cost:,.2f}")

        st.subheader("Selected Suppliers")
        sel_df = pd.DataFrame({
            "Supplier": data.suppliers,
            "Selected": y.astype(int)
        })
        st.dataframe(sel_df)

        st.subheader("Purchase Quantities")
        qty_df = pd.DataFrame(
            x.astype(int),
            index=data.suppliers,
            columns=data.products
        )
        st.dataframe(qty_df)
