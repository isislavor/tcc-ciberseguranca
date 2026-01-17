import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INPUT = "data/processed/respostas_indices.csv"
OUT_DIR = "outputs/cruzamentos"
OUT_TABLE = "outputs/tabelas/correlacoes.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/tabelas", exist_ok=True)

# pares (X, Y, rótulo)
CROSSINGS = [
    ("idx_conhecimento_objetivo", "idx_comportamento_inseguro", "Conhecimento objetivo × Comportamento inseguro"),
    ("idx_conhecimento_declarado", "idx_comportamento_inseguro", "Conhecimento declarado × Comportamento inseguro"),
    ("idx_percepcao_risco", "idx_comportamento_inseguro", "Percepção de risco × Comportamento inseguro"),
    ("idx_conscientizacao", "idx_comportamento_inseguro", "Conscientização × Comportamento inseguro"),
]

def main():
    df = pd.read_csv(INPUT)

    results = []

    for x, y, label in CROSSINGS:
        data = df[[x, y]].dropna()

        # Correlação de Spearman
        rho, p = spearmanr(data[x], data[y])

        results.append({
            "variavel_x": x,
            "variavel_y": y,
            "rho_spearman": rho,
            "p_valor": p
        })

        # Gráfico de dispersão
        plt.figure()
        plt.scatter(data[x], data[y], alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{label}\nρ = {rho:.3f} | p = {p:.4f}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{x}_vs_{y}.png", dpi=200)
        plt.close()

    # Tabela de correlações
    corr_df = pd.DataFrame(results)
    corr_df.to_csv(OUT_TABLE, index=False, encoding="utf-8")

    print("OK!")
    print("Tabela salva em:", OUT_TABLE)
    print("Gráficos salvos em:", OUT_DIR)

if __name__ == "__main__":
    main()
