import pandas as pd
import matplotlib.pyplot as plt

INPUT = "data/processed/respostas_indices.csv"
OUT_TABLE = "outputs/tabelas/resumo_indices.csv"
OUT_FIG = "outputs/graficos"

indices = [
    "idx_conhecimento_objetivo",
    "idx_conhecimento_declarado",
    "idx_percepcao_risco",
    "idx_conscientizacao",
    "idx_comportamento_inseguro",
]

def main():
    df = pd.read_csv(INPUT)

    # ----------------------------
    # Tabela descritiva
    # ----------------------------
    resumo = (
        df[indices]
        .describe()
        .T[["count", "mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
    )

    resumo.to_csv(OUT_TABLE, encoding="utf-8")
    print("Tabela salva em:", OUT_TABLE)

    # ----------------------------
    # Gráficos
    # ----------------------------
    for col in indices:
        # Histograma
        plt.figure()
        df[col].dropna().hist(bins=10)
        plt.title(col)
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"{OUT_FIG}/hist_{col}.png", dpi=200)
        plt.close()

        # Boxplot
        plt.figure()
        df[[col]].boxplot()
        plt.title(col)
        plt.tight_layout()
        plt.savefig(f"{OUT_FIG}/box_{col}.png", dpi=200)
        plt.close()

    print("Gráficos gerados em:", OUT_FIG)

if __name__ == "__main__":
    main()
