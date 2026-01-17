import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "data/processed/respostas_indices.csv"
OUT_DIR = "outputs/grupos"
OUT_TABLE_DIR = "outputs/tabelas"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_TABLE_DIR, exist_ok=True)

INDICES = [
    "idx_comportamento_inseguro",
    "idx_conscientizacao",
    "idx_conhecimento_objetivo",
    "idx_conhecimento_declarado",
    "idx_percepcao_risco",
]


def coalesce_col(df, candidates):
    """Retorna o primeiro nome de coluna existente em df, dentre os candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_parse_list(x):
    """Converte strings de múltipla seleção em lista (robusto para CSV)."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    # fallback: separar por vírgula
    return [p.strip() for p in s.split(",") if p.strip()]


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def map_macro_area(raw):
    """
    Agrupa diversas descrições de 'área' em macrogrupos legíveis.
    """
    x = str(raw).strip().lower()

    # normalizações básicas
    x = x.replace("ui ux", "ui/ux").replace("ux/ui", "ui/ux")

    # vazio/indefinido
    if x in ["", "nan", "none", "não sei", "nao sei", "n/a", "null"]:
        return "Não informado"

    # Segurança
    if any(k in x for k in ["segurança", "security", "ciber", "pentest", "soc", "blue team", "red team"]):
        return "Segurança"

    # QA/Testes
    if any(k in x for k in ["qa", "teste", "testes", "tester", "automação", "automacao", "automation"]):
        return "QA/Testes"

    # UI/UX & Design
    if any(k in x for k in ["ui", "ux", "ui/ux", "design", "product design", "designer"]):
        return "UI/UX & Design"

    # Dados/IA
    if any(k in x for k in ["dados", "data", "bi", "analytics", "cientista de dados", "machine learning", "ml", "ia", "ai"]):
        return "Dados/IA"

    # Infra/Redes/DevOps (inclui devsecops)
    if any(k in x for k in ["infra", "infraestrutura", "redes", "network", "devops", "devsecops", "cloud", "sre", "sysadmin"]):
        return "Infra/Redes/DevOps"

    # Gestão/Produto
    if any(k in x for k in ["produto", "product", "pm", "po", "gestão", "gestao", "gerente", "manager"]):
        return "Gestão/Produto"

    # Desenvolvimento (catch-all dev)
    if any(k in x for k in ["dev", "developer", "desenvolv", "program", "backend", "frontend", "fullstack", "mobile", "software"]):
        return "Desenvolvimento"

    return "Outros"


def create_boxplot(df, group_col, idx_col, title, filename):
    """Gera boxplot por grupo, ordenando categorias pelo tamanho do grupo."""
    plt.figure(figsize=(10, 5))

    # ordenar grupos por frequência (maior -> menor)
    order = df[group_col].value_counts().index.tolist()
    df2 = df.copy()
    df2[group_col] = pd.Categorical(df2[group_col], categories=order, ordered=True)

    df2.boxplot(column=idx_col, by=group_col)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel(group_col)
    plt.ylabel(idx_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def save_group_summary(df, group_col, idx_cols, out_csv):
    """Salva uma tabela com count/mean/std por grupo para cada índice."""
    summary = df.groupby(group_col)[idx_cols].agg(["count", "mean", "std"])
    summary.to_csv(out_csv, encoding="utf-8")


def main():
    df = pd.read_csv(INPUT)

    # -----------------------------
    # 1) Identificar colunas de perfil (com tolerância)
    # -----------------------------
    col_curso = coalesce_col(df, ["perfil_curso", "curso"])
    col_area = coalesce_col(df, ["perfil_area_interesse", "perfil_area_atuacao", "perfil_area", "area", "atuacao"])
    col_si_multi = coalesce_col(df, ["perfil_si_disc_ativ", "perfil_si", "si_disciplina"])

    missing = [
        name for name, col in [
            ("curso", col_curso),
            ("area", col_area),
            ("si_disc_ativ", col_si_multi),
        ] if col is None
    ]
    if missing:
        print("⚠️ Aviso: não encontrei estas colunas:", missing)
        print("Colunas disponíveis:", list(df.columns))
        print("O script vai rodar com o que estiver disponível.\n")

    # -----------------------------
    # 2) Preparar grupos
    # -----------------------------
    # 2.1 Curso
    if col_curso:
        df["grupo_curso"] = df[col_curso].astype(str).str.strip()

    # 2.2 Área (macrogrupos)
    if col_area:
        df["grupo_area"] = df[col_area].apply(map_macro_area)

    # 2.3 Disciplina/atividade de SI (Sim / Não / Não me lembro)
    if col_si_multi:
        parsed = df[col_si_multi].apply(safe_parse_list)

        def classify_si(items):
            norm = [normalize_text(i) for i in items]
            joined = " | ".join(norm)

            has_sim_disc = ("sim" in joined and "disciplina" in joined)
            has_sim_ativ = ("sim" in joined and "atividade" in joined) or ("atividade complementar" in joined)
            has_autodidata = ("autodidata" in joined)

            if has_sim_disc or has_sim_ativ or has_autodidata:
                return "Sim"

            # prioridade para incerteza
            if "não me lembro" in joined or "nao me lembro" in joined:
                return "Não me lembro"

            if "não tive nenhuma" in joined or "nao tive nenhuma" in joined:
                return "Não"

            return "Não"

        df["grupo_cursou_si"] = parsed.apply(classify_si)

    # -----------------------------
    # 3) Definir quais grupos serão analisados
    # -----------------------------
    group_configs = []
    if "grupo_curso" in df.columns:
        group_configs.append(("grupo_curso", "Curso"))
    if "grupo_area" in df.columns:
        group_configs.append(("grupo_area", "Área (macrogrupos)"))
    if "grupo_cursou_si" in df.columns:
        group_configs.append(("grupo_cursou_si", "Cursou disciplina/atividade de SI"))

    if not group_configs:
        print("❌ Nenhum grupo foi criado. Verifique os nomes de colunas.")
        return

    idx_exist = [c for c in INDICES if c in df.columns]
    if not idx_exist:
        print("❌ Nenhum índice encontrado no dataset. Verifique se você está usando respostas_indices.csv.")
        return

    for idx in idx_exist:
        df[idx] = pd.to_numeric(df[idx], errors="coerce")

    # -----------------------------
    # 4) Gerar gráficos e tabelas
    # -----------------------------
    for gcol, glabel in group_configs:
        dfg = df.dropna(subset=[gcol]).copy()

        dfg[gcol] = dfg[gcol].astype(str).str.strip()
        dfg = dfg[dfg[gcol] != ""]

        out_summary = f"{OUT_TABLE_DIR}/resumo_{gcol}.csv"
        save_group_summary(dfg, gcol, idx_exist, out_summary)

        for idx in idx_exist:
            filename = f"{OUT_DIR}/box_{idx}_por_{gcol}.png"
            title = f"{idx} por {glabel}"
            create_boxplot(dfg, gcol, idx, title, filename)

        print(f"✅ Grupo '{gcol}' OK | tabela: {out_summary}")

    print("\nOK! Gráficos em:", OUT_DIR)
    print("Tabelas em:", OUT_TABLE_DIR)


if __name__ == "__main__":
    main()
