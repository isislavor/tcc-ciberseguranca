import pandas as pd

INPUT = "data/processed/respostas_clean.csv"
OUTPUT = "data/processed/respostas_indices.csv"


def mean_index(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Média por linha ignorando valores ausentes."""
    return df[cols].mean(axis=1, skipna=True)


def main():
    df = pd.read_csv(INPUT)

    # ------------------------------------------------------------
    # 1) Conhecimento técnico objetivo (Q20–Q24) -> índice 0..5
    # ------------------------------------------------------------
    gabarito = {
        "obj_q20_senha_segura": "@L1ne27329!",
        "obj_q21_phishing": "Um golpe digital que visa enganar usuários para obter dados sensíveis",
        "obj_q22_atualizacoes": "Para corrigir falhas de segurança que podem ser exploradas por atacantes",
        "obj_q23_wifi_publica": "Usar uma VPN ou protocolo HTTPS em todas as conexões",
        "obj_q24_pratica_senhas": "Utilizar gerenciadores de senha e senhas únicas para cada serviço",
    }

    for col, correta in gabarito.items():
        df[col] = (df[col].astype(str).str.strip() == correta).astype(int)

    df["idx_conhecimento_objetivo"] = df[list(gabarito.keys())].sum(axis=1)

    # ------------------------------------------------------------
    # 2) Conhecimento declarado (Q25–Q30) -> média 1..5
    # ------------------------------------------------------------
    conhecimento_cols = [
        "likert_q25_2fa",
        "likert_q26_link_suspeito",
        "likert_q27_protecao_rede_publica",
        "likert_q28_risco_desbloqueado",
        "likert_q29_permissoes",
        "likert_q30_compartilha_social",
    ]
    for c in conhecimento_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["idx_conhecimento_declarado"] = mean_index(df, conhecimento_cols)

    # ------------------------------------------------------------
    # 3) Percepção de risco (Q31–Q36) -> média 1..5
    # ------------------------------------------------------------
    risco_cols = [
        "risk_q31_vitima_golpes",
        "risk_q32_wifi_arriscado",
        "risk_q33_receio_labs",
        "risk_q34_receio_arquivos",
        "risk_q35_medovazamento",
        "risk_q36_evitacao",
    ]
    for c in risco_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["idx_percepcao_risco"] = mean_index(df, risco_cols)

    # ------------------------------------------------------------
    # 4) Conscientização em segurança (Q37–Q43) -> média 1..5
    # ------------------------------------------------------------
    consc_cols = [
        "cons_q37_atualiza_so",
        "cons_q38_nao_salva_senhas",
        "cons_q39_senhas_diferentes",
        "cons_q40_bloqueia_tela",
        "cons_q41_atento_golpes",
        "cons_q42_logout_publico",
        "beh_q43_evitar_compartilhar_dados",  # Q43 ainda é conscientização (protetiva)
    ]
    for c in consc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["idx_conscientizacao"] = mean_index(df, consc_cols)

    # ------------------------------------------------------------
    # 5) Comportamentos digitais inseguros (Q44–Q50) -> média 1..5
    # ------------------------------------------------------------
    inseguro_cols = [
        "beh_q44_reutiliza_senha",
        "beh_q45_compartilha_senha",
        "beh_q46_deixa_desbloqueado",
        "beh_q47_conta_pessoal_pc_publico",
        "beh_q48_wifi_sem_protecao",
        "beh_q49_clica_link_duvidoso",
        "beh_q50_compartilha_sem_verificar",
    ]
    for c in inseguro_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["idx_comportamento_inseguro"] = mean_index(df, inseguro_cols)

    # ------------------------------------------------------------
    # Salvar
    # ------------------------------------------------------------
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    print("OK!")
    print("Arquivo salvo em:", OUTPUT)
    print("Colunas totais:", df.shape[1])


if __name__ == "__main__":
    main()
