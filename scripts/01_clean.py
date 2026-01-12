import pandas as pd
import numpy as np

INPUT_XLSX = "data/raw/respostas.xlsx"
OUT_CSV = "data/processed/respostas_clean.csv"
OUT_CODEBOOK = "docs/codebook.csv"


def find_col(df: pd.DataFrame, contains: list[str]) -> str:
    """Encontra coluna cujo nome contenha TODOS os trechos em contains (case-insensitive)."""
    for c in df.columns:
        name = c.lower()
        if all(s.lower() in name for s in contains):
            return c
    raise KeyError(f"Coluna não encontrada com trechos: {contains}")


def split_multi(series: pd.Series) -> pd.Series:
    """
    Google Forms exporta múltipla seleção como string separada por vírgula.
    Ex.: 'Celular, Notebook próprio'
    Retorna lista padronizada.
    """
    def _split(x):
        if pd.isna(x):
            return []
        # separador mais comum no seu arquivo: ", "
        parts = [p.strip() for p in str(x).split(",")]
        parts = [p for p in parts if p]
        return parts

    return series.apply(_split)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main():
    df = pd.read_excel(INPUT_XLSX)

    # --------------------------
    # 1) Remover colunas sensíveis e auxiliares
    # --------------------------
    drop_cols = []

    # timestamp
    for c in df.columns:
        cl = c.lower()
        if "carimbo de data/hora" in cl or "timestamp" in cl:
            drop_cols.append(c)

    # consentimento (todo mundo aceitou; não agrega na análise)
    consent_col = df.columns[1]  # no seu arquivo é a 2ª coluna
    drop_cols.append(consent_col)

    # e-mail de contato (sensível)
    email_contact = [c for c in df.columns if "se sim, informe seu e-mail" in c.lower()]
    drop_cols.extend(email_contact)

    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # --------------------------
    # 2) Renomear colunas para nomes curtos (variáveis)
    # --------------------------
    mapping = {}

    # Perfil
    mapping[find_col(df, ["idade"])] = "perfil_idade"
    mapping[find_col(df, ["identidade de gênero"])] = "perfil_genero"
    mapping[find_col(df, ["curso"])] = "perfil_curso"
    mapping[find_col(df, ["período atual"])] = "perfil_periodo"
    mapping[find_col(df, ["frequência", "laboratórios"])] = "perfil_lab_freq"
    mapping[find_col(df, ["dispositivos que utiliza"])] = "perfil_dispositivos"
    mapping[find_col(df, ["já usou wi-fi"])] = "perfil_wifi_publico"
    mapping[find_col(df, ["disciplina", "segurança da informação"])] = "perfil_si_disc_ativ"
    mapping[find_col(df, ["atividades abaixo", "ti"])] = "perfil_ativ_ti"
    mapping[find_col(df, ["contato", "fora da universidade"])] = "perfil_contato_externo"
    mapping[find_col(df, ["incidente de segurança"])] = "perfil_incidente"
    mapping[find_col(df, ["sistema operacional"])] = "perfil_so"
    mapping[find_col(df, ["tempo médio"])] = "perfil_tempo_uso"
    mapping[find_col(df, ["ouviu falar", "segurança da informação"])] = "perfil_ouviu_termo"
    mapping[find_col(df, ["autoavaliação"])] = "perfil_autoavaliacao_ti"
    mapping[find_col(df, ["com qual área da tecnologia"])] = "perfil_area_interesse"
    mapping[find_col(df, ["autoriza ser contatado"])] = "perfil_autoriza_contato"

    # Conhecimento objetivo (Q20-Q24)
    mapping[find_col(df, ["qual destas senhas"])] = "obj_q20_senha_segura"
    mapping[find_col(df, ["o que é phishing"])] = "obj_q21_phishing"
    mapping[find_col(df, ["importante manter", "atualiz"])] = "obj_q22_atualizacoes"
    mapping[find_col(df, ["forma mais segura", "wi-fi pública"])] = "obj_q23_wifi_publica"
    mapping[find_col(df, ["melhor prática", "uso de senhas"])] = "obj_q24_pratica_senhas"

    # Conhecimento declarado (Q25-Q30)
    mapping[find_col(df, ["autenticação em dois fatores"])] = "likert_q25_2fa"
    mapping[find_col(df, ["identificar um link suspeito"])] = "likert_q26_link_suspeito"
    mapping[find_col(df, ["proteger meus dispositivos", "redes públicas"])] = "likert_q27_protecao_rede_publica"
    mapping[find_col(df, ["riscos", "desbloqueado"])] = "likert_q28_risco_desbloqueado"
    mapping[find_col(df, ["configurar permissões", "google drive"])] = "likert_q29_permissoes"
    mapping[find_col(df, ["compartilhar informações pessoais"])] = "likert_q30_compartilha_social"

    # Percepção de risco (Q31-Q36)
    mapping[find_col(df, ["posso ser vítima", "golpes"])] = "risk_q31_vitima_golpes"
    mapping[find_col(df, ["arriscado utilizar redes wi-fi"])] = "risk_q32_wifi_arriscado"
    mapping[find_col(df, ["receoso", "computadores dos laboratórios"])] = "risk_q33_receio_labs"
    mapping[find_col(df, ["colegas", "acessarem", "arquivos"])] = "risk_q34_receio_arquivos"
    mapping[find_col(df, ["dados pessoais", "vazados"])] = "risk_q35_medovazamento"
    mapping[find_col(df, ["já evitei", "por medo"])] = "risk_q36_evitacao"

    # Conscientização (Q37-Q42)
    mapping[find_col(df, ["atualizo o sistema operacional"])] = "cons_q37_atualiza_so"
    mapping[find_col(df, ["evito deixar senhas salvas"])] = "cons_q38_nao_salva_senhas"
    mapping[find_col(df, ["senhas diferentes"])] = "cons_q39_senhas_diferentes"
    mapping[find_col(df, ["bloqueio a tela"])] = "cons_q40_bloqueia_tela"
    mapping[find_col(df, ["atento", "golpes", "e-mails"])] = "cons_q41_atento_golpes"
    mapping[find_col(df, ["logout", "computador público"])] = "cons_q42_logout_publico"

    # “Comportamentos inseguros” (Q43-Q50)
    mapping[find_col(df, ["evito compartilhar meus dados pessoais"])] = "beh_q43_evitar_compartilhar_dados"  # item seguro
    mapping[find_col(df, ["mesma senha para diferentes sistemas"])] = "beh_q44_reutiliza_senha"
    mapping[find_col(df, ["compartilho senhas com colegas"])] = "beh_q45_compartilha_senha"
    mapping[find_col(df, ["deixo meu computador", "desbloqueado"])] = "beh_q46_deixa_desbloqueado"
    mapping[find_col(df, ["acesso contas pessoais", "computadores públicos"])] = "beh_q47_conta_pessoal_pc_publico"
    mapping[find_col(df, ["wi-fi abertas", "vpn"])] = "beh_q48_wifi_sem_protecao"
    mapping[find_col(df, ["cliquei em links", "origem duvidosa"])] = "beh_q49_clica_link_duvidoso"
    mapping[find_col(df, ["compartilho arquivos", "sem verificar"])] = "beh_q50_compartilha_sem_verificar"

    # Abertas (Q51-Q55)
    mapping[find_col(df, ["por que você acredita que estudantes universitários adotam"])] = "open_q51_motivos_inseguranca"
    mapping[find_col(df, ["disciplina, palestra ou orientação"])] = "open_q52_orientacao_universidade"
    mapping[find_col(df, ["prática que você considera segura", "dificuldade"])] = "open_q53_dificuldade_pratica_segura"
    mapping[find_col(df, ["universidade contribui", "ambiente digital seguro"])] = "open_q54_melhorias_universidade"
    mapping[find_col(df, ["que tipo de orientação ou recurso"])] = "open_q55_recursos_desejados"

    # aplicar rename
    df_renamed = df.rename(columns=mapping)

    # --------------------------
    # 3) Padronizações (múltipla seleção e Likert)
    # --------------------------
    # múltipla seleção
    for col in ["perfil_dispositivos", "perfil_si_disc_ativ", "perfil_ativ_ti", "perfil_contato_externo", "perfil_incidente"]:
        if col in df_renamed.columns:
            df_renamed[col] = split_multi(df_renamed[col])

    # Likert numérico (1..5)
    likert_cols = [c for c in df_renamed.columns if c.startswith(("likert_", "risk_", "cons_", "beh_"))]
    for c in likert_cols:
        # algumas beh_ são "discordo..concordo" e outras "nunca..sempre"; ainda assim são 1..5
        df_renamed[c] = to_num(df_renamed[c])

    # --------------------------
    # 4) Salvar outputs
    # --------------------------
    df_renamed.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # codebook: original -> novo
    codebook = pd.DataFrame({
        "coluna_original": list(mapping.keys()),
        "coluna_no_dataset": list(mapping.values())
    }).sort_values("coluna_no_dataset")

    codebook.to_csv(OUT_CODEBOOK, index=False, encoding="utf-8")

    print("OK!")
    print("Linhas:", df_renamed.shape[0], "| Colunas:", df_renamed.shape[1])
    print("Salvo:", OUT_CSV)
    print("Salvo:", OUT_CODEBOOK)


if __name__ == "__main__":
    main()
