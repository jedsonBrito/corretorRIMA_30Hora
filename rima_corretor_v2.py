import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────

def parse_minutes(time_str):
    try:
        if pd.isna(time_str) or str(time_str).strip() == '':
            return None
        t = pd.to_datetime(str(time_str).strip(), format='%H:%M').time()
        return t.hour * 60 + t.minute
    except:
        return None

def minutes_to_hhmm(minutes):
    if minutes is None:
        return ''
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def get_pax_columns(df):
    candidates = ['PAX_LOCAL', 'PAX_CONEXAO_DOMESTICO', 'PAX_CONEXAO_INTERNACIONAL']
    return [c for c in candidates if c in df.columns]

def extract_icao(cod_rima):
    try:
        return str(cod_rima).strip()[:4].upper()
    except:
        return 'XXXX'

def compute_pax_total(df, pax_cols):
    return df[pax_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

def get_limite(limites_dict, icao, movimento):
    """
    Busca o limite na ordem de especificidade:
      1. (ICAO, Movimento) exato
      2. (ICAO, '*')        — qualquer movimento desse aeroporto
      3. ('DEFAULT', Movimento)
      4. ('DEFAULT', '*')
    """
    return (
        limites_dict.get((icao, movimento)) or
        limites_dict.get((icao, '*')) or
        limites_dict.get(('DEFAULT', movimento)) or
        limites_dict.get(('DEFAULT', '*')) or
        9999
    )

def build_limites_dict(limites_df, limite_default):
    """
    Constrói dict  (ICAO, Movimento) -> int  a partir do data_editor.
    Movimento pode ser 'P', 'D' ou '*' (ambos).
    """
    d = {('DEFAULT', '*'): int(limite_default)}
    for _, r in limites_df.iterrows():
        icao = str(r.get('ICAO', '')).strip().upper()
        mov  = str(r.get('Movimento', '*')).strip().upper()
        val  = r.get('Limite PAX/hora', limite_default)
        if icao and pd.notna(val):
            d[(icao, mov)] = int(val)
    return d

def find_violations(df, time_col, pax_cols, limites_dict):
    """
    Detecta janelas de 60 min que ultrapassam o limite correto para
    cada combinação (ICAO, MOVIMENTO_TIPO, DATA).
    """
    df = df.copy()
    df['__min__'] = df[time_col].apply(parse_minutes)
    df['__pax__'] = compute_pax_total(df, pax_cols)

    if 'PREVISTO_DATA' in df.columns:
        df['__data__'] = pd.to_datetime(df['PREVISTO_DATA'], errors='coerce').dt.date
    else:
        df['__data__'] = pd.Timestamp('today').date()

    mov_col_presente = 'MOVIMENTO_TIPO' in df.columns

    violations = []
    seen = set()

    group_keys = ['__icao__', '__data__']
    if mov_col_presente:
        group_keys.insert(1, 'MOVIMENTO_TIPO')

    for keys, grupo in df.groupby(group_keys):
        if mov_col_presente:
            icao, movimento, data = keys
        else:
            icao, data = keys
            movimento = '*'

        limite = get_limite(limites_dict, icao, movimento)
        grupo = grupo.dropna(subset=['__min__']).sort_values('__min__')
        mov_label = movimento if mov_col_presente else '—'

        for _, row in grupo.iterrows():
            start = int(row['__min__'])
            key = (icao, mov_label, str(data), start)
            if key in seen:
                continue

            ops_janela = grupo[
                (grupo['__min__'] >= start) &
                (grupo['__min__'] < start + 60)
            ]
            pax_total = ops_janela['__pax__'].sum()

            if pax_total > limite:
                seen.add(key)
                violations.append({
                    'icao':        icao,
                    'movimento':   mov_label,
                    'data':        data,
                    'inicio_min':  start,
                    'janela_label': f"{minutes_to_hhmm(start)}–{minutes_to_hhmm(start + 60)}",
                    'pax_janela':  int(pax_total),
                    'limite':      limite,
                    'excesso':     int(pax_total - limite),
                    'indices_ops': ops_janela.index.tolist(),
                })

    return violations

# ─────────────────────────────────────────────────────────────────
# ANÁLISE DE ASSENTOS
# ─────────────────────────────────────────────────────────────────

# Aeroportos que usam análise de assentos
ICAOS_ASSENTOS = {'SBCJ', 'SBHT', 'SBUR', 'SBPP', 'SBCR'}

# Service types válidos para análise de assentos
SERVICE_TYPES_ASSENTOS = {'J', 'C', 'G'}

# Mapa de equipamento -> assentos
ASSENTOS_EQUIPAMENTO = {
    '73H':   186,
    '32N':   174,
    'E195':  118,
    'E295':  136,
    'ATR72':  70,
}

def get_assentos(aeronave_tipo):
    """Retorna o número de assentos do equipamento ou None se não mapeado."""
    if pd.isna(aeronave_tipo):
        return None
    return ASSENTOS_EQUIPAMENTO.get(str(aeronave_tipo).strip().upper())

def find_seat_violations(df, time_col, limites_dict):
    """
    Detecta janelas de 60 min onde a soma de assentos ultrapassa o limite
    para os aeroportos que usam análise de assentos.
    Considera apenas SERVICE_TYPE in {J, C, G} e equipamentos mapeados.
    Agrupa por (ICAO, MOVIMENTO_TIPO, DATA) — mesmo limite de PAX reutilizado.
    """
    df = df.copy()

    # Filtrar aeroportos relevantes
    df = df[df['__icao__'].isin(ICAOS_ASSENTOS)].copy()
    if df.empty:
        return []

    # Filtrar service type
    if 'SERVICE_TYPE' in df.columns:
        df = df[df['SERVICE_TYPE'].str.strip().str.upper().isin(SERVICE_TYPES_ASSENTOS)].copy()
    if df.empty:
        return []

    # Mapear assentos
    df['__assentos__'] = df['AERONAVE_TIPO'].apply(get_assentos)
    df = df[df['__assentos__'].notna()].copy()   # ignorar equipamentos não mapeados
    if df.empty:
        return []

    df['__min__'] = df[time_col].apply(parse_minutes)
    if 'PREVISTO_DATA' in df.columns:
        df['__data__'] = pd.to_datetime(df['PREVISTO_DATA'], errors='coerce').dt.date
    else:
        df['__data__'] = pd.Timestamp('today').date()

    mov_col_presente = 'MOVIMENTO_TIPO' in df.columns
    group_keys = ['__icao__', '__data__']
    if mov_col_presente:
        group_keys.insert(1, 'MOVIMENTO_TIPO')

    violations = []
    seen = set()

    for keys, grupo in df.groupby(group_keys):
        if mov_col_presente:
            icao, movimento, data = keys
        else:
            icao, data = keys
            movimento = '*'

        limite = get_limite(limites_dict, icao, movimento)
        grupo = grupo.dropna(subset=['__min__']).sort_values('__min__')
        mov_label = movimento if mov_col_presente else '—'

        for _, row in grupo.iterrows():
            start = int(row['__min__'])
            key = (icao, mov_label, str(data), start, 'seat')
            if key in seen:
                continue

            ops_janela = grupo[
                (grupo['__min__'] >= start) &
                (grupo['__min__'] < start + 60)
            ]
            total_assentos = ops_janela['__assentos__'].sum()

            if total_assentos > limite:
                seen.add(key)
                violations.append({
                    'icao':          icao,
                    'movimento':     mov_label,
                    'data':          data,
                    'inicio_min':    start,
                    'janela_label':  f"{minutes_to_hhmm(start)}–{minutes_to_hhmm(start + 60)}",
                    'total_assentos': int(total_assentos),
                    'limite':        limite,
                    'excesso':       int(total_assentos - limite),
                    'indices_ops':   ops_janela.index.tolist(),
                })

    return violations

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace; }

.hero {
    background: #0a0f1e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
}
.hero h1 { color: #38bdf8; font-size: 1.6rem; margin: 0 0 0.3rem; }
.hero p  { color: #7da8c4; margin: 0; font-size: 0.9rem; }

.kpi {
    background: #f8fafc;
    border-left: 4px solid #0ea5e9;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
}
.kpi small { display:block; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:.06em; }
.kpi b     { font-size:1.5rem; color:#0c4a6e; font-family:'IBM Plex Mono',monospace; }
.kpi-warn  { border-color:#f59e0b; }
.kpi-warn b{ color:#92400e; }
.kpi-ok    { border-color:#22c55e; }
.kpi-ok b  { color:#14532d; }

.tag-P { display:inline-block; padding:1px 8px; border-radius:20px;
          background:#dbeafe; color:#1e40af; font-size:.78rem;
          font-family:'IBM Plex Mono',monospace; font-weight:600; }
.tag-D { display:inline-block; padding:1px 8px; border-radius:20px;
          background:#fce7f3; color:#9d174d; font-size:.78rem;
          font-family:'IBM Plex Mono',monospace; font-weight:600; }

div[data-testid="stSidebar"] { background: #0a0f1e !important; }
div[data-testid="stSidebar"] * { color: #94c9e0 !important; }
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
</style>
"""

MOV_LABELS = {'P': 'Desembarque (P)', 'D': 'Embarque (D)', '*': 'Ambos'}

# ─────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RIMA Corretor", page_icon="✈️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>✈️ RIMA — Corretor de Hora Pico</h1>
        <p>Limites independentes por aeroporto (ICAO) e tipo de movimento (Embarque / Desembarque)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Limites por Aeroporto")
        st.caption("ICAO = 4 primeiros dígitos de COD_RIMA · Movimento: P = Desembarque, D = Embarque, * = Ambos")
        st.markdown("---")

        if 'limites_df' not in st.session_state:
            st.session_state.limites_df = pd.DataFrame([
                {'ICAO': 'SBUL', 'Movimento': 'P', 'Limite PAX/hora': 456},
                {'ICAO': 'SBUL', 'Movimento': 'D', 'Limite PAX/hora': 456},
                {'ICAO': 'SBUR', 'Movimento': 'P', 'Limite PAX/hora': 70},
                {'ICAO': 'SBUR', 'Movimento': 'D', 'Limite PAX/hora': 71},
                {'ICAO': 'SBMK', 'Movimento': 'P', 'Limite PAX/hora': 205},
                {'ICAO': 'SBMK', 'Movimento': 'D', 'Limite PAX/hora': 205},
                {'ICAO': 'SBCG', 'Movimento': 'P', 'Limite PAX/hora': 570},
                {'ICAO': 'SBCG', 'Movimento': 'D', 'Limite PAX/hora': 570},
                {'ICAO': 'SBPP', 'Movimento': 'P', 'Limite PAX/hora': 100},
                {'ICAO': 'SBPP', 'Movimento': 'D', 'Limite PAX/hora': 100},
                {'ICAO': 'SBCR', 'Movimento': 'P', 'Limite PAX/hora': 100},
                {'ICAO': 'SBCR', 'Movimento': 'D', 'Limite PAX/hora': 100},
                {'ICAO': 'SBHT', 'Movimento': 'P', 'Limite PAX/hora': 135},
                {'ICAO': 'SBHT', 'Movimento': 'D', 'Limite PAX/hora': 135},
                {'ICAO': 'SBCJ', 'Movimento': 'P', 'Limite PAX/hora': 170},
                {'ICAO': 'SBCJ', 'Movimento': 'D', 'Limite PAX/hora': 170},
                {'ICAO': 'SBSN', 'Movimento': 'P', 'Limite PAX/hora': 310},
                {'ICAO': 'SBSN', 'Movimento': 'D', 'Limite PAX/hora': 310},
                {'ICAO': 'SBMA', 'Movimento': 'P', 'Limite PAX/hora': 260},
                {'ICAO': 'SBMA', 'Movimento': 'D', 'Limite PAX/hora': 300},
            ])

        limites_editado = st.data_editor(
            st.session_state.limites_df,
            num_rows='dynamic',
            use_container_width=True,
            column_config={
                'ICAO': st.column_config.TextColumn('ICAO', max_chars=4),
                'Movimento': st.column_config.SelectboxColumn(
                    'Movimento',
                    options=['P', 'D', '*'],
                    help='P = Desembarque | D = Embarque | * = Ambos'
                ),
                'Limite PAX/hora': st.column_config.NumberColumn(
                    'Limite PAX/h', min_value=0, step=50
                ),
            },
            key='limites_editor'
        )
        st.session_state.limites_df = limites_editado

        limite_default = st.number_input(
            "Limite padrão (aeroportos / movimentos não listados)",
            min_value=100, max_value=99999, value=3000, step=100
        )

        st.markdown("---")
        time_col_option = st.radio(
            "Coluna de horário",
            ["PREVISTO_HORARIO", "CALCO_HORARIO", "Detectar automaticamente"]
        )

        st.markdown("---")
        if st.button("🔄 Reiniciar correções", use_container_width=True):
            for k in ['df_trabalho', 'correcoes']:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Montar dict de limites ───────────────────────────────────
    limites_dict = build_limites_dict(limites_editado, limite_default)

    # ── Upload ───────────────────────────────────────────────────
    uploaded = st.file_uploader("📂 Upload do arquivo RIMA (.csv)", type=["csv"])
    if uploaded is None:
        st.info("Faça upload de um arquivo CSV do RIMA para começar.")
        st.markdown("""
        **Colunas esperadas:**  
        `COD_RIMA` · `MOVIMENTO_TIPO` · `PAX_LOCAL` · `PAX_CONEXAO_DOMESTICO` · `PAX_CONEXAO_INTERNACIONAL`  
        · `PREVISTO_HORARIO` (ou `CALCO_HORARIO`) · `PREVISTO_DATA`
        """)
        return

    # ── Leitura ──────────────────────────────────────────────────
    try:
        # utf-8-sig remove o BOM (EF BB BF) gerado por Excel/Windows
        # sep=None com engine='python' detecta ; automaticamente
        df_raw = pd.read_csv(
            uploaded,
            sep=None,
            engine='python',
            dtype=str,
            encoding='utf-8-sig',
            skipinitialspace=True,
        )
        # Limpar BOM residual no nome da primeira coluna (fallback extra)
        df_raw.columns = [c.lstrip('\ufeff').strip() for c in df_raw.columns]
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return

    if 'COD_RIMA' not in df_raw.columns:
        st.error(
            f"Coluna **COD_RIMA** não encontrada. "
            f"Colunas detectadas: `{list(df_raw.columns)}`"
        )
        return

    df_raw['__icao__'] = df_raw['COD_RIMA'].apply(extract_icao)

    # Detectar coluna de horário
    if time_col_option == "Detectar automaticamente":
        time_col = next(
            (c for c in ['PREVISTO_HORARIO', 'CALCO_HORARIO', 'TOQUE_HORARIO'] if c in df_raw.columns),
            None
        )
        if not time_col:
            st.error("Nenhuma coluna de horário padrão encontrada (PREVISTO_HORARIO, CALCO_HORARIO, TOQUE_HORARIO).")
            return
        st.info(f"Coluna de horário detectada automaticamente: **{time_col}**")
    else:
        time_col = time_col_option
        if time_col not in df_raw.columns:
            st.error(f"Coluna **{time_col}** não encontrada. Disponíveis: {list(df_raw.columns)}")
            return

    pax_cols = get_pax_columns(df_raw)
    if not pax_cols:
        st.error("Nenhuma coluna PAX encontrada.")
        return

    # Verificar MOVIMENTO_TIPO
    if 'MOVIMENTO_TIPO' not in df_raw.columns:
        st.warning("Coluna **MOVIMENTO_TIPO** não encontrada — limites serão aplicados sem distinção de embarque/desembarque.")

    # ICAOs e movimentos detectados
    icaos_arquivo = sorted(df_raw['__icao__'].unique().tolist())
    movs_arquivo  = sorted(df_raw['MOVIMENTO_TIPO'].unique().tolist()) if 'MOVIMENTO_TIPO' in df_raw.columns else ['—']

    st.success(
        f"{len(df_raw):,} registros · ICAOs: **{', '.join(icaos_arquivo)}** · "
        f"Movimentos: **{', '.join(movs_arquivo)}**"
    )

    with st.expander("👀 Preview dos dados"):
        st.dataframe(df_raw.drop(columns=['__icao__']).head(10))

    # ── Diagnóstico ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Diagnóstico")

    violations = find_violations(df_raw, time_col, pax_cols, limites_dict)

    # Agrupar violações por (ICAO, Movimento) para mostrar resumo
    resumo = {}
    for v in violations:
        k = (v['icao'], v['movimento'])
        resumo.setdefault(k, {'janelas': 0, 'excesso': 0})
        resumo[k]['janelas'] += 1
        resumo[k]['excesso'] += v['excesso']

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi"><small>Registros analisados</small><b>{len(df_raw):,}</b></div>', unsafe_allow_html=True)
    with c2:
        cls = "kpi-warn" if violations else "kpi-ok"
        st.markdown(f'<div class="kpi {cls}"><small>Janelas violadas</small><b>{len(violations):,}</b></div>', unsafe_allow_html=True)
    with c3:
        total_exc = sum(v['excesso'] for v in violations)
        cls = "kpi-warn" if total_exc > 0 else "kpi-ok"
        st.markdown(f'<div class="kpi {cls}"><small>PAX excedente total</small><b>{total_exc:,}</b></div>', unsafe_allow_html=True)

    if resumo:
        st.markdown("**Resumo por aeroporto e movimento:**")
        resumo_rows = [
            {
                'ICAO': k[0],
                'Movimento': MOV_LABELS.get(k[1], k[1]),
                'Janelas violadas': v['janelas'],
                'PAX excedente': v['excesso'],
                'Limite aplicado': get_limite(limites_dict, k[0], k[1]),
            }
            for k, v in sorted(resumo.items())
        ]
        st.dataframe(pd.DataFrame(resumo_rows), use_container_width=True, hide_index=True)

    # ── Diagnóstico de Assentos ──────────────────────────────────
    seat_violations = find_seat_violations(df_raw, time_col, limites_dict)

    icaos_assentos_presentes = sorted(df_raw['__icao__'].unique().tolist())
    icaos_com_analise = [i for i in icaos_assentos_presentes if i in ICAOS_ASSENTOS]

    if icaos_com_analise:
        st.markdown("---")
        st.subheader("🪑 Diagnóstico de Assentos")
        st.caption(
            f"Aeroportos com análise de assentos neste arquivo: **{', '.join(icaos_com_analise)}** · "
            f"Equipamentos: {', '.join(f'{k}={v}' for k,v in ASSENTOS_EQUIPAMENTO.items())} · "
            f"Service types: {', '.join(sorted(SERVICE_TYPES_ASSENTOS))}"
        )

        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            ops_seat = df_raw[
                df_raw['__icao__'].isin(ICAOS_ASSENTOS) &
                (df_raw.get('SERVICE_TYPE', pd.Series(dtype=str))
                    .str.strip().str.upper().isin(SERVICE_TYPES_ASSENTOS)
                 if 'SERVICE_TYPE' in df_raw.columns else pd.Series([True]*len(df_raw)))
            ]
            st.markdown(f'<div class="kpi"><small>Operações analisadas (assentos)</small><b>{len(ops_seat):,}</b></div>', unsafe_allow_html=True)
        with cs2:
            cls = "kpi-warn" if seat_violations else "kpi-ok"
            st.markdown(f'<div class="kpi {cls}"><small>Janelas violadas (assentos)</small><b>{len(seat_violations):,}</b></div>', unsafe_allow_html=True)
        with cs3:
            total_exc_seat = sum(v['excesso'] for v in seat_violations)
            cls = "kpi-warn" if total_exc_seat > 0 else "kpi-ok"
            st.markdown(f'<div class="kpi {cls}"><small>Assentos excedentes total</small><b>{total_exc_seat:,}</b></div>', unsafe_allow_html=True)

        if seat_violations:
            resumo_seat = {}
            for v in seat_violations:
                k = (v['icao'], v['movimento'])
                resumo_seat.setdefault(k, {'janelas': 0, 'excesso': 0})
                resumo_seat[k]['janelas'] += 1
                resumo_seat[k]['excesso'] += v['excesso']
            resumo_seat_rows = [
                {
                    'ICAO': k[0],
                    'Movimento': MOV_LABELS.get(k[1], k[1]),
                    'Janelas violadas': v['janelas'],
                    'Assentos excedentes': v['excesso'],
                    'Limite aplicado': get_limite(limites_dict, k[0], k[1]),
                }
                for k, v in sorted(resumo_seat.items())
            ]
            with st.expander("📋 Ver resumo de violações de assentos"):
                st.dataframe(pd.DataFrame(resumo_seat_rows), use_container_width=True, hide_index=True)
                st.markdown("**Detalhe por janela:**")
                for sv in seat_violations:
                    mov_lbl = MOV_LABELS.get(sv['movimento'], sv['movimento'])
                    ops_sv = df_raw.loc[sv['indices_ops'],
                        [time_col, 'AERONAVE_TIPO', 'COD_RIMA'] +
                        (['SERVICE_TYPE'] if 'SERVICE_TYPE' in df_raw.columns else []) +
                        (['MOVIMENTO_TIPO'] if 'MOVIMENTO_TIPO' in df_raw.columns else [])
                    ].copy()
                    ops_sv['Assentos'] = ops_sv['AERONAVE_TIPO'].apply(get_assentos)
                    st.markdown(
                        f"**[{sv['icao']}] {mov_lbl} · {sv['data']} · {sv['janela_label']}** — "
                        f"{sv['total_assentos']:,} assentos (excesso: +{sv['excesso']:,} · limite: {sv['limite']:,})"
                    )
                    st.dataframe(ops_sv, use_container_width=True)
        else:
            st.markdown('<div style="background:#dcfce7;border-left:4px solid #22c55e;border-radius:6px;padding:.8rem 1rem">✅ Nenhuma violação de assentos detectada nos aeroportos analisados.</div>', unsafe_allow_html=True)

    if not violations:
        st.markdown('<br><div style="background:#dcfce7;border-left:4px solid #22c55e;border-radius:6px;padding:.8rem 1rem">✅ Nenhuma violação de PAX detectada — arquivo dentro dos limites em todos os aeroportos e movimentos.</div>', unsafe_allow_html=True)
        if not seat_violations:
            return

    # ── Colunas de horário disponíveis ───────────────────────────
    # Sempre trabalha com PREVISTO e CALCO se existirem; time_col é a de análise
    hor_cols_disponiveis = [c for c in ['PREVISTO_HORARIO', 'CALCO_HORARIO'] if c in df_raw.columns]

    if not violations:
        st.info("Nenhuma violação de PAX para corrigir. Apenas violações de assentos foram detectadas (revisar manualmente acima).")
        return

    # ── Correção manual por operação ─────────────────────────────
    st.markdown("---")
    st.subheader("🔧 Correção Manual por Operação — PAX")
    st.caption("Cada janela é analisada pelo seu próprio limite (ICAO + tipo de movimento). Escolha o que ajustar em cada operação.")

    if 'correcoes' not in st.session_state:
        st.session_state.correcoes = {}   # row_idx -> lista de dicts {coluna, valor, tipo_label}

    if 'df_trabalho' not in st.session_state:
        st.session_state.df_trabalho = df_raw.copy()

    df_work = st.session_state.df_trabalho

    # ── AÇÃO EM MASSA: ajustar PAX de todas as violações ─────────
    st.markdown("#### ⚡ Ação em massa")
    col_massa1, col_massa2 = st.columns([3, 1])
    with col_massa1:
        st.caption(
            "Aplica a sugestão de PAX_LOCAL para **todas** as operações excedentes de todas as violações. "
            "Cada operação recebe a redução mínima necessária para zerar o excesso da sua janela."
        )
    with col_massa2:
        aplicar_massa = st.button("📉 Aplicar PAX a todas as violações", use_container_width=True)

    if aplicar_massa:
        for viol in violations:
            excesso_v   = viol['excesso']
            ops_idx     = viol['indices_ops']
            ops_sub     = df_work.loc[ops_idx, pax_cols].copy()
            ops_sub     = ops_sub.apply(pd.to_numeric, errors='coerce').fillna(0)
            ops_sub['__total__'] = ops_sub.sum(axis=1)
            ops_sub     = ops_sub.sort_values('__total__', ascending=True)

            acum = 0
            for ridx, rrow in ops_sub.iterrows():
                if acum >= excesso_v:
                    break
                pax_local_atual = int(pd.to_numeric(df_work.at[ridx, 'PAX_LOCAL'], errors='coerce') or 0)
                restante        = max(0, excesso_v - acum)
                novo_pax        = max(0, pax_local_atual - restante)
                acum           += (pax_local_atual - novo_pax)
                corrs = st.session_state.correcoes.setdefault(ridx, [])
                # Substitui correção PAX anterior se existir
                corrs = [c for c in corrs if c['coluna'] != 'PAX_LOCAL']
                corrs.append({'coluna': 'PAX_LOCAL', 'valor': novo_pax, 'tipo_label': 'PAX'})
                st.session_state.correcoes[ridx] = corrs

        st.success(f"PAX sugerido aplicado a {len(st.session_state.correcoes)} operação(ões). Revise abaixo se necessário.")

    st.markdown("---")

    # ── Expanders por violação ────────────────────────────────────
    for v_idx, viol in enumerate(violations):
        icao      = viol['icao']
        movimento = viol['movimento']
        data      = viol['data']
        janela    = viol['janela_label']
        pax_v     = viol['pax_janela']
        limite_v  = viol['limite']
        excesso_v = viol['excesso']
        ops_idx   = viol['indices_ops']

        mov_tag_html = f'<span class="tag-{movimento}">{MOV_LABELS.get(movimento, movimento)}</span>'
        titulo = (
            f"⚠️  [{icao}]  {mov_tag_html}  {data}  ·  {janela}  ·  "
            f"{pax_v:,} PAX  ·  excesso: +{excesso_v:,}  (limite: {limite_v:,})"
        )

        with st.expander(titulo, expanded=(v_idx == 0)):
            # Preview com todas colunas relevantes
            cols_exibir = (
                hor_cols_disponiveis +
                pax_cols +
                (['MOVIMENTO_TIPO'] if 'MOVIMENTO_TIPO' in df_work.columns else []) +
                ['COD_RIMA']
            )
            ops_df = df_work.loc[ops_idx, cols_exibir].copy()
            ops_df[pax_cols] = ops_df[pax_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            ops_df['Total PAX'] = ops_df[pax_cols].sum(axis=1)

            st.markdown(f"**{len(ops_df)} operação(ões) · excesso de {excesso_v:,} PAX a redistribuir:**")
            st.dataframe(ops_df, use_container_width=True)

            acumulado_corrigido = 0
            for row_idx in ops_df.index:
                row    = df_work.loc[row_idx]
                pax_op = int(ops_df.at[row_idx, 'Total PAX'])
                mov_op = str(row.get('MOVIMENTO_TIPO', '—'))

                # Horários atuais de ambas as colunas
                hor_previsto = str(row.get('PREVISTO_HORARIO', '—')) if 'PREVISTO_HORARIO' in df_work.columns else '—'
                hor_calco    = str(row.get('CALCO_HORARIO',    '—')) if 'CALCO_HORARIO'    in df_work.columns else '—'
                hor_display  = f"Prev: `{hor_previsto}`"
                if 'CALCO_HORARIO' in df_work.columns:
                    hor_display += f"  ·  Calço: `{hor_calco}`"

                st.markdown(
                    f"---\n**Operação #{row_idx}** &nbsp;·&nbsp; `{row['COD_RIMA']}` &nbsp;·&nbsp; "
                    f"<span class='tag-{mov_op}'>{MOV_LABELS.get(mov_op, mov_op)}</span> &nbsp;·&nbsp; "
                    f"{hor_display} &nbsp;·&nbsp; PAX: `{pax_op:,}`",
                    unsafe_allow_html=True
                )

                # Opções dinâmicas de acordo com colunas presentes
                opcoes = ["Não alterar"]
                if 'PREVISTO_HORARIO' in df_work.columns:
                    opcoes.append("Ajustar PREVISTO_HORARIO")
                if 'CALCO_HORARIO' in df_work.columns:
                    opcoes.append("Ajustar CALCO_HORARIO")
                if 'PREVISTO_HORARIO' in df_work.columns and 'CALCO_HORARIO' in df_work.columns:
                    opcoes.append("Ajustar ambos os horários")
                opcoes.append("Ajustar PAX_LOCAL")

                # Pré-selecionar se já existe correção em massa
                corrs_atuais = st.session_state.correcoes.get(row_idx, [])
                colunas_com_corr = [c['coluna'] for c in corrs_atuais]
                if 'PAX_LOCAL' in colunas_com_corr:
                    idx_default = opcoes.index("Ajustar PAX_LOCAL")
                elif 'PREVISTO_HORARIO' in colunas_com_corr and 'CALCO_HORARIO' in colunas_com_corr:
                    idx_default = opcoes.index("Ajustar ambos os horários") if "Ajustar ambos os horários" in opcoes else 0
                elif 'PREVISTO_HORARIO' in colunas_com_corr:
                    idx_default = opcoes.index("Ajustar PREVISTO_HORARIO") if "Ajustar PREVISTO_HORARIO" in opcoes else 0
                elif 'CALCO_HORARIO' in colunas_com_corr:
                    idx_default = opcoes.index("Ajustar CALCO_HORARIO") if "Ajustar CALCO_HORARIO" in opcoes else 0
                else:
                    idx_default = 0

                tipo_corr = st.radio(
                    "O que ajustar?",
                    opcoes,
                    index=idx_default,
                    key=f"tipo_{v_idx}_{row_idx}",
                    horizontal=True
                )

                def horario_input(col_name, hor_atual_val, key_suffix):
                    """Renderiza input de horário com sugestão."""
                    min_atual = parse_minutes(hor_atual_val) or 0
                    sugestao  = minutes_to_hhmm(min(1439, min_atual + 60))
                    return st.text_input(
                        f"Novo {col_name} (HH:MM) — sugestão fora da janela: `{sugestao}`",
                        value=sugestao,
                        key=f"{key_suffix}_{v_idx}_{row_idx}",
                        placeholder="Ex: 15:30"
                    )

                novas_corrs = []

                if "Ajustar PREVISTO_HORARIO" in tipo_corr:
                    novo = horario_input('PREVISTO_HORARIO', hor_previsto, 'prev')
                    novas_corrs.append({'coluna': 'PREVISTO_HORARIO', 'valor': novo, 'tipo_label': 'Horário'})

                if "Ajustar CALCO_HORARIO" in tipo_corr:
                    novo = horario_input('CALCO_HORARIO', hor_calco, 'calco')
                    novas_corrs.append({'coluna': 'CALCO_HORARIO', 'valor': novo, 'tipo_label': 'Horário'})

                if tipo_corr == "Ajustar PAX_LOCAL":
                    pax_local_atual = int(pd.to_numeric(row.get('PAX_LOCAL', 0), errors='coerce') or 0)
                    restante        = max(0, excesso_v - acumulado_corrigido)
                    sugestao_pax    = max(0, pax_local_atual - restante)

                    # Pré-popular com valor da ação em massa se existir
                    massa_val = next((c['valor'] for c in corrs_atuais if c['coluna'] == 'PAX_LOCAL'), sugestao_pax)

                    novo_pax = st.number_input(
                        f"Novo PAX_LOCAL  (atual: {pax_local_atual:,}  ·  sugestão: {sugestao_pax:,})",
                        min_value=0, value=int(massa_val), step=1,
                        key=f"pax_{v_idx}_{row_idx}"
                    )
                    acumulado_corrigido += (pax_local_atual - novo_pax)
                    novas_corrs.append({'coluna': 'PAX_LOCAL', 'valor': novo_pax, 'tipo_label': 'PAX'})

                if novas_corrs:
                    st.session_state.correcoes[row_idx] = novas_corrs
                elif tipo_corr == "Não alterar":
                    st.session_state.correcoes.pop(row_idx, None)

    # ── Aplicar ──────────────────────────────────────────────────
    st.markdown("---")
    n_pend = sum(len(v) for v in st.session_state.correcoes.values())
    st.markdown(f"**{n_pend} alteração(ões) em {len(st.session_state.correcoes)} operação(ões) prontas para aplicar.**")

    if st.button("✅ Aplicar todas as correções e gerar arquivo", type="primary", use_container_width=True):
        if not st.session_state.correcoes:
            st.warning("Nenhuma correção configurada.")
            return

        df_final = df_work.copy()
        log = []

        for idx, corrs in st.session_state.correcoes.items():
            for corr in corrs:
                valor_ant = df_final.at[idx, corr['coluna']]
                df_final.at[idx, corr['coluna']] = corr['valor']
                log.append({
                    'Índice':         idx,
                    'COD_RIMA':       df_final.at[idx, 'COD_RIMA'],
                    'ICAO':           df_final.at[idx, '__icao__'],
                    'Movimento':      df_final.at[idx, 'MOVIMENTO_TIPO'] if 'MOVIMENTO_TIPO' in df_final.columns else '—',
                    'Coluna':         corr['coluna'],
                    'Valor Original': valor_ant,
                    'Valor Novo':     corr['valor'],
                    'Tipo':           corr['tipo_label'],
                })

        df_export = df_final.drop(columns=['__icao__'], errors='ignore')
        st.success(f"✅ {len(log)} alteração(ões) aplicada(s) em {len(st.session_state.correcoes)} operação(ões).")

        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📥 Baixar RIMA Corrigido (.csv)",
                data=df_export.to_csv(index=False, sep=';').encode('utf-8-sig'),
                file_name="RIMA_corrigido.csv", mime="text/csv",
                use_container_width=True
            )
        with col_b:
            st.download_button(
                "📋 Baixar Log de Alterações (.csv)",
                data=log_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="RIMA_log_correcoes.csv", mime="text/csv",
                use_container_width=True
            )

        # ── Verificação pós-correção ─────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Verificação Pós-Correção")

        df_pos = df_export.copy()
        df_pos['__icao__'] = df_pos['COD_RIMA'].apply(extract_icao)
        violations_pos = find_violations(df_pos, time_col, pax_cols, limites_dict)

        if violations_pos:
            st.warning(f"⚠️ Ainda existem {len(violations_pos)} janela(s) de PAX violada(s) após as correções.")
            for vp in violations_pos:
                mov_label = MOV_LABELS.get(vp['movimento'], vp['movimento'])
                st.markdown(
                    f"- `[{vp['icao']}]` **{mov_label}** · {vp['data']} · "
                    f"{vp['janela_label']} · {vp['pax_janela']:,} PAX "
                    f"(excesso: +{vp['excesso']:,} · limite: {vp['limite']:,})"
                )
        else:
            st.markdown(
                '<div style="background:#dcfce7;border-left:4px solid #22c55e;'
                'border-radius:6px;padding:.8rem 1rem">'
                '✅ Todas as janelas de PAX dentro do limite após as correções.</div>',
                unsafe_allow_html=True
            )

        # Verificação pós-correção: assentos
        seat_violations_pos = find_seat_violations(df_pos, time_col, limites_dict)
        icaos_com_analise_pos = [i for i in df_pos['__icao__'].unique() if i in ICAOS_ASSENTOS]
        if icaos_com_analise_pos:
            if seat_violations_pos:
                st.warning(f"⚠️ Ainda existem {len(seat_violations_pos)} janela(s) de **assentos** violada(s) após as correções.")
                for sv in seat_violations_pos:
                    mov_label = MOV_LABELS.get(sv['movimento'], sv['movimento'])
                    st.markdown(
                        f"- `[{sv['icao']}]` **{mov_label}** · {sv['data']} · "
                        f"{sv['janela_label']} · {sv['total_assentos']:,} assentos "
                        f"(excesso: +{sv['excesso']:,} · limite: {sv['limite']:,})"
                    )
            else:
                st.markdown(
                    '<div style="background:#dcfce7;border-left:4px solid #22c55e;'
                    'border-radius:6px;padding:.8rem 1rem">'
                    '✅ Todas as janelas de assentos dentro do limite após as correções.</div>',
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
