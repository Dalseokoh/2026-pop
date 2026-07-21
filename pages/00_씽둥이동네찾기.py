import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="연령별 인구 구조 시각화", layout="wide")

DATA_FILE = "202606_202606_연령별인구현황_월간.csv"  # 코드와 같은 폴더에 위치해야 함


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """행정안전부 연령별 인구현황 CSV를 읽어 정제한다.

    - 원본 인코딩은 CP949
    - 파일 끝부분이 손상(잘림)되어 있을 수 있으므로 문제가 되는 행은 건너뜀
    - 숫자 컬럼의 천 단위 콤마(,)를 제거하고 숫자형으로 변환
    """
    df = pd.read_csv(path, encoding="cp949", engine="python", on_bad_lines="skip")

    # 행정구역 컬럼을 제외한 나머지는 전부 숫자(콤마 포함 문자열) → 숫자형 변환
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def parse_age_columns(columns: list[str]):
    """컬럼명에서 (연월, 성별, 연령) 구조를 파싱해 연령별 컬럼 매핑을 만든다.

    컬럼명 예: '2026년06월_계_0세', '2026년06월_남_100세 이상'
    """
    pattern = re.compile(r"^(\d{4}년\d{2}월)_(계|남|여)_(.+)$")

    year_month = None
    age_cols = {"계": {}, "남": {}, "여": {}}

    for col in columns:
        m = pattern.match(col)
        if not m:
            continue
        ym, gender, label = m.group(1), m.group(2), m.group(3)
        year_month = year_month or ym

        if label in ("총인구수", "연령구간인구수"):
            continue

        if label == "100세 이상":
            age_num = 100
        else:
            age_match = re.match(r"^(\d+)세$", label)
            if not age_match:
                continue
            age_num = int(age_match.group(1))

        age_cols[gender][age_num] = col

    return year_month, age_cols


def build_age_structure_df(row: pd.Series, age_cols: dict) -> pd.DataFrame:
    """선택한 지역 1개 행(row)으로부터 연령 x 성별 인구 데이터프레임을 만든다."""
    ages = sorted(age_cols["계"].keys())

    data = {"연령": ages}
    for gender in ("계", "남", "여"):
        data[gender] = [row[age_cols[gender][a]] for a in ages]

    return pd.DataFrame(data)


@st.cache_data
def build_structure_matrix(df: pd.DataFrame, age_cols: dict):
    """전체 지역 x 연령(계) 인구비율 행렬을 만든다.

    총인구가 0인 지역은 비교 대상에서 제외한다(구조 비교가 불가능하므로).
    반환값: (지역명 배열, 연령 리스트, 비율 행렬 prop, 총인구 배열)
    """
    ages = sorted(age_cols["계"].keys())
    cols_order = [age_cols["계"][a] for a in ages]

    mat = df[cols_order].to_numpy(dtype=float)
    totals = mat.sum(axis=1)

    valid = totals > 0
    regions = df.iloc[:, 0].to_numpy()[valid]
    mat = mat[valid]
    totals = totals[valid]

    prop = mat / totals[:, None]  # 지역별 연령 인구비율(구조)

    return regions, ages, prop, totals


def find_similar_regions(
    selected_region: str,
    regions: "np.ndarray",
    prop: "np.ndarray",
    top_n: int = 5,
):
    """코사인 유사도를 기준으로 인구 구조가 가장 비슷한 지역 top_n을 찾는다."""
    idx_arr = np.where(regions == selected_region)[0]
    if len(idx_arr) == 0:
        return pd.DataFrame(columns=["지역", "유사도"])

    idx = idx_arr[0]
    target = prop[idx]

    norms = np.linalg.norm(prop, axis=1)
    target_norm = np.linalg.norm(target)
    sims = (prop @ target) / (norms * target_norm + 1e-12)

    order = np.argsort(-sims)
    result = []
    for i in order:
        if regions[i] == selected_region:
            continue
        result.append((regions[i], sims[i]))
        if len(result) == top_n:
            break

    return pd.DataFrame(result, columns=["지역", "유사도"])


# ------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------
try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"'{DATA_FILE}' 파일을 찾을 수 없습니다. "
        "이 앱 코드(app.py)와 같은 폴더에 데이터 파일을 넣어주세요."
    )
    st.stop()

year_month, age_cols = parse_age_columns(df.columns.tolist())

# ------------------------------------------------------------
# 사이드바: 지역 선택 (검색 입력 + 드롭다운 선택)
# ------------------------------------------------------------
st.sidebar.header("지역 선택")

all_regions = df.iloc[:, 0].tolist()

search_text = st.sidebar.text_input(
    "지역명 검색 (예: 서울, 강남구, 수원시 등)", value=""
)

if search_text:
    filtered_regions = [r for r in all_regions if search_text in r]
else:
    filtered_regions = all_regions

if not filtered_regions:
    st.sidebar.warning("검색 결과가 없습니다. 다른 키워드로 검색해보세요.")
    st.stop()

selected_region = st.sidebar.selectbox(
    "지역 선택", filtered_regions, index=0
)

# ------------------------------------------------------------
# 메인 화면
# ------------------------------------------------------------
st.title("연령별 인구 구조 시각화")
if year_month:
    st.caption(f"기준 연월: {year_month}")

row = df[df.iloc[:, 0] == selected_region].iloc[0]
structure_df = build_age_structure_df(row, age_cols)

total_pop = int(row[age_cols["계"][0]] * 0 + structure_df["계"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("선택 지역", selected_region)
col2.metric("총인구수(연령 합계)", f"{int(structure_df['계'].sum()):,}")
col3.metric("남/여 비율", f"{structure_df['남'].sum() / max(structure_df['여'].sum(),1):.2f}")

# ------------------------------------------------------------
# Plotly 꺾은선 그래프
# ------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=structure_df["연령"],
        y=structure_df["계"],
        mode="lines",
        name="계(전체)",
        line=dict(width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=structure_df["연령"],
        y=structure_df["남"],
        mode="lines",
        name="남",
        line=dict(width=2, dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=structure_df["연령"],
        y=structure_df["여"],
        mode="lines",
        name="여",
        line=dict(width=2, dash="dot"),
    )
)

fig.update_layout(
    title=f"{selected_region} 연령별 인구 구조",
    xaxis_title="연령(세, 100=100세 이상)",
    yaxis_title="인구수(명)",
    hovermode="x unified",
    legend_title="구분",
    height=550,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("원본 데이터 보기"):
    st.dataframe(structure_df, use_container_width=True)

# ------------------------------------------------------------
# 인구 구조가 가장 비슷한 지역 Top 5
# ------------------------------------------------------------
st.divider()
st.header("전국에서 인구 구조가 가장 비슷한 지역 Top 5")
st.caption(
    "연령별 인구 '비율(구조)'을 기준으로 코사인 유사도를 계산합니다. "
    "총인구 규모가 달라도 연령대별 분포 모양이 비슷하면 높은 유사도로 나타납니다. "
    "(읍·면·동 단위부터 시·군·구, 시·도 단위까지 전체 지역이 비교 대상에 포함됩니다.)"
)

regions_all, ages_all, prop_all, totals_all = build_structure_matrix(df, age_cols)

similar_df = find_similar_regions(selected_region, regions_all, prop_all, top_n=5)

if similar_df.empty:
    st.warning(
        "선택한 지역은 총인구가 0이거나 데이터가 없어 유사 지역을 계산할 수 없습니다."
    )
else:
    similar_df_display = similar_df.copy()
    similar_df_display["유사도"] = similar_df_display["유사도"].round(4)
    st.dataframe(similar_df_display, use_container_width=True, hide_index=True)

    # 선택 지역 + top5 지역의 연령별 인구 '비율(%)'을 함께 시각화
    fig_sim = go.Figure()

    selected_idx = np.where(regions_all == selected_region)[0][0]
    selected_prop_pct = prop_all[selected_idx] * 100

    fig_sim.add_trace(
        go.Scatter(
            x=ages_all,
            y=selected_prop_pct,
            mode="lines",
            name=f"{selected_region} (선택)",
            line=dict(width=4, color="black"),
        )
    )

    for _, r in similar_df.iterrows():
        region_name = r["지역"]
        sim_score = r["유사도"]
        region_idx = np.where(regions_all == region_name)[0][0]
        region_prop_pct = prop_all[region_idx] * 100

        fig_sim.add_trace(
            go.Scatter(
                x=ages_all,
                y=region_prop_pct,
                mode="lines",
                name=f"{region_name} (유사도 {sim_score:.3f})",
                line=dict(width=2),
            )
        )

    fig_sim.update_layout(
        title="선택 지역 vs 인구 구조 유사 지역 Top 5 (연령별 인구 비율 비교)",
        xaxis_title="연령(세, 100=100세 이상)",
        yaxis_title="해당 연령 인구 비율(%)",
        hovermode="x unified",
        legend_title="지역",
        height=600,
    )

    st.plotly_chart(fig_sim, use_container_width=True)
