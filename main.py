import re

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
