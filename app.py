"""
零碳比赛 - 智能控碳赛道
中国宏观能源结构与碳达峰情景模拟 交互式数据看板
数据来源：CEADs 中国碳核算数据库（中国国家表观碳排放清单 1997-2022）

开发框架：Streamlit + Plotly + Scikit-learn
"""

# ==================== 依赖检查 ====================
# 在代码开头检查所有必需的第三方库是否已安装
import importlib
import sys

required_packages = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
}

missing_packages = []
for module_name, pip_name in required_packages.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print(f"❌ 缺少以下依赖包：{missing_packages}")
    print(f"请运行：pip install {' '.join(missing_packages)}")
    sys.exit(1)

# 正式导入所有需要的库
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io

# ======================== 第一轮：数据处理与基础看板 ========================

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="零碳比赛 · 碳达峰情景模拟",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- 标题与说明 --------------------
st.title("🌍 中国宏观能源结构与碳达峰情景模拟")
st.markdown(
    "**赛道**：智能控碳 · 零碳比赛 &nbsp;&nbsp;|&nbsp;&nbsp;"
    "**数据来源**：CEADs（中国碳核算数据库）—— 中国国家表观碳排放清单（1997-2022）"
)
st.markdown("---")

# -------------------- 会话状态初始化 --------------------
# 用于"应用最优策略"按钮动态修改滑块值
if "slider_coal" not in st.session_state:
    st.session_state.slider_coal = 0.0
if "slider_clean" not in st.session_state:
    st.session_state.slider_clean = 1.0
if "slider_intensity" not in st.session_state:
    st.session_state.slider_intensity = 2.0
if "optimal_triggered" not in st.session_state:
    st.session_state.optimal_triggered = False

# 关键：在滑块控件渲染之前检查触发标志，提前修改 session_state
# 这样滑块控件初始化时会自动读取已修改的值
OPT_COAL = -2.0
OPT_CLEAN = 3.0
OPT_INTENSITY = 3.0

if st.session_state.optimal_triggered:
    st.session_state.slider_coal = OPT_COAL
    st.session_state.slider_clean = OPT_CLEAN
    st.session_state.slider_intensity = OPT_INTENSITY
    st.session_state.optimal_triggered = False  # 重置标志，避免后续运行也被覆盖

# -------------------- 数据读取与解析 --------------------
@st.cache_data  # Streamlit 缓存装饰器，避免每次交互都重新读取文件
def load_data():
    """
    读取 CEADs Excel 文件，解析 'emissions' Sheet。

    原始数据结构：
      - 第 A 列（Unnamed: 0）：分类标签（如 Fossil fuel, Process）
      - 第 B 列（Items）：指标名称（如 Raw coal total, Crude oil total）
      - 第 C ~ AB 列：1997-2022 各年数据

    处理逻辑：
      1. 将 Items 列设为索引，丢弃分类列
      2. 转置使得"年份"变索引、"指标"变列名
      3. 提取我们关心的 5 个核心指标
      4. 计算能源结构占比

    返回：
      - df_emissions: 以年份为索引的排放数据 DataFrame
      - df_share: 以年份为索引的能源结构占比 DataFrame（%）
    """
    file_path = r"CEADs_China_CO2_1997-2022.xlsx"

    # 读取 emissions Sheet
    # CEADs 官方 Excel 为标准英文内容，openpyxl 引擎可直接读取
    df_raw = pd.read_excel(file_path, sheet_name="emissions", engine="openpyxl")

    # 原始数据包含 32 行（含分组标题行和空行），我们只关心其中 5 行关键指标：
    #   行0: Fossil fuel / Raw coal total      → 原煤排放
    #   行1: NaN          / Crude oil total     → 原油排放
    #   行2: NaN          / Natural gas total   → 天然气排放
    #   行3: Process      / Cement              → 水泥（工业过程）排放
    #   行4: Total apparent CO2 emissions (mt)  → 总排放
    #   行5-31: 各能源品种的细分项（生产、进口、出口等），本次分析不需要

    # 关键修正：Excel 中 Items 列的值带单引号（如 'Raw coal total'），
    # 且 "Total apparent CO2 emissions (mt)" 出现在 Unnamed:0 列而非 Items 列。
    # 因此需要：
    #   1. 合并两列：Items 列优先，为空时取 Unnamed:0 列的值
    #   2. 去除值两边的单引号
    df_raw["指标名称"] = df_raw["Items"].fillna(df_raw["Unnamed: 0"])
    df_raw["指标名称"] = df_raw["指标名称"].str.strip("'\"")  # 去除首尾的单/双引号

    # 将合并后的"指标名称"列设为行索引
    df_raw.set_index("指标名称", inplace=True)

    # 删除不需要的列：原始的前两列和年份之外的非数据列
    cols_to_keep = [c for c in df_raw.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())]
    # 实际上直接丢弃 Unnamed:0 和 Items 两列，保留年份列
    df_raw.drop(columns=["Unnamed: 0", "Items"], inplace=True, errors="ignore")

    # 转置：行变列、列变行 → 年份为索引，指标名为列名
    df = df_raw.transpose()
    df.index.name = "年份"
    df.index = df.index.astype(int)  # 年份转为整数

    # 定义需要的 5 个核心指标
    required_items = [
        "Raw coal total",                    # 原煤总排放
        "Crude oil total",                   # 原油总排放
        "Natural gas total",                 # 天然气总排放
        "Cement",                            # 水泥工业过程排放
        "Total apparent CO2 emissions (mt)", # 表观 CO₂ 总排放
    ]

    # 健壮性检查：确认所有需要的指标都存在于数据中
    missing_items = [item for item in required_items if item not in df.columns]
    if missing_items:
        st.error(f"❌ 数据文件中缺少以下指标列：{missing_items}\n请检查 Excel 文件内容。")
        st.stop()

    # 提取需要的列并去除全空行
    df_emissions = df[required_items].copy()
    df_emissions.dropna(how="all", inplace=True)
    df_emissions = df_emissions.astype(float)

    # -------------------- 计算能源结构占比 --------------------
    # 能源结构占比 = 各能源品种排放 / 总排放 × 100%
    # 注意：水泥是工业过程排放，不属于能源燃烧，因此不纳入能源结构计算
    total_col = "Total apparent CO2 emissions (mt)"
    coal_col = "Raw coal total"
    oil_col = "Crude oil total"
    gas_col = "Natural gas total"

    df_share = pd.DataFrame(index=df_emissions.index)
    df_share["煤炭占比(%)"] = (df_emissions[coal_col] / df_emissions[total_col]) * 100
    df_share["石油占比(%)"] = (df_emissions[oil_col] / df_emissions[total_col]) * 100
    df_share["天然气占比(%)"] = (df_emissions[gas_col] / df_emissions[total_col]) * 100

    return df_emissions, df_share


# -------------------- 加载数据 --------------------
df_emissions, df_share = load_data()

# 提取常用的数据摘要，方便后续模块使用
LAST_HISTORICAL_YEAR = df_emissions.index[-1]   # 历史数据最后一年（2022）
TOTAL_COL = "Total apparent CO2 emissions (mt)"
COAL_COL = "Raw coal total"
OIL_COL = "Crude oil total"
GAS_COL = "Natural gas total"
CEMENT_COL = "Cement"

# -------------------- 侧边栏：数据来源说明 --------------------
st.sidebar.header("📖 数据来源与说明")
st.sidebar.markdown("""
**中国碳核算数据库 (CEADs)**

本看板使用的碳排放清单数据来自 CEADs 团队
公开发布的《中国国家表观碳排放清单（1997-2022）》。

> **学术引用**：
> Shan, Y., Guan, D., Zheng, H. et al.
> *China CO₂ emission accounts 1997–2015*.
> *Scientific Data* 5, 170201 (2018).
> [CEADs 官网](https://www.ceads.net.cn/)

---
**赛道**：智能控碳 · 零碳比赛  
**框架**：Streamlit + Plotly + Scikit-learn
""")

# -------------------- 数据预览区（折叠展示） --------------------
col_preview1, col_preview2 = st.columns(2)

with col_preview1:
    with st.expander("📊 查看原始排放数据（单位：百万吨 CO₂）", expanded=False):
        st.dataframe(
            df_emissions.style.format("{:.2f}"),
            use_container_width=True
        )

with col_preview2:
    with st.expander("📈 查看能源结构占比数据（%）", expanded=False):
        st.dataframe(
            df_share.style.format("{:.2f}"),
            use_container_width=True
        )

st.success("✅ 第一轮完成：数据读取、解析、能源结构占比计算及基础页面搭建。")
st.markdown("---")

# ======================== 第二轮：历史趋势可视化 ========================

st.header("📈 历史碳排放趋势（1997-2022）")

# -------------------- 图表 1：堆叠面积图 --------------------
st.subheader("▎碳排放来源堆叠面积图")

# 堆叠面积图：展示煤炭、石油、天然气、水泥四类排放源随年份的演变
# 每层代表一种排放源，堆叠后总高度 ≈ 总排放量
fig_area = go.Figure()

# 定义四类排放源及其颜色（使用专业配色）
emission_sources = [
    (COAL_COL,   "原煤",     "#E74C3C"),   # 红色系——煤炭
    (OIL_COL,    "原油",     "#F39C12"),   # 橙色系——石油
    (GAS_COL,    "天然气",   "#2ECC71"),   # 绿色系——天然气
    (CEMENT_COL, "水泥",     "#9B59B6"),   # 紫色系——水泥
]

for col, label, color in emission_sources:
    fig_area.add_trace(go.Scatter(
        x=df_emissions.index,
        y=df_emissions[col],
        name=label,
        mode="lines",
        stackgroup="one",        # 堆叠模式
        line=dict(width=0.5, color=color),
        fillcolor=color,
        hovertemplate=f"{label}: %{{y:,.0f}} Mt<extra></extra>",
    ))

# 添加总排放折线（叠加在堆叠图上方，便于对比）
fig_area.add_trace(go.Scatter(
    x=df_emissions.index,
    y=df_emissions[TOTAL_COL],
    name="总排放",
    mode="lines+markers",
    line=dict(color="black", width=2, dash="dot"),
    marker=dict(size=5),
    hovertemplate="总排放: %{y:,.0f} Mt<extra></extra>",
))

fig_area.update_layout(
    title="中国碳排放来源构成（1997-2022）",
    xaxis=dict(title="年份", dtick=2),
    yaxis=dict(title="碳排放量（百万吨 CO₂）"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig_area, use_container_width=True)

# -------------------- 图表 2：百分比堆积柱状图 --------------------
st.subheader("▎能源结构演变 — 百分比堆积柱状图")

# 展示煤炭、石油、天然气三者占总排放的比例随时间演变
# 使用百分比堆叠柱状图，总高度始终为 100%
fig_bar = go.Figure()

energy_sources = [
    ("煤炭占比(%)", "煤炭", "#E74C3C"),
    ("石油占比(%)", "石油", "#F39C12"),
    ("天然气占比(%)", "天然气", "#2ECC71"),
]

for col, label, color in energy_sources:
    fig_bar.add_trace(go.Bar(
        x=df_share.index,
        y=df_share[col],
        name=label,
        marker_color=color,
        hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
    ))

fig_bar.update_layout(
    title="中国能源结构演变 — 煤炭/石油/天然气占比（1997-2022）",
    xaxis=dict(title="年份", dtick=2),
    yaxis=dict(title="占比（%）", range=[0, 100]),
    barmode="stack",           # 百分比堆叠
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig_bar, use_container_width=True)

# 简要解读
col_left, col_right = st.columns(2)
with col_left:
    st.metric("2022 年总排放", f"{df_emissions.loc[2022, TOTAL_COL]:,.0f} Mt")
    st.metric("2022 年煤炭占比", f"{df_share.loc[2022, '煤炭占比(%)']:.1f}%")
with col_right:
    st.metric("较 1997 年总排放增长", 
              f"{((df_emissions.loc[2022, TOTAL_COL] / df_emissions.loc[1997, TOTAL_COL]) - 1) * 100:.1f}%")
    st.metric("较 1997 年煤炭占比变化",
              f"{df_share.loc[2022, '煤炭占比(%)'] - df_share.loc[1997, '煤炭占比(%)']:+.1f} 个百分点")

st.success("✅ 第二轮完成：堆叠面积图 + 百分比堆积柱状图，清晰展示历史排放趋势与能源结构演变。")
st.markdown("---")

# ======================== 第三轮：未来趋势预测模型 ========================

st.header("🔮 未来碳排放趋势预测（2023-2030）")

# -------------------- 趋势预测函数 --------------------
@st.cache_data
def train_and_predict(df_emissions):
    """
    基于历史总排放数据，使用线性回归拟合趋势，预测 2023-2030 年碳排放。

    思路：
      1. 将年份作为自变量 X（数值），总排放作为因变量 y
      2. 使用 sklearn LinearRegression 拟合历史趋势
      3. 对 2023-2030 各年进行预测
      4. 同时对各分项（煤/油/气/水泥）分别拟合，得到分项预测值

    返回：
      - future_years: 预测年份列表 [2023, ..., 2030]
      - pred_total: 预测的总排放值
      - pred_detail: DataFrame，包含各分项的预测值
      - model_score: 模型 R² 分数（反映拟合优度）
    """
    years = df_emissions.index.values.reshape(-1, 1)
    total_vals = df_emissions[TOTAL_COL].values
    future_years = np.arange(2023, 2031).reshape(-1, 1)

    # --- 模型 1：总排放趋势 ---
    model_total = LinearRegression()
    model_total.fit(years, total_vals)
    pred_total = model_total.predict(future_years)
    r2_total = model_total.score(years, total_vals)

    # --- 模型 2-5：各分项分别拟合 ---
    pred_detail = pd.DataFrame(index=np.arange(2023, 2031))
    components = [COAL_COL, OIL_COL, GAS_COL, CEMENT_COL]
    for col in components:
        model = LinearRegression()
        model.fit(years, df_emissions[col].values)
        pred_detail[col] = model.predict(future_years)
        # 确保预测值非负（碳排放不可能为负）
        pred_detail[col] = pred_detail[col].clip(lower=0)

    # 也确保总排放预测值非负
    pred_total = np.clip(pred_total, 0, None)

    return (np.arange(2023, 2031), pred_total, pred_detail, r2_total, model_total)


# 执行预测
future_years, pred_total, pred_detail, r2_score, model_total = train_and_predict(df_emissions)

# -------------------- 趋势折线图：历史 + 预测 --------------------
st.subheader("▎总排放趋势：历史数据 + 线性回归预测")

fig_trend = go.Figure()

# 历史总排放（实线）
fig_trend.add_trace(go.Scatter(
    x=df_emissions.index,
    y=df_emissions[TOTAL_COL],
    name="历史总排放",
    mode="lines+markers",
    line=dict(color="#2980B9", width=3),
    marker=dict(size=8, color="#2980B9"),
    hovertemplate="历史 %{x}: %{y:,.0f} Mt<extra></extra>",
))

# 预测总排放（虚线）
fig_trend.add_trace(go.Scatter(
    x=future_years,
    y=pred_total,
    name="预测总排放（基准趋势）",
    mode="lines+markers",
    line=dict(color="#E74C3C", width=3, dash="dash"),
    marker=dict(size=8, symbol="diamond", color="#E74C3C"),
    hovertemplate="预测 %{x}: %{y:,.0f} Mt<extra></extra>",
))

# 在历史与预测交界处添加分隔线标注
fig_trend.add_vline(
    x=2022.5, line_dash="dot", line_color="gray",
    annotation_text="← 历史 | 预测 →",
    annotation_position="top",
    annotation_font=dict(size=12, color="gray"),
)

fig_trend.update_layout(
    title="中国碳排放总量趋势及线性回归预测（1997-2030）",
    xaxis=dict(title="年份", dtick=2),
    yaxis=dict(title="碳排放量（百万吨 CO₂）"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig_trend, use_container_width=True)

# -------------------- 分项预测详情 --------------------
st.subheader("▎各排放源预测值（2023-2030）")

# 构建完整的预测 DataFrame 用于展示
df_pred_display = pred_detail.copy()
df_pred_display[TOTAL_COL] = pred_total
# 重命名列为中文
df_pred_display.columns = ["原煤", "原油", "天然气", "水泥", "总排放"]

# 格式化显示
st.dataframe(
    df_pred_display.style.format("{:,.0f}"),
    use_container_width=True,
)

# 关键预测指标
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric(
        "2030 年预测总排放",
        f"{pred_total[-1]:,.0f} Mt",
        delta=f"{pred_total[-1] - df_emissions.loc[2022, TOTAL_COL]:+,.0f} Mt vs 2022"
    )
with col_p2:
    avg_growth = (pred_total[-1] / df_emissions.loc[2022, TOTAL_COL]) ** (1/8) - 1
    st.metric("年均预测增速", f"{avg_growth * 100:.2f}%")
with col_p3:
    st.metric("模型拟合优度 R²", f"{r2_score:.4f}",
              help="R² 越接近 1 表示线性趋势拟合越好。本模型使用简单线性回归，仅反映历史平均增长趋势。")

st.info(
    "💡 **说明**：上述预测基于 1997-2022 年历史数据的简单线性回归趋势外推。"
    "实际碳排放受政策、技术、经济等多重因素影响，此处仅作为基准参考情景。"
    "下方第四轮将通过滑块交互，模拟不同政策力度下的优化路径。"
)

st.success("✅ 第三轮完成：基于 sklearn 线性回归的 2023-2030 碳排放趋势预测。")
st.markdown("---")

# ======================== 第四轮：人机协同优化层 ========================

st.header("🎛️ 人机协同优化 — 政策情景模拟")

st.markdown("""
调整下方滑块，模拟不同政策力度对 **2023-2030 年** 碳排放路径的影响。
所有调整基于 2022 年实际数据，逐年递推计算。
""")

# -------------------- 滑块区域：置于页面顶部，横排三列 --------------------
slider_col1, slider_col2, slider_col3 = st.columns(3)

with slider_col1:
    st.markdown("##### 🏭 煤炭压减系数")
    coal_adj = st.slider(
        "调整煤炭排放年均增速（百分点）",
        min_value=-3.0, max_value=2.0, step=0.1,
        help="负值表示压减煤炭消费增速，正值表示放松约束。\n\n"
             "例如 -2% 表示在基准增速基础上每年额外降低 2 个百分点。",
        key="slider_coal"
    )

with slider_col2:
    st.markdown("##### 🌱 清洁能源替代系数")
    clean_adj = st.slider(
        "提升天然气/清洁能源增速（百分点）",
        min_value=0.0, max_value=5.0, step=0.1,
        help="正值表示加速清洁能源（天然气等）对煤炭的替代。\n\n"
             "例如 3% 表示天然气增速额外提升 3 个百分点。",
        key="slider_clean"
    )

with slider_col3:
    st.markdown("##### ⚡ 能耗强度下降系数")
    intensity_adj = st.slider(
        "年度单位能耗下降幅度（%）",
        min_value=0.0, max_value=5.0, step=0.1,
        help="反映技术进步和能效提升带来的能耗强度下降。\n\n"
             "例如 2% 表示每年单位 GDP 能耗下降 2%。",
        key="slider_intensity"
    )

# -------------------- 动态重算函数 --------------------
def compute_adjusted_prediction(df_emissions, coal_adj, clean_adj, intensity_adj):
    """
    基于滑块参数，重新计算 2023-2030 年的碳排放预测。

    核心逻辑：
      1. 获取各排放源历史线性回归的斜率（年均增量）
      2. 以 2022 年实际值为起点
      3. 逐年应用滑块的调整系数：
         - 煤炭：基准增速 + 煤炭压减系数（可为负），再受能耗强度下降影响
         - 石油：基准增速，受能耗强度下降影响
         - 天然气：基准增速 + 清洁替代系数，受能耗强度下降影响
         - 水泥：保持基准增速不变（工业过程排放，受政策影响较小）
      4. 总排放 = 各分项之和

    参数：
      - coal_adj: 煤炭压减系数（百分点，-3 到 +2）
      - clean_adj: 清洁能源替代系数（百分点，0 到 5）
      - intensity_adj: 能耗强度下降系数（%，0 到 5）

    返回：
      - df_result: DataFrame，索引为年份，列为各排放源 + 总排放
    """
    years_hist = df_emissions.index.values.reshape(-1, 1)
    future_years = np.arange(2023, 2031)

    # 计算各排放源的基线斜率（年均增量 Mt/年）和 2022 年基准值
    slopes = {}
    base_2022 = {}
    components = [COAL_COL, OIL_COL, GAS_COL, CEMENT_COL]

    for col in components:
        m = LinearRegression()
        m.fit(years_hist, df_emissions[col].values)
        slopes[col] = m.coef_[0]   # 年均增量 (Mt/年)
        base_2022[col] = df_emissions.loc[2022, col]

    # 构建结果 DataFrame
    df_result = pd.DataFrame(index=future_years)

    # 逐年递推计算
    for i, year in enumerate(future_years):
        t = i + 1  # 第 t 年（2023 为 t=1，2030 为 t=8）

        # --- 煤炭：基准增量 + 压减系数调整 ---
        # 压减系数（百分点）转换为增长率调整：coal_adj/100
        # 能耗强度下降使增速进一步放缓
        coal_slope_adj = slopes[COAL_COL] * (1 + coal_adj / 100) * (1 - intensity_adj / 100)
        coal_val = base_2022[COAL_COL] + coal_slope_adj * t

        # --- 石油：仅受能耗强度下降影响 ---
        oil_slope_adj = slopes[OIL_COL] * (1 - intensity_adj / 100)
        oil_val = base_2022[OIL_COL] + oil_slope_adj * t

        # --- 天然气：基准增量 + 清洁替代提升，再受能耗强度影响 ---
        gas_slope_adj = slopes[GAS_COL] * (1 + clean_adj / 100) * (1 - intensity_adj / 100)
        gas_val = base_2022[GAS_COL] + gas_slope_adj * t

        # --- 水泥：保持基准增速（工业过程排放） ---
        cement_val = base_2022[CEMENT_COL] + slopes[CEMENT_COL] * t

        # 确保非负
        df_result.loc[year, "原煤"] = max(0, coal_val)
        df_result.loc[year, "原油"] = max(0, oil_val)
        df_result.loc[year, "天然气"] = max(0, gas_val)
        df_result.loc[year, "水泥"] = max(0, cement_val)
        df_result.loc[year, "总排放"] = (
            df_result.loc[year, "原煤"]
            + df_result.loc[year, "原油"]
            + df_result.loc[year, "天然气"]
            + df_result.loc[year, "水泥"]
        )

    return df_result.astype(float)


# 计算当前滑块参数下的优化预测
df_optimized = compute_adjusted_prediction(df_emissions, coal_adj, clean_adj, intensity_adj)

# -------------------- 对比图：基准预测 vs 优化预测 --------------------
st.subheader("▎情景对比：基准趋势 vs 优化路径")

fig_compare = go.Figure()

# 历史总排放
fig_compare.add_trace(go.Scatter(
    x=df_emissions.index,
    y=df_emissions[TOTAL_COL],
    name="历史总排放",
    mode="lines",
    line=dict(color="#2980B9", width=3),
    hovertemplate="历史 %{x}: %{y:,.0f} Mt<extra></extra>",
))

# 基准预测（虚线）
fig_compare.add_trace(go.Scatter(
    x=future_years,
    y=pred_total,
    name="基准预测（无干预）",
    mode="lines+markers",
    line=dict(color="#95A5A6", width=2, dash="dash"),
    marker=dict(size=6, symbol="circle", color="#95A5A6"),
    hovertemplate="基准 %{x}: %{y:,.0f} Mt<extra></extra>",
))

# 优化预测（根据当前滑块值）
fig_compare.add_trace(go.Scatter(
    x=df_optimized.index,
    y=df_optimized["总排放"],
    name="优化路径（当前参数）",
    mode="lines+markers",
    line=dict(color="#27AE60", width=3),
    marker=dict(size=8, symbol="diamond", color="#27AE60"),
    hovertemplate="优化 %{x}: %{y:,.0f} Mt<extra></extra>",
))

# 添加 120 亿吨（12,000 Mt）参考线
fig_compare.add_hline(
    y=12000, line_dash="dot", line_color="#E67E22",
    annotation_text="120 亿吨达峰参考线",
    annotation_position="bottom right",
    annotation_font=dict(size=11, color="#E67E22"),
)

fig_compare.add_vline(
    x=2022.5, line_dash="dot", line_color="gray",
    annotation_text="← 历史 | 预测 →",
    annotation_position="top",
)

fig_compare.update_layout(
    title=f"碳排放路径对比（当前参数：煤炭{coal_adj:+.1f}% / 清洁+{clean_adj:.1f}% / 能耗-{intensity_adj:.1f}%）",
    xaxis=dict(title="年份", dtick=2, range=[1997, 2030]),
    yaxis=dict(title="碳排放量（百万吨 CO₂）"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig_compare, use_container_width=True)

# -------------------- 优化效果指标卡 --------------------
st.subheader("▎优化效果速览")

opt_2030 = df_optimized.loc[2030, "总排放"]
baseline_2030 = pred_total[-1]
reduction = baseline_2030 - opt_2030

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("2030 基准预测", f"{baseline_2030:,.0f} Mt")
with m2:
    st.metric("2030 优化预测", f"{opt_2030:,.0f} Mt",
              delta=f"{reduction:+,.0f} Mt")
with m3:
    st.metric("预估累计减碳", f"{reduction * 8 / 2:,.0f} Mt",
              help="粗略估算：8 年累计减碳量 ≈ 年均减排量 × 8 年（梯形近似）")
with m4:
    coal_share_2030 = df_optimized.loc[2030, "原煤"] / opt_2030 * 100
    st.metric("2030 优化后煤炭占比", f"{coal_share_2030:.1f}%",
              delta=f"{coal_share_2030 - df_share.loc[2022, '煤炭占比(%)']:+.1f} pp vs 2022")

# 优化后的分项数据表格
with st.expander("📊 查看优化后逐年分项预测数据", expanded=False):
    st.dataframe(
        df_optimized.style.format("{:,.0f}"),
        use_container_width=True,
    )

st.success(
    f"✅ 第四轮完成：3 个交互滑块已就位，当前参数下 2030 年预计减排 {reduction:,.0f} Mt。"
    f"实时调整滑块即可看到曲线动态变化。"
)
st.markdown("---")

# ======================== 第五轮：预警与辅助决策 ========================

st.header("🚨 智能预警与辅助决策")

# -------------------- 智能预警灯 --------------------
# 设定阈值：2030 年碳达峰目标 ≤ 120 亿吨 CO₂（即 12,000 Mt）
DAPEK_THRESHOLD = 12000  # Mt

# 获取当前路径和历史数据的关键指标
current_2030 = df_optimized.loc[2030, "总排放"]
historical_peak = df_emissions[TOTAL_COL].max()
historical_peak_year = df_emissions[TOTAL_COL].idxmax()

# 判断预警等级
if current_2030 > DAPEK_THRESHOLD or current_2030 > historical_peak:
    # 红色预警
    st.error(
        f"### ⚠️ 预警：当前路径下，2030年达峰压力较大\n\n"
        f"- 当前预测 2030 年排放：**{current_2030:,.0f} Mt**（{current_2030/100:.0f} 亿吨）\n"
        f"- 达峰目标阈值：**{DAPEK_THRESHOLD:,} Mt**（{DAPEK_THRESHOLD/100:.0f} 亿吨）\n"
        f"- 历史峰值：**{historical_peak:,.0f} Mt**（{historical_peak_year} 年）\n\n"
        f"**建议**：加大煤炭压减力度、提升清洁能源替代比例、加快能耗强度下降速度。"
    )
else:
    # 绿色通过
    st.success(
        f"### ✅ 达峰可期，路径合理\n\n"
        f"- 当前预测 2030 年排放：**{current_2030:,.0f} Mt**（{current_2030/100:.0f} 亿吨）\n"
        f"- 达峰目标阈值：**{DAPEK_THRESHOLD:,} Mt**（{DAPEK_THRESHOLD/100:.0f} 亿吨）\n"
        f"- 低于阈值 **{(DAPEK_THRESHOLD - current_2030):,.0f} Mt**，路径安全可控。"
    )

st.markdown("---")

# -------------------- 一键最优策略对比 --------------------
st.subheader("🏆 一键最优策略对比（Baseline vs Optimized）")

# 使用顶部定义的全局最优参数 OPT_COAL / OPT_CLEAN / OPT_INTENSITY
df_optimal = compute_adjusted_prediction(df_emissions, OPT_COAL, OPT_CLEAN, OPT_INTENSITY)
baseline_2030_val = pred_total[-1]
optimal_2030_val = df_optimal.loc[2030, "总排放"]
optimal_reduction = baseline_2030_val - optimal_2030_val

# 按钮区：应用最优策略 + 双柱对比
btn_col1, btn_col2 = st.columns([1, 3])

with btn_col1:
    if st.button("🎯 应用最优策略", type="primary", use_container_width=True,
                  help=f"将滑块自动设置为：煤炭{OPT_COAL:+.0f}%、清洁+{OPT_CLEAN:.0f}%、能耗-{OPT_INTENSITY:.0f}%"):
        # 设置触发标志，在下一轮渲染时（滑块之前）修改 session_state
        st.session_state.optimal_triggered = True
        st.rerun()

with btn_col2:
    st.markdown(
        f"点击按钮后，上方三个滑块将自动跳转到最优参数："
        f"**煤炭 {OPT_COAL:+.0f}%** / **清洁 +{OPT_CLEAN:.0f}%** / **能耗 -{OPT_INTENSITY:.0f}%**"
    )

# -------------------- 双柱对比图 --------------------
st.subheader("▎2030 年碳排放对比：不干预 vs 最优策略")

fig_dual = go.Figure()

# 柱 1：不采取行动（Baseline）
fig_dual.add_trace(go.Bar(
    x=["不采取行动\n（Baseline）"],
    y=[baseline_2030_val],
    name="不采取行动",
    marker_color="#E74C3C",
    text=[f"{baseline_2030_val:,.0f} Mt"],
    textposition="outside",
    textfont=dict(size=14, color="#E74C3C"),
    hovertemplate="Baseline 2030: %{y:,.0f} Mt<extra></extra>",
))

# 柱 2：最优策略（Optimized）
fig_dual.add_trace(go.Bar(
    x=["最优策略\n（Optimized）"],
    y=[optimal_2030_val],
    name="最优策略",
    marker_color="#27AE60",
    text=[f"{optimal_2030_val:,.0f} Mt"],
    textposition="outside",
    textfont=dict(size=14, color="#27AE60"),
    hovertemplate="Optimized 2030: %{y:,.0f} Mt<extra></extra>",
))

# 添加达峰阈值参考线
fig_dual.add_hline(
    y=DAPEK_THRESHOLD, line_dash="dash", line_color="#E67E22", line_width=2,
    annotation_text=f"达峰阈值：{DAPEK_THRESHOLD/100:.0f} 亿吨",
    annotation_position="right",
    annotation_font=dict(size=13, color="#E67E22"),
)

# 在两根柱子之间添加减碳量标注箭头
mid_x = 0.5  # 两根柱子中间
fig_dual.add_annotation(
    x=mid_x, y=max(baseline_2030_val, optimal_2030_val) + 500,
    text=f"<b>预估减碳量：{optimal_reduction/100:.1f} 亿吨</b>",
    showarrow=False,
    font=dict(size=16, color="#2C3E50"),
    bgcolor="rgba(255, 255, 255, 0.8)",
    borderpad=8,
)

# 添加从 Baseline 柱顶到 Optimized 柱顶的虚线箭头
fig_dual.add_shape(
    type="line",
    x0=0, y0=baseline_2030_val,
    x1=1, y1=optimal_2030_val,
    line=dict(color="#8E44AD", width=2, dash="dot"),
)

fig_dual.update_layout(
    title=dict(
        text=f"2030 年碳排放情景对比<br><sup>最优策略：煤炭{OPT_COAL:+.0f}% / 清洁+{OPT_CLEAN:.0f}% / 能耗-{OPT_INTENSITY:.0f}% ｜ 预估减碳 {optimal_reduction/100:.1f} 亿吨</sup>",
        font=dict(size=16),
    ),
    yaxis=dict(title="碳排放量（百万吨 CO₂）", range=[0, max(baseline_2030_val, optimal_2030_val) * 1.2]),
    showlegend=False,
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig_dual, use_container_width=True)

# -------------------- 最优策略分项对比表 --------------------
st.subheader("▎最优策略分项对比（2030 年）")

# 构建对比表
df_compare = pd.DataFrame({
    "排放源": ["原煤", "原油", "天然气", "水泥", "总排放"],
    "不干预 Baseline (Mt)": [
        model_total.predict([[2030]])[0] * (df_emissions.loc[2022, COAL_COL] / df_emissions.loc[2022, TOTAL_COL]),
        model_total.predict([[2030]])[0] * (df_emissions.loc[2022, OIL_COL] / df_emissions.loc[2022, TOTAL_COL]),
        model_total.predict([[2030]])[0] * (df_emissions.loc[2022, GAS_COL] / df_emissions.loc[2022, TOTAL_COL]),
        # 用各分项LR预测值更准确
        None, None  # placeholder
    ],
})

# 用实际的各分项LR预测值
years_hist = df_emissions.index.values.reshape(-1, 1)
baseline_components = {}
for col in [COAL_COL, OIL_COL, GAS_COL, CEMENT_COL]:
    m = LinearRegression(); m.fit(years_hist, df_emissions[col].values)
    baseline_components[col] = m.predict([[2030]])[0]

df_compare = pd.DataFrame({
    "排放源": ["原煤", "原油", "天然气", "水泥", "总排放"],
    "不干预 Baseline (Mt)": [
        max(0, baseline_components[COAL_COL]),
        max(0, baseline_components[OIL_COL]),
        max(0, baseline_components[GAS_COL]),
        max(0, baseline_components[CEMENT_COL]),
        baseline_2030_val,
    ],
    "最优策略 Optimized (Mt)": [
        df_optimal.loc[2030, "原煤"],
        df_optimal.loc[2030, "原油"],
        df_optimal.loc[2030, "天然气"],
        df_optimal.loc[2030, "水泥"],
        optimal_2030_val,
    ],
})
df_compare["减排量 (Mt)"] = df_compare["不干预 Baseline (Mt)"] - df_compare["最优策略 Optimized (Mt)"]
df_compare["减排比例"] = (df_compare["减排量 (Mt)"] / df_compare["不干预 Baseline (Mt)"] * 100)

st.dataframe(
    df_compare.style.format({
        "不干预 Baseline (Mt)": "{:,.0f}",
        "最优策略 Optimized (Mt)": "{:,.0f}",
        "减排量 (Mt)": "{:,.0f}",
        "减排比例": "{:.1f}%",
    }),
    use_container_width=True,
)

st.success(
    f"✅ 第五轮完成：智能预警灯实时监控达峰路径，"
    f"最优策略（煤炭{OPT_COAL:+.0f}% / 清洁+{OPT_CLEAN:.0f}% / 能耗-{OPT_INTENSITY:.0f}%）"
    f"可减碳 {optimal_reduction/100:.1f} 亿吨。"
)
st.markdown("---")

# ======================== 第六轮：美化与导出 ========================

st.header("📥 报告导出与总结")

# -------------------- 报告摘要生成与 CSV 下载 --------------------
st.subheader("▎生成分析报告")

# 收集当前所有关键数据，构建报告内容
def build_report_csv():
    """
    构建完整的分析报告 CSV 字符串，包含：
      - 报告元信息（时间、参数）
      - 历史排放摘要
      - 当前情景预测数据（2023-2030）
      - 最优策略对比数据
      - 优化建议
    """
    output = io.StringIO()

    # ===== 第一部分：报告头 =====
    output.write("=" * 60 + "\n")
    output.write("零碳比赛 · 碳达峰情景模拟 — 分析报告\n")
    output.write(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write(f"数据来源：CEADs 中国国家表观碳排放清单（1997-2022）\n")
    output.write("=" * 60 + "\n\n")

    # ===== 第二部分：当前参数设置 =====
    output.write("【当前政策参数】\n")
    output.write(f"煤炭压减系数,{coal_adj:+.1f}%\n")
    output.write(f"清洁能源替代系数,+{clean_adj:.1f}%\n")
    output.write(f"能耗强度下降系数,{intensity_adj:.1f}%\n\n")

    # ===== 第三部分：历史排放摘要 =====
    output.write("【历史排放摘要（1997-2022）】\n")
    output.write("指标,1997年(Mt),2022年(Mt),变化(%)\n")
    for col, label in [(COAL_COL, "原煤"), (OIL_COL, "原油"),
                        (GAS_COL, "天然气"), (CEMENT_COL, "水泥"),
                        (TOTAL_COL, "总排放")]:
        v97 = df_emissions.loc[1997, col]
        v22 = df_emissions.loc[2022, col]
        chg = (v22 / v97 - 1) * 100
        output.write(f"{label},{v97:.0f},{v22:.0f},{chg:+.1f}%\n")

    # 能源结构
    output.write("\n【2022年能源结构】\n")
    output.write(f"煤炭占比,{df_share.loc[2022, '煤炭占比(%)']:.1f}%\n")
    output.write(f"石油占比,{df_share.loc[2022, '石油占比(%)']:.1f}%\n")
    output.write(f"天然气占比,{df_share.loc[2022, '天然气占比(%)']:.1f}%\n\n")

    # ===== 第四部分：当前情景预测 =====
    output.write("【当前情景预测（2023-2030）】\n")
    output.write("年份,原煤(Mt),原油(Mt),天然气(Mt),水泥(Mt),总排放(Mt)\n")
    for year in df_optimized.index:
        row = df_optimized.loc[year]
        output.write(f"{int(year)},{row['原煤']:.0f},{row['原油']:.0f},"
                     f"{row['天然气']:.0f},{row['水泥']:.0f},{row['总排放']:.0f}\n")

    # ===== 第五部分：达峰预警状态 =====
    output.write("\n【达峰预警状态】\n")
    current_2030 = df_optimized.loc[2030, "总排放"]
    status = "⚠️ 预警：当前路径下达峰压力较大" if current_2030 > DAPEK_THRESHOLD else "✅ 达峰可期，路径合理"
    output.write(f"2030年预测排放,{current_2030:.0f} Mt\n")
    output.write(f"达峰阈值,{DAPEK_THRESHOLD} Mt（120亿吨）\n")
    output.write(f"预警状态,{status}\n\n")

    # ===== 第六部分：最优策略对比 =====
    output.write("【最优策略对比（2030年）】\n")
    output.write(f"最优参数,煤炭{OPT_COAL:+.0f}% / 清洁+{OPT_CLEAN:.0f}% / 能耗-{OPT_INTENSITY:.0f}%\n")
    output.write("排放源,Baseline(Mt),Optimized(Mt),减排量(Mt),减排比例(%)\n")
    for _, row in df_compare.iterrows():
        output.write(f"{row['排放源']},{row['不干预 Baseline (Mt)']:.0f},"
                     f"{row['最优策略 Optimized (Mt)']:.0f},"
                     f"{row['减排量 (Mt)']:.0f},{row['减排比例']:.1f}%\n")

    # ===== 第七部分：优化建议 =====
    output.write("\n【优化建议】\n")
    if current_2030 > DAPEK_THRESHOLD:
        output.write("1. 建议进一步加大煤炭压减力度（推荐 -2% 以上）\n")
        output.write("2. 加快清洁能源替代步伐（推荐 +3% 以上）\n")
        output.write("3. 提升能效标准，降低单位能耗（推荐 -3% 以上）\n")
        output.write(f"4. 按最优策略可减碳约 {optimal_reduction/100:.1f} 亿吨\n")
    else:
        output.write("1. 当前路径基本满足达峰要求，建议保持政策力度\n")
        output.write("2. 可适度探索更深度的减排方案，争取提前达峰\n")

    output.write("\n" + "=" * 60 + "\n")
    output.write("报告结束。数据来源：CEADs (https://www.ceads.net.cn/)\n")
    output.write("引用：Shan et al., Scientific Data 5, 170201 (2018)\n")

    return output.getvalue()


# 生成报告按钮
report_col1, report_col2 = st.columns([1, 2])

with report_col1:
    st.download_button(
        label="📄 生成报告摘要（CSV）",
        data=build_report_csv(),
        file_name=f"碳达峰分析报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="下载包含当前参数、预测数据、最优策略对比和优化建议的完整报告。",
        use_container_width=True,
    )

with report_col2:
    st.markdown(
        "点击左侧按钮即可下载一份 **CSV 格式的分析报告**，内容包括：\n\n"
        "- 📋 当前政策参数设置\n"
        "- 📊 历史排放摘要（1997-2022）\n"
        "- 🔮 当前情景逐年预测（2023-2030）\n"
        "- 🚨 达峰预警状态\n"
        "- 🏆 Baseline vs Optimized 对比\n"
        "- 💡 个性化优化建议"
    )

# -------------------- 应用总结 --------------------
st.markdown("---")
st.subheader("📋 应用功能总览")

# 功能概览卡片
f1, f2, f3 = st.columns(3)
with f1:
    st.info(
        "**📊 历史分析**\n\n"
        "- 堆叠面积图：排放来源构成\n"
        "- 堆积柱状图：能源结构演变\n"
        "- 关键指标卡：2022年速览"
    )
with f2:
    st.info(
        "**🔮 预测与优化**\n\n"
        "- 线性回归趋势预测\n"
        "- 3 滑块情景模拟\n"
        "- 基准 vs 优化对比"
    )
with f3:
    st.info(
        "**🚨 决策支持**\n\n"
        "- 智能达峰预警灯\n"
        "- 一键最优策略\n"
        "- CSV 报告导出"
    )

# 技术栈说明
st.markdown("---")
st.caption(
    "🛠️ **技术栈**：Python 3 + Streamlit + Plotly + Pandas + Scikit-learn ｜ "
    "**数据源**：CEADs 中国碳核算数据库 ｜ "
    "**学术引用**：Shan et al., *Scientific Data* 5, 170201 (2018) ｜ "
    "**开发用途**：零碳比赛 · 智能控碳赛道"
)

st.success("✅ 第六轮完成：报告导出 + 美化总结，全功能看板已就绪！")
st.balloons()