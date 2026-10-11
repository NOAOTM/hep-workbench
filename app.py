"""
app.py
=======
高能物理在线计算工作台 —— Streamlit 主界面

运行方式：
    streamlit run app.py

界面与 physics_core.py（纯物理计算内核）完全解耦：本文件只负责交互、展示与缓存，
所有数值计算都委托给 physics_core。
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import physics_core as pc

# ----------------------------------------------------------------------------
# 页面配置（必须是第一个 Streamlit 命令）
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="高能物理在线计算工作台",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# plotly 中文字体族（按平台优先级回退，浏览器端解析）
PLOTLY_FONT = dict(family="Microsoft YaHei, SimHei, Noto Sans CJK SC, PingFang SC, Arial",
                   size=13, color="#1f2933")
ORDER_LABEL = {1: "LO（领头阶）", 2: "NLO（次领头阶）", 3: "NNLO（次次领头阶）"}
ORDER_COLOR = {1: "#4c78a8", 2: "#e45756", 3: "#54a24b"}

# ----------------------------------------------------------------------------
# 物理过程元数据：介绍 + 核心 LaTeX 公式
# ----------------------------------------------------------------------------
PROCESS_INFO = {
    "Drell-Yan 过程": {
        "title": "Drell-Yan 过程  pp → γ*/Z → ℓ⁺ℓ⁻",
        "intro": (
            "Drell-Yan 过程是强子对撞机上产生高不变质量轻子对的核心通道，"
            "由夸克-反夸克湮灭经虚光子 γ* 或 Z 玻色子产生轻子对。它是精确检验"
            "标准模型电弱部分、测量质子部分子分布函数 (PDF) 与寻找新物理共振的黄金过程。"
            "下方曲线给出在双轻子不变质量窗口内的总截面随质心能量 √s 的变化，"
            "矩阵元中已完整包含 γ、Z 传播子及其干涉效应。"
        ),
        "formulas": [
            r"\frac{d\sigma}{dM^{2}}=\frac{4\pi\alpha_{em}^{2}}{9\,M^{2}s}\sum_{q}W_{q}(M^{2})\,\mathcal{L}_{q}(\tau),\qquad \tau=\frac{M^{2}}{s}",
            r"\mathcal{L}_{q}(\tau)=\int_{\tau}^{1}\frac{dx}{x}\left[q(x)\,\bar{q}\!\left(\frac{\tau}{x}\right)+\bar{q}(x)\,q\!\left(\frac{\tau}{x}\right)\right]",
            r"W_{q}=Q_{q}^{2}Q_{\ell}^{2}+2Q_{q}Q_{\ell}v_{q}v_{\ell}\,\mathrm{Re}\,\chi+(v_{q}^{2}+a_{q}^{2})(v_{\ell}^{2}+a_{\ell}^{2})\,|\chi|^{2}",
        ],
    },
    "重整化群演化": {
        "title": "重整化群演化  α_s(μ) 的跑动",
        "intro": (
            "强耦合常数 α_s 随能标 μ 的变化由 QCD 重整化群方程（β 函数）决定，"
            "其负号对应的“渐进自由”是 2004 年诺贝尔物理学奖的核心成果：能标越高，"
            "耦合越弱，微扰论越可靠。下方曲线展示从低能到 √s 的 α_s(μ) 跑动，"
            "并可对比不同微扰阶数 (LO/NLO/NNLO) 的差异。"
        ),
        "formulas": [
            r"\mu\frac{d\alpha_{s}}{d\mu}=\beta(\alpha_{s})=-\frac{\beta_{0}}{2\pi}\alpha_{s}^{2}-\frac{\beta_{1}}{4\pi^{2}}\alpha_{s}^{3}-\frac{\beta_{2}}{8\pi^{3}}\alpha_{s}^{4}",
            r"\beta_{0}=11-\frac{2}{3}n_{f},\qquad \beta_{1}=102-\frac{38}{3}n_{f}",
            r"\alpha_{s}(\mu)=\frac{\alpha_{s}(\mu_{0})}{1+\dfrac{\beta_{0}}{2\pi}\,\alpha_{s}(\mu_{0})\,\ln(\mu/\mu_{0})}\quad(\text{LO 解析解})",
        ],
    },
    "部分子分布函数演化": {
        "title": "部分子分布函数演化  DGLAP（非单态）",
        "intro": (
            "部分子分布函数 q(x, Q²) 描述在能标 Q 下、携带动量分数 x 的夸克密度，"
            "其随 Q 的演化由 DGLAP 方程给出。本工作台在 x 空间数值求解非单态（价夸克）"
            "演化方程，展示渐进自由下低 x 区分布随能标升高而增长、大 x 区被消耗的标度破坏现象。"
        ),
        "formulas": [
            r"\frac{\partial q(x,Q^{2})}{\partial \ln Q^{2}}=\frac{\alpha_{s}(Q^{2})}{2\pi}\int_{x}^{1}\frac{dz}{z}\,P_{qq}(z)\,q\!\left(\frac{x}{z},Q^{2}\right)",
            r"P_{qq}(z)=C_{F}\left[\frac{1+z^{2}}{(1-z)_{+}}+\frac{3}{2}\,\delta(1-z)\right],\qquad C_{F}=\frac{4}{3}",
        ],
    },
}

PROCESS_KEYS = list(PROCESS_INFO.keys())


# ----------------------------------------------------------------------------
# 带缓存的计算封装（@st.cache_data 避免相同参数重复计算）
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_drell_yan(sqrt_s_max, m_min, m_max, alpha_s_mz, order, n_points):
    """缓存的 Drell-Yan 计算封装。参数均为标量，便于 Streamlit 哈希。"""
    return pc.calc_drell_yan(sqrt_s_min=100.0, sqrt_s_max=float(sqrt_s_max),
                             n_points=int(n_points), m_min=float(m_min),
                             m_max=float(m_max), alpha_s_mz=float(alpha_s_mz),
                             order=int(order))


@st.cache_data(show_spinner=False)
def cached_rge(mu_min, mu_max, alpha_s_mz, order, compare, n_points):
    """缓存的 α_s 跑动计算封装。compare=True 时同时返回 LO/NLO/NNLO 三条曲线。"""
    mu = np.geomspace(float(mu_min), float(mu_max), int(n_points))
    orders = [1, 2, 3] if compare else [int(order)]
    curves = {}
    for o in orders:
        curves[o] = pc.running_alpha_s(mu, alpha_s_mz=float(alpha_s_mz), order=o)
    return {"mu": mu, "curves": curves}


@st.cache_data(show_spinner=False)
def cached_rge_reference_scales(mu_min, mu_max, alpha_s_mz, order):
    """在若干标准能标处求 α_s，用于数值表格。逐点求值并容错。"""
    candidates = [("m_c", 1.27), ("m_b", 4.18), ("M_Z", pc.MZ), ("m_t", 172.5),
                  ("1 TeV", 1000.0), ("3 TeV", 3000.0)]
    rows = []
    for name, scale in candidates:
        if not (mu_min <= scale <= mu_max):
            continue
        try:
            val = pc.running_alpha_s(scale, alpha_s_mz=float(alpha_s_mz), order=int(order))
            note = ""
        except Exception as exc:  # noqa: BLE001 - 单点失败不应中断整表
            val = float("nan")
            note = f"求值失败：{exc}"
        rows.append({"能标": name, "μ (GeV)": round(scale, 3),
                     "α_s(μ)": round(val, 5) if np.isfinite(val) else val, "备注": note})
    return rows


@st.cache_data(show_spinner=False)
def cached_dglap(q0, q1, flavor, alpha_s_mz, order, n_x, n_q):
    """缓存的 DGLAP 非单态演化封装。"""
    return pc.evolve_valence_q2(q0=float(q0), q1=float(q1), flavor=flavor,
                                alpha_s_mz=float(alpha_s_mz), order=int(order),
                                n_x=int(n_x), n_q=int(n_q))


@st.cache_data(show_spinner=False)
def cached_symbolic(n_f):
    """缓存的 sympy 符号推导结果（缺失 sympy 时会在调用处报错，由界面捕获）。"""
    return pc.symbolic_summary(n_f=int(n_f))


# ----------------------------------------------------------------------------
# 绘图辅助
# ----------------------------------------------------------------------------
def _base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str,
                 log_x: bool = False, log_y: bool = False) -> go.Figure:
    """统一图表样式：中文字体、宽屏、网格、坐标轴类型。"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=xtitle, yaxis_title=ytitle,
        font=PLOTLY_FONT, height=460, hovermode="x unified",
        template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=70, b=60),
    )
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def render_drell_yan(res: dict, sqrt_s_selected: float):
    """展示 Drell-Yan 计算结果：截面曲线 + 数值表格 + 关键指标。"""
    sqs = res["sqrt_s"]
    sig = res["sigma_pb"]
    params = res["params"]

    # 在用户选定的 √s 处插值取截面
    sigma_at_sel = float(np.interp(sqrt_s_selected, sqs, sig))

    c1, c2, c3 = st.columns(3)
    c1.metric("选定 √s", f"{sqrt_s_selected:.0f} GeV")
    c2.metric("该能量处截面 σ", f"{sigma_at_sel:,.2f} pb")
    c3.metric("微扰阶数", ORDER_LABEL[params["order"]].split("（")[0])

    st.warning(
        "⚠️ 本工作台的 PDF 为**简化玩具模型**（非 CTEQ/NNPDF/MMHT 全球拟合），"
        "Drell-Yan 截面的**绝对值仅用于演示能量依赖趋势**，不可与实验数据做定量比较；"
        "其中 NNLO(order=3) 的高阶 K 因子为经验估计——**此参数未经验证**。"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sqs, y=sig, mode="lines+markers",
                             name=f"σ(√s)，{ORDER_LABEL[params['order']]}",
                             line=dict(color=ORDER_COLOR[params["order"]], width=2.5),
                             marker=dict(size=5)))
    fig.add_vline(x=sqrt_s_selected, line_dash="dash", line_color="#888",
                  annotation_text=f"选定 √s={sqrt_s_selected:.0f}")
    _base_layout(fig, "Drell-Yan 总截面随质心能量的变化",
                 "质心能量 √s (GeV)",
                 f"σ (pb)　[M ∈ {params['m_min']:.0f}–{params['m_max']:.0f} GeV]")
    st.plotly_chart(fig, width="stretch")

    st.subheader("数值结果表")
    df = pd.DataFrame({"√s (GeV)": np.round(sqs, 1), "σ (pb)": np.round(sig, 4)})
    st.dataframe(df, width="stretch", height=320)
    st.caption("提示：截面在固定不变质量窗口内随 √s 升高而增大，源于低 x 部分子亮度的增长。")


def render_rge(res: dict, ref_rows: list, order: int):
    """展示 α_s 跑动结果：多阶对比曲线 + 标准能标数值表。"""
    mu = res["mu"]
    curves = res["curves"]

    fig = go.Figure()
    for o in sorted(curves.keys()):
        arr = np.asarray(curves[o])
        fig.add_trace(go.Scatter(x=mu, y=arr, mode="lines", name=ORDER_LABEL[o],
                                 line=dict(color=ORDER_COLOR[o], width=2.5)))
    fig.add_vline(x=pc.MZ, line_dash="dot", line_color="#aaa", annotation_text="M_Z")
    _base_layout(fig, "强耦合常数 α_s(μ) 的重整化群跑动",
                 "能标 μ (GeV)", "α_s(μ)", log_x=True)
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"当前主阶数：**{ORDER_LABEL[order]}**。曲线随能标升高单调下降，体现 QCD 渐进自由；"
        "阶数越高（含更多 β 函数系数），跑动越平缓。阈值 m_b、m_t 处自动切换活动味数 n_f。"
    )

    st.subheader("标准能标处的 α_s 数值")
    if ref_rows:
        st.dataframe(pd.DataFrame(ref_rows), width="stretch")
    else:
        st.write("当前 μ 范围内没有预置的标准能标，请调整 μ 范围。")


def render_dglap(res: dict):
    """展示 DGLAP 演化结果：不同 Q 下的 x·q_v(x) 曲线 + 数值表格。"""
    x = res["x"]
    Q = res["Q"]
    dist = res["dist"]              # (n_q, n_x)，价夸克密度 q_v(x, Q)
    flavor = res["params"]["flavor"]

    st.warning(
        "⚠️ 初始价夸克分布为**玩具参数化**；此处仅实现 **LO 非单态** DGLAP 演化，"
        "未包含单态-胶子耦合与 NLO 分裂函数——**此参数未经验证**，仅用于演示标度破坏的定性行为。"
    )

    fig = go.Figure()
    n_curves = min(len(Q), 6)
    idx_list = np.linspace(0, len(Q) - 1, n_curves).astype(int)
    cmap = ["#4c78a8", "#54a24b", "#e45756", "#f58518", "#72b7b2", "#b279a2"]
    for k, idx in enumerate(idx_list):
        xq = x * dist[idx]          # 绘制更常用、行为更好的 x·q_v(x)
        fig.add_trace(go.Scatter(x=x, y=xq, mode="lines",
                                 name=f"Q = {Q[idx]:.1f} GeV",
                                 line=dict(color=cmap[k % len(cmap)], width=2.5)))
    _base_layout(fig, f"{flavor} 价夸克分布 x·q_v(x, Q) 的 DGLAP 演化",
                 "Bjorken x", f"x·q_v(x, Q)", log_x=True)
    st.plotly_chart(fig, width="stretch")

    st.info(
        "随能标 Q 升高，低 x 区分布增长、高 x 区被消耗（标度破坏），"
        "但价夸克数求和规则 ∫q_v dx 守恒。这源于辐射出的胶子进一步劈裂为夸克对。"
    )

    st.subheader("数值结果表：x·q_v(x, Q)")
    # 选取若干代表性 x 点构成表格
    sample_idx = np.linspace(0, len(x) - 1, min(10, len(x))).astype(int)
    table = {"x": np.round(x[sample_idx], 5)}
    for idx in idx_list:
        table[f"Q={Q[idx]:.1f} GeV"] = np.round(x[sample_idx] * dist[idx][sample_idx], 5)
    st.dataframe(pd.DataFrame(table), width="stretch")


def _render_saved(state: dict):
    """渲染已保存的计算结果（按过程分发），并显示耗时。"""
    res = state["result"]
    proc = state["process"]
    elapsed = state["elapsed"]
    cache_note = "（命中缓存，未重复计算）" if state["cache_hit"] else ""
    if proc == "Drell-Yan 过程":
        render_drell_yan(res, state["sqrt_s"])
    elif proc == "重整化群演化":
        ref_rows = res.get("_ref", [])
        render_rge(res, ref_rows, state.get("order", 2))
    else:
        render_dglap(res)
    st.success(f"⏱️ 计算耗时：{elapsed * 1000:.1f} ms {cache_note}")


# ----------------------------------------------------------------------------
# 侧边栏：参数控制
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ 参数控制")

process = st.sidebar.selectbox("选择物理过程", PROCESS_KEYS, index=0)

sqrt_s = st.sidebar.slider(
    "质心能量 √s / 最高能标 (GeV)",
    min_value=100, max_value=3000, value=1000, step=50,
    help="Drell-Yan：截面扫描的上限能量；RGE：跑动的最高能标 μ；DGLAP：演化的终止能标 Q。",
)
alpha_s_mz = st.sidebar.slider(
    r"耦合常数 α_s(M_Z)",
    min_value=0.100, max_value=0.200, value=0.118, step=0.001, format="%.3f",
)
order = st.sidebar.number_input(
    "微扰阶数 n（1=LO, 2=NLO, 3=NNLO）",
    min_value=1, max_value=3, value=2, step=1,
)

st.sidebar.divider()
st.sidebar.caption("过程专属参数")

# 过程专属控件
dglap_flavor = "u"
rge_mu_min = 2.0
rge_compare = False
dy_m_min, dy_m_max = 10.0, 150.0
dglap_q0 = 2.0

if process == "Drell-Yan 过程":
    dy_m_min = st.sidebar.number_input("双轻子质量窗口下限 M_min (GeV)", 2.0, 200.0, 10.0, 1.0)
    dy_m_max = st.sidebar.number_input("双轻子质量窗口上限 M_max (GeV)", 5.0, 500.0, 150.0, 5.0)
elif process == "重整化群演化":
    rge_mu_min = st.sidebar.slider("起始能标 μ_min (GeV)", 1.0, 20.0, 2.0, 0.5)
    rge_compare = st.sidebar.checkbox("同时对比 LO / NLO / NNLO", value=True)
elif process == "部分子分布函数演化":
    dglap_q0 = st.sidebar.slider("起始能标 Q₀ (GeV)", 1.5, 20.0, 2.0, 0.5)
    dglap_flavor = st.sidebar.selectbox("价夸克味", ["u", "d"], index=0)

st.sidebar.divider()
compute_btn = st.sidebar.button("🚀 开始计算", type="primary", width="stretch")
st.sidebar.caption("计算结果使用 @st.cache_data 缓存：相同参数不重复运算。")


# ----------------------------------------------------------------------------
# 主界面
# ----------------------------------------------------------------------------
st.title("⚛️ 高能物理在线计算工作台")
st.caption("High-Energy Physics Online Workbench　|　重整化 · 规范场 · 微扰计算　|　"
           "计算内核见 physics_core.py")

info = PROCESS_INFO[process]
st.subheader(info["title"])
st.markdown(info["intro"])

with st.expander("📐 核心公式", expanded=True):
    for f in info["formulas"]:
        st.latex(f)

with st.expander("🧮 符号推导 (sympy)", expanded=False):
    try:
        sym = cached_symbolic(5)
        st.caption("β 函数系数（n_f = 5）：")
        st.latex(sym["beta0"])
        st.latex(sym["beta1"])
        st.caption("价夸克归一化（保证 ∫ q_v dx = N）：")
        st.latex(sym["valence_integral"])
        st.latex(sym["valence_norm"])
        st.caption("LO 重整化群方程的解析解：")
        st.latex(sym["lo_running"])
    except ImportError:
        st.warning("未安装 sympy，符号推导不可用。请运行 `pip install -r requirements.txt`。")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"符号推导暂不可用：{exc}")

# 当前参数概览
pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("物理过程", process.replace("过程", "").strip() or process)
pc2.metric("√s / 高能标", f"{sqrt_s} GeV")
pc3.metric(r"α_s(M_Z)", f"{alpha_s_mz:.3f}")
pc4.metric("微扰阶数", ORDER_LABEL[order].split("（")[0])

st.divider()

# ----------------------------------------------------------------------------
# 触发计算并渲染结果
# ----------------------------------------------------------------------------
if compute_btn:
    # 参数合法性预检查
    if process == "Drell-Yan 过程" and dy_m_max <= dy_m_min:
        st.error("质量窗口需满足 M_max > M_min，请修正后重试。")
        st.stop()
    if process == "重整化群演化" and sqrt_s <= rge_mu_min:
        st.error("最高能标 √s 必须大于起始能标 μ_min。")
        st.stop()
    if process == "部分子分布函数演化" and sqrt_s <= dglap_q0:
        st.error("终止能标 √s 必须大于起始能标 Q₀。")
        st.stop()

    with st.spinner("正在运行微扰计算..."):
        t0 = time.perf_counter()
        try:
            if process == "Drell-Yan 过程":
                result = cached_drell_yan(sqrt_s, dy_m_min, dy_m_max,
                                          alpha_s_mz, order, n_points=40)
            elif process == "重整化群演化":
                rge_res = cached_rge(rge_mu_min, sqrt_s, alpha_s_mz, order,
                                     rge_compare, n_points=60)
                ref_rows = cached_rge_reference_scales(rge_mu_min, sqrt_s,
                                                       alpha_s_mz, order)
                result = {"mu": rge_res["mu"], "curves": rge_res["curves"],
                          "_ref": ref_rows}
            else:  # 部分子分布函数演化
                result = cached_dglap(dglap_q0, sqrt_s, dglap_flavor,
                                      alpha_s_mz, order, n_x=80, n_q=24)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - 计算异常需友好呈现
            st.error(f"计算失败：{exc}")
            st.stop()

    # 命中缓存的判断（极短耗时）
    cache_hit = elapsed < 5e-3
    st.session_state["last_result"] = {
        "process": process, "result": result,
        "elapsed": elapsed, "cache_hit": cache_hit,
        "sqrt_s": sqrt_s, "order": order,
    }

# 渲染已保存的结果（保证切换侧边栏控件时结果不丢失）
state = st.session_state.get("last_result")
if state is None:
    st.info("👈 在左侧设置参数后，点击 **🚀 开始计算** 运行微扰计算并查看结果。")
else:
    if state["process"] != process:
        st.info(f"当前展示的是 **{state['process']}** 的结果。切换过程后请点击“开始计算”以更新。")
    _render_saved(state)
