"""
physics_core.py
================
高能物理在线计算工作台 —— 纯物理计算内核（与界面完全解耦）

本模块只包含物理与数值计算逻辑，不依赖 Streamlit，可被 app.py 调用，也可独立测试。
所有函数均带有详细 docstring，说明参数含义与单位。

实现的三个物理过程
------------------
1. 重整化群演化 (RGE)：QCD 强耦合常数 alpha_s(mu) 随能标的跑动，支持 LO/NLO/NNLO。
2. Drell-Yan 过程：pp -> gamma*/Z -> l+ l- 的强子对撞总截面，含电弱 gamma-Z 干涉，
   以及 QCD 高阶 K 因子修正。
3. 部分子分布函数 (PDF) 演化：非单态 (valence) 的 DGLAP 方程数值求解 (LO)。

单位约定
--------
- 能量、质量、质心能量 sqrt(s)、能标 mu/Q：GeV
- 截面：pb（1 GeV^-2 = 0.389379 mb = 3.89379e8 pb）
- alpha_s、alpha_em、sin^2(theta_W) 等耦合常数：无量纲

物理输入 (PDG 2022 近似值)
--------------------------
- M_Z = 91.1876 GeV,  Gamma_Z = 2.4952 GeV
- alpha_em^-1 = 137.036
- sin^2(theta_W) = 0.2312
- alpha_s(M_Z) 由用户输入（典型 0.118）
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import beta as _betafn

# ----------------------------------------------------------------------------
# 物理常数与单位换算
# ----------------------------------------------------------------------------
GEV2_TO_PB = 3.89379372e8      # 1 GeV^-2 -> pb
ALPHA_EM = 1.0 / 137.036        # 电磁精细结构常数 alpha_em (无量纲)
MZ = 91.1876                    # Z 玻色子质量 (GeV)
GAMMA_Z = 2.4952                # Z 玻色子宽度 (GeV)
SIN2_W = 0.2312                 # 弱混合角 sin^2(theta_W) (on-shell 近似)
CF = 4.0 / 3.0                  # SU(3) 二次 Casimir (夸克基础表示)

# 味相关量子数：{味: (电荷 Q_f, 弱同位旋第三分量 T3_f)}
_FERMION_QA = {
    "u": (2.0 / 3.0, +0.5),
    "c": (2.0 / 3.0, +0.5),
    "d": (-1.0 / 3.0, -0.5),
    "s": (-1.0 / 3.0, -0.5),
    "b": (-1.0 / 3.0, -0.5),
    "l": (-1.0, -0.5),          # 末态带电轻子 (e/mu)
}

# 阈值：当 mu 越过 (m_b, m_t) 时活动味数 n_f 递增
_NF_THRESHOLDS = (4.18, 172.5)  # (m_b, m_t) in GeV

# 玩具 PDF 价夸克归一化常数（预先算好，避免每次调用重复计算 Beta 函数）
# q_v(x) = A x^(a-1) (1-x)^b，其中 A = N / Beta(a, b+1)
_A_U_VAL = 2.0 / _betafn(0.5, 4.0)   # u 价夸克：a=0.5, b=3.0, N=2
_A_D_VAL = 1.0 / _betafn(0.5, 5.0)   # d 价夸克：a=0.5, b=4.0, N=1


# ----------------------------------------------------------------------------
# 基础工具函数
# ----------------------------------------------------------------------------
def beta_coefficients(n_f: int) -> tuple[float, float, float]:
    """返回 QCD beta 函数系数 (beta0, beta1, beta2)。

    beta 函数定义（t = ln mu^2）：
        d alpha_s / d t = -(beta0/(4 pi)) a^2 - (beta1/(16 pi^2)) a^3 - (beta2/(64 pi^3)) a^4

    参数
    ----
    n_f : int
        活动夸克味数 (2..6)。

    返回
    ----
    (beta0, beta1, beta2) : tuple[float, float, float]
        各阶 beta 函数系数 (无量纲)。
    """
    b0 = 11.0 - 2.0 * n_f / 3.0
    b1 = 102.0 - 38.0 * n_f / 3.0
    b2 = 2857.0 / 2.0 - 5033.0 / 18.0 * n_f + 325.0 / 54.0 * n_f ** 2
    return b0, b1, b2


def _nf_at_scale(mu: float) -> int:
    """根据能标 mu (GeV) 决定活动味数 n_f。mu < m_b -> 4; m_b <= mu < m_t -> 5; mu >= m_t -> 6。"""
    if mu < _NF_THRESHOLDS[0]:
        return 4
    if mu < _NF_THRESHOLDS[1]:
        return 5
    return 6


def _beta_rhs(t: float, y: np.ndarray, n_f: int, order: int) -> list[float]:
    """RGE 右端项，供 solve_ivp 使用。y = [alpha_s]，t = ln mu。"""
    a = float(y[0])
    b0, b1, b2 = beta_coefficients(n_f)
    dadt = -(b0 / (2.0 * np.pi)) * a ** 2
    if order >= 2:
        dadt -= (b1 / (4.0 * np.pi ** 2)) * a ** 3
    if order >= 3:
        dadt -= (b2 / (8.0 * np.pi ** 3)) * a ** 4
    return [dadt]


def running_alpha_s(mu, alpha_s_mz: float = 0.118, order: int = 2,
                    n_f: int | None = None):
    """求解重整化群方程，得到 alpha_s 在能标 mu 处的跑动值。

    数值积分 RGE：mu d alpha_s / d mu = beta(alpha_s)，从 M_Z 积分到目标能标。
    支持标量输入与 numpy 数组输入。

    参数
    ----
    mu : float 或 np.ndarray
        目标能标 (GeV)，必须为正。
    alpha_s_mz : float
        参考点 alpha_s(M_Z) 的取值 (无量纲)，典型值 0.118。
    order : int
        微扰阶数：1 = LO(仅 beta0)，2 = NLO(含 beta1)，3 = NNLO(含 beta2)。
    n_f : int 或 None
        活动味数。None 表示按能标自动跨越 m_b/m_t 阈值（更物理）；
        给定整数则在整段积分中固定 n_f。

    返回
    ----
    np.ndarray 或 float
        与输入 mu 同形状的 alpha_s(mu) 取值 (无量纲)。

    异常
    ----
    ValueError
        当 mu 非正、order 不在 {1,2,3}、或 alpha_s 在积分中发散 (Landau 极点) 时抛出。
    """
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    was_scalar = (np.ndim(mu) == 0)
    if np.any(mu_arr <= 0.0):
        raise ValueError("能标 mu 必须为正 (单位 GeV)。")
    if order not in (1, 2, 3):
        raise ValueError("微扰阶数 order 只能是 1(LO)/2(NLO)/3(NNLO)。")
    if not (0.0 < alpha_s_mz < 1.0):
        raise ValueError("alpha_s(M_Z) 取值不合理，应在 (0, 1) 区间。")

    out = np.empty_like(mu_arr)
    for i, m in enumerate(mu_arr):
        # 若正好在参考点，直接返回，避免积分误差
        if np.isclose(m, MZ, rtol=1e-12, atol=1e-9):
            out[i] = alpha_s_mz
            continue
        if n_f is not None:
            # 用户指定固定味数：单段积分
            val = _integrate_segment(np.log(MZ), np.log(m), alpha_s_mz, order, int(n_f))
        else:
            # 自动在 m_b/m_t 阈值处分段切换味数（更物理）
            val = _integrate_with_thresholds(m, alpha_s_mz, order)
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"alpha_s 在 mu={m:g} GeV 处发散 (可能越过 Landau 极点)。")
        out[i] = val
    return float(out[0]) if was_scalar else out


def _integrate_segment(t0: float, t1: float, a0: float, order: int, n_f: int) -> float:
    """在单一固定 n_f 下，把 alpha_s 从 t0=ln(mu0) 积分到 t1=ln(mu1)。"""
    sol = solve_ivp(_beta_rhs, (t0, t1), [a0], args=(n_f, order),
                    method="RK45", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise ValueError(f"RGE 分段积分失败：{sol.message}")
    return float(sol.y[0, -1])


def _integrate_with_thresholds(mu_target: float, alpha_s_mz: float, order: int) -> float:
    """从 M_Z 积分到 mu_target，自动在 m_b/m_t 阈值处分段并切换活动味数 n_f。

    每段内 n_f 恒定，用段内几何中点确定；阈值处 alpha_s 连续（未加匹配修正，
    对教学演示精度足够）。
    """
    if mu_target > MZ:
        inner = sorted(thr for thr in _NF_THRESHOLDS if MZ < thr < mu_target)
        bnds = [MZ] + inner + [mu_target]
    else:
        inner = sorted((thr for thr in _NF_THRESHOLDS if mu_target < thr < MZ), reverse=True)
        bnds = [MZ] + inner + [mu_target]
    a = alpha_s_mz
    for k in range(len(bnds) - 1):
        m0, m1 = bnds[k], bnds[k + 1]
        mid = np.sqrt(abs(m0 * m1))          # 段内代表能标 -> 决定 n_f
        a = _integrate_segment(np.log(m0), np.log(m1), a, order, _nf_at_scale(mid))
    return a


# ----------------------------------------------------------------------------
# 玩具 PDF 模型（简化，用于 Drell-Yan 演示）
# ----------------------------------------------------------------------------
def toy_pdf(x, Q: float = 10.0, flavor: str = "u"):
    """一个简化的部分子分布函数 (PDF) 玩具模型，仅用于教学演示。

    重要提示：这不是真实的全球拟合 PDF (CTEQ/NNPDF/MMHT)，只在大 x 区给出
    物理上合理的形状，用于演示 Drell-Yan 截面的能标依赖与 Z 峰结构。
    结果不应与实验数据做定量比较。

    价夸克密度：q_v(x) = A x^(a-1) (1-x)^b，满足求和规则 int_0^1 q_v(x) dx = N_valence
    海夸克密度：qbar(x) = S x^(-0.85) (1-x)^7（u/d/s 味对称），并随 Q 做轻微软化。

    参数
    ----
    x : float 或 np.ndarray
        Bjorken x (0 < x < 1)。
    Q : float
        因子化/重整化能标 (GeV)，此处仅用于对海夸克做弱依赖的参数化。
    flavor : str
        味标记：'u' 返回 u(x)=u_v+ubar；'ubar' 返回 ubar(x)；
        'd' 返回 d(x)=d_v+dbar；'dbar' 返回 dbar(x)；'sbar' 返回 sbar(x)。

    返回
    ----
    np.ndarray 或 float
        对应味的 PDF 密度 q(x, Q)（无量纲，标准 number-density 约定，即 int q dx = 粒子数）。
    """
    x_arr = np.asarray(x, dtype=float)
    scalar_input = (x_arr.ndim == 0)
    x_arr = np.atleast_1d(x_arr)

    eps = 1e-12
    xs = np.clip(x_arr, eps, 1.0 - eps)

    # 价夸克形状（归一化常数已预计算为模块级常量 _A_U_VAL / _A_D_VAL）
    def uv(xv):
        return _A_U_VAL * xv ** (-0.5) * (1.0 - xv) ** 3.0

    def dv(xv):
        return _A_D_VAL * xv ** (-0.5) * (1.0 - xv) ** 4.0

    # 海夸克：味对称，小 x 区随 x^(-0.85) 上升（模拟胶子劈裂驱动的海增长），
    # 随 Q 缓慢增强。纯玩具参数化，未做动量求和规则约束，也非真实全球拟合。
    S0 = 0.35
    lam = 0.25
    sea_norm = S0 * (1.0 + lam * np.log(max(Q, 1.0) / 10.0))

    def sea(xv):
        return sea_norm * xv ** (-0.85) * (1.0 - xv) ** 7.0

    if flavor == "u":
        val = uv(xs) + sea(xs)
    elif flavor == "ubar":
        val = sea(xs)
    elif flavor == "d":
        val = dv(xs) + sea(xs)
    elif flavor == "dbar":
        val = sea(xs)
    elif flavor == "sbar":
        val = 0.5 * sea(xs)   # s 海压低因子 0.5
    else:
        raise ValueError(f"未知 flavor={flavor}，应为 'u','ubar','d','dbar','sbar' 之一。")

    val = np.where((x_arr > 0) & (x_arr < 1), val, 0.0)
    return float(val[0]) if scalar_input else val


def ew_propagator_factor(M2: np.ndarray, q_flavor: str) -> np.ndarray:
    """计算 q qbar -> gamma*/Z -> l+ l- 的电弱传播子/耦合因子 W_q(M^2)。

    W_q = Q_q^2 Q_l^2
          + 2 Q_q Q_l v_q v_l Re(chi)
          + (v_q^2 + a_q^2)(v_l^2 + a_l^2) |chi|^2

    其中 chi(M^2) = [1/(4 s_w^2 c_w^2)] * M^2 / (M^2 - M_Z^2 + i M_Z Gamma_Z)，
    v_f = T3_f - 2 Q_f s_w^2, a_f = T3_f。

    参数
    ----
    M2 : np.ndarray
        双轻子不变质量的平方 (GeV^2)。
    q_flavor : str
        初态夸克味 ('u','c','d','s','b')。

    返回
    ----
    np.ndarray
        无量纲的电弱因子 W_q(M^2)。
    """
    Qq, T3q = _FERMION_QA[q_flavor]
    Ql, T3l = _FERMION_QA["l"]
    vq = T3q - 2.0 * Qq * SIN2_W
    aq = T3q
    vl = T3l - 2.0 * Ql * SIN2_W
    al = T3l
    denom = M2 - MZ ** 2 + 1j * MZ * GAMMA_Z
    chi = (1.0 / (4.0 * SIN2_W * (1.0 - SIN2_W))) * (M2 / denom)
    W = (Qq ** 2 * Ql ** 2
         + 2.0 * Qq * Ql * vq * vl * np.real(chi)
         + (vq ** 2 + aq ** 2) * (vl ** 2 + al ** 2) * np.abs(chi) ** 2)
    return W


def qcd_k_factor(alpha_s_scale: np.ndarray, order: int) -> np.ndarray:
    """Drell-Yan 过程的 QCD 高阶修正 K 因子（近似）。

    order = 1 (LO)：K = 1
    order = 2 (NLO)：K = 1 + (alpha_s/(2 pi)) * C_F * pi^2
              （保留软+虚胶子主导的 pi^2 增强项，是标准 NLO K 因子的主要部分）
    order = 3 (NNLO)：在 NLO 基础上叠加一个近似的二阶项

    参数
    ----
    alpha_s_scale : np.ndarray
        在相应能标处的 alpha_s 值 (无量纲)。
    order : int
        微扰阶数 1/2/3。

    返回
    ----
    np.ndarray
        K 因子 (无量纲)。
    """
    a = np.asarray(alpha_s_scale, dtype=float)
    K = np.ones_like(a)
    if order >= 2:
        K = K + (a / (2.0 * np.pi)) * CF * np.pi ** 2
    if order >= 3:
        # 近似 NNLO 增量项（系数为经验估计，非精确计算）
        K = K + (a / (2.0 * np.pi)) ** 2 * 100.0
    return K


def calc_drell_yan(sqrt_s_min: float = 100.0, sqrt_s_max: float = 3000.0,
                   n_points: int = 40, m_min: float = 10.0, m_max: float = 150.0,
                   alpha_s_mz: float = 0.118, order: int = 1):
    """计算 Drell-Yan 过程 pp -> gamma*/Z -> l+ l- 的总截面随质心能量 sqrt(s) 的变化。

    在双轻子不变质量窗口 [m_min, m_max] 内积分 d sigma / d M^2，并对所有相关夸克味求和：

        d sigma / d M^2 = (4 pi alpha_em^2 / (9 M^2 s)) * sum_q W_q(M^2) * L_q(tau) * K

        L_q(tau) = int_tau^1 (dx/x) [ q(x) qbar(tau/x) + qbar(x) q(tau/x) ]，  tau = M^2 / s

    参数
    ----
    sqrt_s_min, sqrt_s_max : float
        质心能量 sqrt(s) 扫描范围 (GeV)。
    n_points : int
        sqrt(s) 网格点数。
    m_min, m_max : float
        双轻子不变质量积分窗口 (GeV)，用于定义"总截面"。
    alpha_s_mz : float
        参考 alpha_s(M_Z)，用于计算 K 因子处的 alpha_s(M)。
    order : int
        QCD 微扰阶数 1(LO)/2(NLO)/3(NNLO)，通过 K 因子体现。

    返回
    ----
    dict
        {
          'sqrt_s': np.ndarray,   # 质心能量 (GeV)
          'sigma_pb': np.ndarray, # 对应总截面 (pb)
          'params': dict          # 本次计算使用的参数快照
        }

    异常
    ----
    ValueError
        当能标范围非法或数值积分失败时抛出。
    """
    if sqrt_s_min <= 0 or sqrt_s_max <= sqrt_s_min:
        raise ValueError("需要 0 < sqrt_s_min < sqrt_s_max。")
    if m_min <= 0 or m_max <= m_min:
        raise ValueError("需要 0 < m_min < m_max。")
    n_points = int(max(2, min(n_points, 200)))

    sqrt_s_arr = np.linspace(sqrt_s_min, sqrt_s_max, n_points)
    sigma_pb = np.zeros(n_points, dtype=float)

    n_M = 48          # 不变质量积分网格点数
    n_x = 90          # 部分子亮度卷积网格点数（对数间距）

    # --- 预计算所有与 sqrt(s) 无关、只依赖 M 的量（关键性能优化）---
    # 否则会在 sqrt(s)×M 的双重循环里重复调用含 solve_ivp 的 running_alpha_s，极慢。
    M_grid = np.linspace(m_min, m_max, n_M)
    M2_grid = M_grid ** 2
    # 电弱因子：u 型 (u,c) 与 d 型 (d,s,b) 按 2:1 权重合并
    W_eff = (2.0 * ew_propagator_factor(M2_grid, "u")
             + ew_propagator_factor(M2_grid, "d")) / 3.0
    # 在能标 M 处的强耦合与 QCD K 因子
    a_M = np.atleast_1d(running_alpha_s(M_grid, alpha_s_mz=alpha_s_mz, order=order))
    K_arr = qcd_k_factor(a_M, order)
    # 与 s 无关的 M 部分前因子：4π α²/(9 M²) · W_eff · K · (GeV^-2 -> pb)
    m_prefactor = (4.0 * np.pi * ALPHA_EM ** 2) / (9.0 * M2_grid) * GEV2_TO_PB * W_eff * K_arr

    for i, sqs in enumerate(sqrt_s_arr):
        s = sqs ** 2
        dsdM2 = np.zeros(n_M, dtype=float)
        for j, M in enumerate(M_grid):
            tau = M2_grid[j] / s
            if tau >= 1.0:
                continue  # 运动学禁止
            # 部分子亮度积分 L_q(tau) = int_tau^1 (dx/x)[q(x) qbar(tau/x) + qbar(x) q(tau/x)]
            # 用对数间距 x 网格（等价于在 u=ln x 上均匀积分 dx/x = du），对小 x 分辨更好
            xg = np.geomspace(tau, 1.0 - 1e-9, n_x)
            x2g = np.clip(tau / xg, 0.0, 1.0 - 1e-9)
            lumin = np.zeros_like(xg)
            # u ubar + d dbar（正反两方向已含在 q(x1)qbar(x2)+qbar(x1)q(x2) 中）
            for qf, qbf in (("u", "ubar"), ("d", "dbar")):
                lumin += (toy_pdf(xg, Q=M, flavor=qf) * toy_pdf(x2g, Q=M, flavor=qbf)
                          + toy_pdf(xg, Q=M, flavor=qbf) * toy_pdf(x2g, Q=M, flavor=qf))
            # s sbar（海味对称，权重较低）
            lumin += 2.0 * toy_pdf(xg, Q=M, flavor="sbar") * toy_pdf(x2g, Q=M, flavor="sbar")
            Lq = trapezoid(lumin, np.log(xg))       # dx/x = d(ln x)
            dsdM2[j] = m_prefactor[j] / s * Lq

        # 对 M^2 积分：dM^2 = 2 M dM
        integrand = dsdM2 * 2.0 * M_grid
        sigma_pb[i] = float(trapezoid(integrand, M_grid))

    return {
        "sqrt_s": sqrt_s_arr,
        "sigma_pb": sigma_pb,
        "params": {
            "m_min": m_min, "m_max": m_max,
            "alpha_s_mz": alpha_s_mz, "order": order,
            "n_points": n_points,
        },
    }


# ----------------------------------------------------------------------------
# DGLAP 非单态演化 (x 空间数值求解)
# ----------------------------------------------------------------------------
def initial_valence_pdf(x, flavor: str = "u"):
    """DGLAP 演化的初始价夸克分布 q_v(x, Q0)，形状与 toy_pdf 的价部分一致。

    参数
    ----
    x : np.ndarray
        Bjorken x 网格 (0 < x < 1)。
    flavor : str
        'u' (2 个价夸克) 或 'd' (1 个价夸克)。

    返回
    ----
    np.ndarray
        初始价夸克密度 q_v(x, Q0)（标准 number-density 约定）。
    """
    x_arr = np.clip(np.asarray(x, dtype=float), 1e-9, 1.0 - 1e-9)
    if flavor == "u":
        return _A_U_VAL * x_arr ** (-0.5) * (1.0 - x_arr) ** 3.0
    elif flavor == "d":
        return _A_D_VAL * x_arr ** (-0.5) * (1.0 - x_arr) ** 4.0
    raise ValueError("flavor 只能是 'u' 或 'd'。")


def _pqq_convolution(q, x_grid, idx):
    """计算 P_qq (+  prescriptions) 与 q 的卷积在 x_grid[idx] 处的值。

    (P⊗q)(x)/C_F = -q(x) A(x) + int_x^1 dz (1+z^2)/(1-z) [ q(x/z)/z - q(x) ]
    A(x) = int_0^x (1+z^2)/(1-z) dz = -x - x^2/2 - 2 ln(1-x)

    参数
    ----
    q : np.ndarray
        当前分布值 (与 x_grid 对应)。
    x_grid : np.ndarray
        x 网格 (递增，log 间距)。
    idx : int
        目标网格点索引。

    返回
    ----
    float
        C_F * [...]（即未乘 alpha_s/(2 pi) 的卷积核）。
    """
    x = x_grid[idx]
    if x <= 0 or x >= 1:
        return 0.0
    qx = float(q[idx])
    Ax = -x - x ** 2 / 2.0 - 2.0 * np.log(1.0 - x)
    term_const = -qx * Ax

    # 卷积积分：z in [x, 1)，避开 z=1 端点奇异（被 bracket 抵消）
    n_z = 48
    z_grid = np.linspace(x, 1.0 - 1e-6, n_z)
    x_over_z = x / z_grid
    q_interp = np.interp(x_over_z, x_grid, q, left=0.0, right=qx)
    bracket = q_interp / z_grid - qx
    kernel = (1.0 + z_grid ** 2) / (1.0 - z_grid)
    integrand = kernel * bracket
    term_int = trapezoid(integrand, z_grid)
    return CF * (term_const + term_int)


def evolve_valence_q2(q0: float, q1: float, flavor: str = "u",
                      alpha_s_mz: float = 0.118, order: int = 2,
                      n_x: int = 80, n_q: int = 30):
    """用 DGLAP 方程在 x 空间数值演化非单态（价夸克）分布。

    求解 (t = ln Q^2)：
        dq(x, t)/dt = (alpha_s(Q)/(2 pi)) * (P_qq ⊗ q)(x, t)

    参数
    ----
    q0, q1 : float
        演化的起始与终止能标 (GeV)，需满足 1 < q0 < q1。
    flavor : str
        'u' 或 'd'，决定初始价夸克分布。
    alpha_s_mz : float
        参考 alpha_s(M_Z)，用于计算 alpha_s(Q)。
    order : int
        alpha_s 跑动的微扰阶数 1/2/3。
    n_x : int
        x 网格点数。
    n_q : int
        输出能标网格点数。

    返回
    ----
    dict
        {
          'x': np.ndarray,          # x 网格 (n_x,)
          'Q': np.ndarray,          # 能标网格 (n_q,)
          'dist': np.ndarray,       # 形状 (n_q, n_x)，dist[i] = q_v(x, Q_i) 价夸克密度
          'alpha_s': np.ndarray,    # alpha_s(Q_i)
          'params': dict
        }

    异常
    ----
    ValueError
        当能标范围非法或 ODE 求解失败时抛出。
    """
    if not (1.0 < q0 < q1):
        raise ValueError("需要 1 < q0 < q1 (单位 GeV)。")
    n_x = int(max(20, min(n_x, 200)))
    n_q = int(max(2, min(n_q, 60)))

    x_grid = np.logspace(-2, np.log10(0.95), n_x)
    q_init = initial_valence_pdf(x_grid, flavor)

    t0 = 2.0 * np.log(q0)
    t1 = 2.0 * np.log(q1)
    t_eval = np.linspace(t0, t1, n_q)

    def rhs(t, y):
        Q = np.exp(0.5 * t)
        a_s = running_alpha_s(np.array([Q]), alpha_s_mz=alpha_s_mz, order=order)[0]
        coeff = a_s / (2.0 * np.pi)
        dqdt = np.empty_like(y)
        for k in range(len(y)):
            dqdt[k] = coeff * _pqq_convolution(y, x_grid, k)
        return dqdt

    # 先把初始分布从 q0 处起步（假定初始参数化定义在 Q0 = q0）
    sol = solve_ivp(rhs, (t0, t1), q_init, t_eval=t_eval, method="RK45",
                    rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise ValueError(f"DGLAP 演化求解失败：{sol.message}")

    Q_arr = np.exp(0.5 * sol.t)
    dist = sol.y.T                       # (n_q, n_x)
    dist = np.clip(dist, 0.0, None)
    a_s_arr = running_alpha_s(Q_arr, alpha_s_mz=alpha_s_mz, order=order)

    return {
        "x": x_grid,
        "Q": Q_arr,
        "dist": dist,
        "alpha_s": np.atleast_1d(a_s_arr),
        "params": {"flavor": flavor, "q0": q0, "q1": q1,
                   "alpha_s_mz": alpha_s_mz, "order": order},
    }


# ----------------------------------------------------------------------------
# 符号推导（可选依赖 sympy）
# ----------------------------------------------------------------------------
def symbolic_summary(n_f: int = 5, a_val: float = 0.5, b_val: float = 3.0,
                     N_val: int = 2) -> dict:
    """用 sympy 做若干符号推导，返回 LaTeX 字符串字典，供界面展示。

    包含：
      - beta0 / beta1 作为 n_f 的符号表达式
      - 价夸克归一化积分 int_0^1 x^(a-1)(1-x)^b dx 与其归一化常数 A
      - LO 重整化群方程的解析解 alpha_s(mu)

    参数
    ----
    n_f : int
        活动味数，用于 beta 函数系数。
    a_val, b_val, N_val : float, float, int
        价夸克形状参数与价夸克数（归一化目标）。

    返回
    ----
    dict
        键为名称、值为 sympy 生成的 LaTeX 字符串。

    异常
    ----
    ImportError
        当未安装 sympy 时抛出（由调用方捕获并降级）。
    """
    import sympy as sp

    x, mu, mu0, alpha0 = sp.symbols("x mu mu_0 alpha_0", positive=True)
    nf = sp.Integer(int(n_f))
    beta0 = 11 - sp.Rational(2, 3) * nf
    beta1 = 102 - sp.Rational(38, 3) * nf

    a_sym = sp.Rational(float(a_val)).limit_denominator(100)
    b_sym = sp.Rational(float(b_val)).limit_denominator(100)
    N_sym = sp.Integer(int(N_val))
    integral = sp.integrate(x ** (a_sym - 1) * (1 - x) ** b_sym, (x, 0, 1))
    A_norm = sp.simplify(N_sym / integral)

    lo_running = alpha0 / (1 + beta0 / (2 * sp.pi) * alpha0 * sp.log(mu / mu0))

    return {
        "beta0": sp.latex(sp.Eq(sp.Symbol("beta_0"), beta0)),
        "beta1": sp.latex(sp.Eq(sp.Symbol("beta_1"), beta1)),
        "valence_integral": sp.latex(sp.Eq(
            sp.Integral(x ** (a_sym - 1) * (1 - x) ** b_sym, (x, 0, 1)),
            sp.simplify(integral))),
        "valence_norm": sp.latex(sp.Eq(sp.Symbol("A"), A_norm)),
        "lo_running": sp.latex(sp.Eq(sp.Function("alpha_s")(mu), sp.simplify(lo_running))),
    }


# ----------------------------------------------------------------------------
# 自测：直接运行本文件时做一次冒烟测试
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== physics_core 自测 ===")
    print("alpha_s(M_Z) LO/NLO/NNLO:")
    for o in (1, 2, 3):
        print(f"  order={o}: alpha_s(91.1876)={running_alpha_s(MZ, 0.118, o):.5f}, "
              f"alpha_s(1000)={running_alpha_s(1000.0, 0.118, o):.5f}")
    dy = calc_drell_yan(order=1, n_points=8)
    print("Drell-Yan sigma(pb) @ sqrt(s):",
          np.array2string(dy['sqrt_s'], precision=0), "->",
          np.array2string(dy['sigma_pb'], precision=3))
    ev = evolve_valence_q2(2.0, 100.0, "u", n_q=5, n_x=40)
    print("DGLAP Q 网格:", np.array2string(ev['Q'], precision=1))
    print("  低 x (x=%.3f) 分布随 Q 上升:", ev['x'][5],
          np.array2string(ev['dist'][:, 5], precision=4))
    try:
        sym = symbolic_summary(n_f=5)
        print("sympy 符号推导 OK，beta0 =", sym['beta0'][:40], "...")
    except ImportError:
        print("sympy 未安装，跳过符号推导自测。")
    print("自测完成。")
