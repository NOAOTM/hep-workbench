# ⚛️ 高能物理在线计算工作台

> High-Energy Physics Online Calculation Workbench
> 把平时跑的重整化、规范场、微扰计算 Python 代码，变成一个可交互的网页工作台。
> 在浏览器里调参数、跑计算、看交互式图、看 LaTeX 公式，并一键部署到免费云服务器分享给导师和同行。

技术栈：**Streamlit**（纯 Python 前端） + **numpy / scipy / sympy / pandas**（物理计算） + **plotly**（交互式可视化）。
计算内核与界面完全解耦：`physics_core.py` 只做物理，`app.py` 只做交互与展示。

---

## ✨ 功能特性

| 物理过程 | 说明 | 关键公式 |
| --- | --- | --- |
| **Drell-Yan 过程** | `pp → γ*/Z → ℓ⁺ℓ⁻` 总截面随质心能量 √s 变化，含 γ、Z 传播子与干涉，及 QCD 高阶 K 因子 | `dσ/dM²` 因子化 + 部分子亮度卷积 |
| **重整化群演化** | 强耦合常数 `α_s(μ)` 的跑动，支持 LO/NLO/NNLO 三阶 β 函数，自动跨 `m_b`、`m_t` 阈值切换 `n_f` | QCD β 函数 RGE |
| **部分子分布函数演化** | 非单态（价夸克）DGLAP 方程的 x 空间数值求解，展示标度破坏 | DGLAP + `P_qq` 分裂函数 |

其他亮点：
- 🖱️ 侧边栏滑块/输入框实时调参，点击「开始计算」触发后端运算。
- 📐 用 `st.latex` 渲染核心物理公式；`sympy` 做符号推导（β 系数、价夸克归一化、LO 解析解）。
- 📊 `plotly` 交互式图表 + `pandas` 数值结果表格 + 计算耗时提示。
- ⚡ `@st.cache_data` 缓存：相同参数不重复计算。
- ⚠️ 对玩具 PDF、经验 K 因子等未经验证之处用 `st.warning` 明确标注。

---

## 📁 项目结构

```
hep-workbench/
├── app.py              # Streamlit 主界面（交互 / 展示 / 缓存）
├── physics_core.py     # 纯物理计算内核（RGE / Drell-Yan / DGLAP，不依赖 Streamlit）
├── requirements.txt    # Python 依赖
├── README.md           # 本文件
└── .gitignore          # 忽略 __pycache__ / .env 等
```

---

## 🚀 本地安装与运行

> 建议使用 **Python 3.10 – 3.12**（Streamlit Community Cloud 默认 3.11/3.12）。
> 更高版本（如 3.14）部分依赖可能尚无预编译 wheel。

### 1. （推荐）创建并激活虚拟环境

**Windows PowerShell：**
```powershell
cd hep-workbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> 若提示脚本被禁用，先执行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux：**
```bash
cd hep-workbench
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行应用
```bash
streamlit run app.py
```
浏览器会自动打开 <http://localhost:8501>。若未自动打开，手动访问该地址即可。

### 4.（可选）单独验证物理内核
```bash
python physics_core.py
```
会打印 `α_s` 跑动、Drell-Yan 截面、DGLAP 演化的冒烟测试结果。

---

## 🧪 物理背景与实现说明

### 1. 重整化群演化（`α_s` 跑动）
数值求解 QCD β 函数（`t = ln μ²`）：

```
dα_s/dt = -(β0/4π) α_s² - (β1/16π²) α_s³ - (β2/64π³) α_s⁴
β0 = 11 - 2n_f/3 ,  β1 = 102 - 38n_f/3 ,  β2 = 2857/2 - 5033 n_f/18 + 325 n_f²/54
```

- 用 `scipy.integrate.solve_ivp` 从 `M_Z` 积分到目标能标，`α_s(M_Z)` 由用户输入锚定。
- 微扰阶数 `n`：1=LO（仅 β0）、2=NLO（含 β1）、3=NNLO（含 β2）。
- 跨越 `m_b=4.18`、`m_t=172.5` GeV 时分段切换活动味数 `n_f`。

### 2. Drell-Yan 过程
强子截面因子化，在双轻子不变质量窗口 `[M_min, M_max]` 内积分：

```
dσ/dM² = (4π α_em² / (9 M² s)) · Σ_q W_q(M²) · L_q(τ) · K ,   τ = M²/s
L_q(τ) = ∫_τ^1 (dx/x) [ q(x) q̄(τ/x) + q̄(x) q(τ/x) ]
```

`W_q` 为完整电弱因子（γ、Z 传播子 + 干涉），`K` 为 QCD 高阶修正因子（NLO 取主导 π² 增强项）。

### 3. DGLAP 非单态演化
在 x 空间数值求解（`t = ln Q²`）：

```
∂q(x,t)/∂t = (α_s(Q²)/2π) ∫_x^1 (dz/z) P_qq(z) q(x/z, t)
P_qq(z) = C_F [ (1+z²)/(1-z)_+ + (3/2) δ(1-z) ]
```

采用 plus  prescriptions 的正则化卷积，保证价夸克数求和规则 `∫ q_v dx = N` 守恒。

### ⚠️ 已知近似（务必阅读）
- **PDF 为简化玩具模型**，非 CTEQ/NNPDF/MMHT 全球拟合；Drell-Yan 截面**绝对值仅用于演示趋势**，不可与实验数据定量比较。
- **NNLO(order=3) 的 K 因子**为经验估计，非精确计算。
- DGLAP 仅实现 **LO 非单态**，未含单态-胶子耦合与 NLO 分裂函数。

以上在界面中均以 `st.warning` 明确提示。

---

## 📤 上传到 GitHub

在项目根目录（`hep-workbench/`）执行：

```bash
# 1. 初始化仓库
git init

# 2. 添加文件（.gitignore 会自动排除 __pycache__、.env 等）
git add .

# 3. 首次提交
git commit -m "feat: 高能物理在线计算工作台 (Streamlit + Drell-Yan/RGE/DGLAP)"

# 4. 关联远程仓库（先在 GitHub 网页上新建一个空仓库，例如 hep-workbench）
git branch -M main
git remote add origin https://github.com/<你的用户名>/hep-workbench.git

# 5. 推送
git push -u origin main
```

> 已有仓库更新：`git add . && git commit -m "update" && git push`

---

## ☁️ 部署到 Streamlit Community Cloud（免费）

1. **准备**：确保代码已推送到 **GitHub**（上面一步），且根目录含 `app.py` 与 `requirements.txt`。
2. **登录**：打开 <https://share.streamlit.io/>，点击 **Sign in with GitHub** 授权登录。
3. **新建应用**：点击右上角 **New app**。
4. **填写部署信息**：
   - **Repository**：选择 `你的用户名/hep-workbench`
   - **Branch**：`main`
   - **Main file path**：`app.py`
   - （可选）Advanced settings 里选择 Python 版本，推荐 **3.11** 或 **3.12**。
5. **部署**：点击 **Deploy!**，等待 1–3 分钟构建。
6. **获取公网链接**：部署完成后会得到形如
   `https://<你的子域名>.streamlit.app`
   的公网地址，直接分享给导师/同行即可（可在应用设置里自定义子域名）。

> 每次向 GitHub `main` 分支 `git push` 后，Cloud 会自动重新构建部署。

---

## 🛠️ 常见问题

- **本地能跑、Cloud 报错缺包**：检查 `requirements.txt` 是否包含全部依赖并已提交。
- **图表中文乱码**：plotly 在浏览器端渲染，已配置 `Microsoft YaHei / SimHei / Noto Sans CJK` 字体回退；若仍异常，多为浏览器缺中文字体。
- **Python 3.14 安装失败**：`streamlit`/部分依赖对最新版 Python 支持滞后，请改用 3.11/3.12 虚拟环境。
- **计算慢**：首次计算较慢，相同参数会命中 `@st.cache_data` 缓存而秒回。

---

## 📄 许可

本项目为科研教学演示用途，物理输入含简化近似，请勿用于正式发表结果的定量计算。
