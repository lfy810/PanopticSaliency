import os
import re
import sys
import time
import subprocess
from pathlib import Path

import streamlit as st


# =========================================================
# 路径配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "demo_inputs"
OUTPUT_DIR = PROJECT_ROOT / "demo_outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

# 优先调用支持 --out_dir 的推理脚本
if (PROJECT_ROOT / "demo_ablation_ready.py").exists():
    DEMO_SCRIPT = PROJECT_ROOT / "demo_ablation_ready.py"
    SUPPORT_EXTRA_ARGS = True
else:
    DEMO_SCRIPT = PROJECT_ROOT / "demo.py"
    SUPPORT_EXTRA_ARGS = False


# =========================================================
# 页面基础设置
# =========================================================

st.set_page_config(
    page_title="全景图像显著性排序系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 页面样式
# =========================================================

st.markdown(
    """
<style>
    :root {
        --page-bg: #f4f7fb;
        --card-bg: #ffffff;
        --card-bg-soft: #f8fafc;
        --border: #e2e8f0;
        --border-strong: #cbd5e1;
        --text-main: #0f172a;
        --text-sub: #475569;
        --text-muted: #64748b;
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --primary-soft: #eff6ff;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
        --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        --radius-lg: 22px;
        --radius-md: 16px;
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-main);
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--text-main);
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }

    .brand {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text-main);
        letter-spacing: 0.02em;
    }

    .brand-subtitle {
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    .status-pill {
        padding: 0.48rem 0.8rem;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid var(--border);
        color: var(--text-sub);
        font-size: 0.82rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(255, 255, 255, 0.96)),
            #ffffff;
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 2.2rem 2.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -110px;
        top: -130px;
        border-radius: 50%;
        background: rgba(37, 99, 235, 0.10);
        pointer-events: none;
    }

    .hero-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--primary-dark);
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.65rem;
        line-height: 1.15;
        font-weight: 900;
        color: var(--text-main);
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.04em;
    }

    .hero-desc {
        max-width: 980px;
        color: var(--text-sub);
        font-size: 1rem;
        line-height: 1.75;
        margin-bottom: 1.2rem;
    }

    .chip-row {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
    }

    .chip {
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid var(--border);
        padding: 0.46rem 0.75rem;
        color: var(--text-sub);
        font-size: 0.82rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.3rem 0;
    }

    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 850;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }

    .metric-desc {
        color: var(--text-muted);
        font-size: 0.78rem;
    }

    .panel {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 1.35rem;
        box-shadow: var(--shadow);
        min-height: 100%;
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.95rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 850;
        color: var(--text-main);
    }

    .panel-tag {
        font-size: 0.75rem;
        color: var(--text-muted);
        background: var(--card-bg-soft);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.35rem 0.65rem;
    }

    .notice {
        padding: 0.85rem 0.95rem;
        border-radius: 14px;
        background: var(--primary-soft);
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        font-size: 0.86rem;
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    .empty-state {
        padding: 3.2rem 1.2rem;
        text-align: center;
        border-radius: 20px;
        border: 1px dashed var(--border-strong);
        background: #f8fafc;
        color: var(--text-muted);
        line-height: 1.75;
    }

    .empty-title {
        color: var(--text-main);
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .summary-card {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
    }

    .summary-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.32rem;
        font-weight: 700;
    }

    .summary-value {
        font-size: 1.08rem;
        color: var(--text-main);
        font-weight: 850;
        word-break: break-word;
    }

    .pipeline-panel {
        margin-top: 1.2rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 1.35rem;
        box-shadow: var(--shadow);
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.9rem;
    }

    .step-card {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 0.85rem;
        text-align: center;
    }

    .step-index {
        width: 28px;
        height: 28px;
        line-height: 28px;
        margin: 0 auto 0.55rem auto;
        border-radius: 50%;
        background: var(--primary);
        color: white;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .step-name {
        color: var(--text-main);
        font-size: 0.88rem;
        font-weight: 800;
        margin-bottom: 0.28rem;
    }

    .step-desc {
        color: var(--text-muted);
        font-size: 0.74rem;
        line-height: 1.45;
    }

    .footer {
        color: var(--text-muted);
        text-align: center;
        padding: 1.5rem 0 0.4rem 0;
        font-size: 0.82rem;
    }

    .stButton > button {
        height: 3rem;
        border-radius: 14px !important;
        border: 1px solid var(--primary-dark) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
    }

    .stButton > button:hover {
        border-color: #1e40af !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }

    .stButton > button:disabled {
        background: #e2e8f0 !important;
        color: #94a3b8 !important;
        border-color: #cbd5e1 !important;
        box-shadow: none;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: var(--text-sub);
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: var(--primary-dark) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }

    img {
        border-radius: 16px !important;
        border: 1px solid var(--border);
    }

    code {
        border-radius: 10px !important;
    }

    @media (max-width: 1100px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .pipeline-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .hero-title {
            font-size: 2rem;
        }

        .summary-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# 工具函数
# =========================================================

def safe_filename(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fa5]", "_", stem)
    return f"{stem}{suffix}"


def parse_rank_text(text: str) -> dict:
    info = {
        "rank1_class": "-",
        "rank1_score": "-",
        "rank1_mean": "-",
        "rank1_yolo": "-",
        "rank_count": 0
    }

    ranks = re.findall(r"Rank\d+:", text)
    info["rank_count"] = len(ranks)

    m = re.search(
        r"Rank1:\s*Region\s+\d+\s+score=([0-9.]+)\s+class=([a-zA-Z_ ]+?)\s+yolo=([0-9.]+)\s+mean=([0-9.]+)",
        text
    )
    if m:
        info["rank1_score"] = m.group(1)
        info["rank1_class"] = m.group(2).strip()
        info["rank1_yolo"] = m.group(3)
        info["rank1_mean"] = m.group(4)

    return info


def parse_stdout(stdout: str) -> dict:
    data = {
        "candidate_count": "-",
        "output_count": "-",
        "supplement_count": "-",
        "source": "-"
    }

    patterns = {
        "candidate_count": r"检测到候选区域数量:\s*(\d+)",
        "output_count": r"筛选出的重要区域数量:\s*(\d+)",
        "supplement_count": r"显著图补充区域数量:\s*(\d+)",
        "source": r"区域来源:\s*(.+)"
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, stdout)
        if m:
            data[key] = m.group(1).strip()

    return data


def run_inference(image_path: Path, out_dir: Path):
    cmd = [str(PYTHON_EXE), str(DEMO_SCRIPT), str(image_path)]

    if SUPPORT_EXTRA_ARGS:
        cmd += [
            "--out_dir", str(out_dir),
            "--variant", "ui_demo"
        ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env
    )
    elapsed = time.time() - start_time

    return result, elapsed, cmd


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def file_download_button(path: Path, label: str, mime: str):
    if path.exists():
        with open(path, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=path.name,
                mime=mime,
                use_container_width=True
            )


# =========================================================
# 侧边栏
# =========================================================

with st.sidebar:
    st.markdown("## 系统设置")
    st.caption("全景图像显著性排序演示系统")

    st.markdown("---")
    st.markdown("### 推理脚本")
    st.code(DEMO_SCRIPT.name, language="text")

    st.markdown("### Python 环境")
    st.code(str(PYTHON_EXE), language="text")

    st.markdown("### 输出目录")
    st.code(str(OUTPUT_DIR), language="text")

    st.markdown("---")
    show_log = st.toggle("显示运行日志", value=False)
    show_download = st.toggle("显示下载按钮", value=True)

    st.markdown("---")
    st.info("建议上传 2:1 比例的全景图像，例如 1024×512 或 2048×1024。")


# =========================================================
# 顶部标题区域
# =========================================================

st.markdown(
    """
<div class="topbar">
    <div class="brand">
        <div class="brand-title">全景图像显著性排序系统</div>
        <div class="brand-subtitle">Panoramic Image Saliency Ranking System</div>
    </div>
    <div class="status-pill">Graduation Design Demo</div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero">
    <div class="hero-kicker">Transformer Based Panoramic Saliency Ranking</div>
    <div class="hero-title">面向全景图像的目标显著性排序演示平台</div>
    <div class="hero-desc">
        本系统面向全景图像中的多目标显著性排序任务，结合 YOLOv11x 实例分割、
        SG-DAF 畸变感知显著性预测、多特征融合评分、Attention 修正与 Top-K 输出，
        实现候选目标区域的自动提取、显著性响应分析和排序结果可视化。
    </div>
    <div class="chip-row">
        <div class="chip">YOLOv11x Instance Segmentation</div>
        <div class="chip">Swin Transformer Backbone</div>
        <div class="chip">SG-DAF Distortion-Aware Module</div>
        <div class="chip">Multi-feature Ranking</div>
        <div class="chip">Top-K Output</div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Candidate</div>
        <div class="metric-value">YOLOv11x</div>
        <div class="metric-desc">实例分割候选区域</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Saliency</div>
        <div class="metric-value">SG-DAF</div>
        <div class="metric-desc">畸变感知显著性预测</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Ranking</div>
        <div class="metric-value">Top-K</div>
        <div class="metric-desc">多特征目标级排序</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Dataset</div>
        <div class="metric-value">F-360iSOD</div>
        <div class="metric-desc">全景图像数据集</div>
    </div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# 主体内容
# =========================================================

left_col, right_col = st.columns([0.92, 1.08], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel-header">
    <div class="panel-title">图像上传与预览</div>
    <div class="panel-tag">Input</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="notice">
上传一张全景图像后，系统将自动调用后端推理脚本，完成候选目标提取、显著性响应预测与目标排序。
</div>
""",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "上传全景图像",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    input_path = None

    if uploaded_file is not None:
        file_name = safe_filename(uploaded_file.name)
        input_path = INPUT_DIR / file_name

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(str(input_path), caption=f"输入图像：{file_name}", use_container_width=True)
    else:
        st.markdown(
            """
<div class="empty-state">
    <div class="empty-title">等待上传图像</div>
    <div>支持 PNG、JPG、JPEG、BMP 格式。建议使用 2:1 比例的全景图像。</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel-header">
    <div class="panel-title">模型分析与输出结果</div>
    <div class="panel-tag">Output</div>
</div>
""",
        unsafe_allow_html=True
    )

    run_button = st.button(
        "开始显著性排序分析",
        disabled=(uploaded_file is None),
        use_container_width=True
    )

    if run_button and input_path is not None:
        progress = st.progress(0)
        status_text = st.empty()

        status_text.write("正在读取图像...")
        progress.progress(15)
        time.sleep(0.1)

        status_text.write("正在执行候选区域提取与显著性预测...")
        progress.progress(45)

        result, elapsed, cmd = run_inference(input_path, OUTPUT_DIR)

        status_text.write("正在整理输出结果...")
        progress.progress(85)
        time.sleep(0.1)

        if result.returncode == 0:
            progress.progress(100)
            status_text.success("分析完成")
        else:
            progress.progress(100)
            status_text.error("模型运行失败")

        if result.returncode != 0:
            st.error("模型运行失败，请检查推理脚本、模型权重或运行环境。")
            st.markdown("#### 执行命令")
            st.code(" ".join(cmd), language="bash")
            st.markdown("#### 错误信息")
            st.code(result.stderr or result.stdout, language="text")
        else:
            stem = input_path.stem

            result_img = OUTPUT_DIR / f"{stem}_important_ranking.png"
            saliency_img = OUTPUT_DIR / f"{stem}_saliency.png"
            result_txt = OUTPUT_DIR / f"{stem}_important_ranking.txt"

            rank_text = read_text_file(result_txt)
            rank_info = parse_rank_text(rank_text)
            stdout_info = parse_stdout(result.stdout)

            tab_result, tab_saliency, tab_text, tab_log = st.tabs(
                ["排序结果", "显著性图", "详细数据", "运行日志"]
            )

            with tab_result:
                if result_img.exists():
                    st.image(str(result_img), caption="显著目标排序可视化", use_container_width=True)
                    if show_download:
                        file_download_button(result_img, "下载排序结果图", "image/png")
                else:
                    st.warning("未找到排序结果图，请检查输出路径。")

            with tab_saliency:
                if saliency_img.exists():
                    st.image(str(saliency_img), caption="显著性响应热力图", use_container_width=True)
                    if show_download:
                        file_download_button(saliency_img, "下载显著性图", "image/png")
                else:
                    st.warning("未找到显著性图，请检查输出路径。")

            with tab_text:
                if rank_text:
                    st.code(rank_text, language="text")
                    if show_download:
                        st.download_button(
                            "下载排序文本",
                            data=rank_text.encode("utf-8"),
                            file_name=result_txt.name,
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.warning("未找到详细排序文本。")

            with tab_log:
                if show_log:
                    st.markdown("#### 执行命令")
                    st.code(" ".join(cmd), language="bash")
                    st.markdown("#### 标准输出")
                    st.code(result.stdout, language="text")
                    if result.stderr:
                        st.markdown("#### 警告或错误输出")
                        st.code(result.stderr, language="text")
                else:
                    st.info("运行日志已隐藏，可在左侧设置中开启。")
    else:
        st.markdown(
            """
<div class="empty-state">
    <div class="empty-title">等待开始分析</div>
    <div>上传图像后点击按钮，系统将在此展示排序图、显著性图和 Rank 结果。</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 流程展示
# =========================================================

st.markdown(
    """
<div class="pipeline-panel">
    <div class="panel-header">
        <div class="panel-title">系统处理流程</div>
        <div class="panel-tag">Pipeline</div>
    </div>
    <div class="pipeline-grid">
        <div class="step-card">
            <div class="step-index">1</div>
            <div class="step-name">图像输入</div>
            <div class="step-desc">上传全景图像并保存到本地输入目录</div>
        </div>
        <div class="step-card">
            <div class="step-index">2</div>
            <div class="step-name">候选提取</div>
            <div class="step-desc">利用 YOLOv11x 获取实例级候选区域</div>
        </div>
        <div class="step-card">
            <div class="step-index">3</div>
            <div class="step-name">显著预测</div>
            <div class="step-desc">通过 SG-DAF 生成显著性响应图</div>
        </div>
        <div class="step-card">
            <div class="step-index">4</div>
            <div class="step-name">区域评分</div>
            <div class="step-desc">融合显著性、面积、中心度等特征</div>
        </div>
        <div class="step-card">
            <div class="step-index">5</div>
            <div class="step-name">排序修正</div>
            <div class="step-desc">结合 Attention 与 NMS 优化输出结果</div>
        </div>
        <div class="step-card">
            <div class="step-index">6</div>
            <div class="step-name">结果展示</div>
            <div class="step-desc">输出热力图、排序图和详细数据</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="footer">
    基于 Transformer 的全景图像显著性排序系统 | 毕业设计演示平台
</div>
""",
    unsafe_allow_html=True
)
