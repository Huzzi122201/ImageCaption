
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import sys
import time

# Import Vocabulary at module level for pickle compatibility
from model import ImageCaptioningModel, Vocabulary

# Make Vocabulary available in __main__ namespace for pickle
sys.modules['__main__'].Vocabulary = Vocabulary

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Caption — AI Image Captioning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Global ── */
:root {
    --accent: #6C63FF;
    --accent-light: #8B83FF;
    --accent-dark: #4F46E5;
    --surface: #0E1117;
    --surface-2: #161B22;
    --surface-3: #1C2333;
    --border: rgba(108, 99, 255, 0.15);
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --success: #3FB950;
    --glow: rgba(108, 99, 255, 0.4);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main container ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(108, 99, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
    animation: aurora 8s ease-in-out infinite alternate;
}

@keyframes aurora {
    0% { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(-5%, 5%) rotate(3deg); }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #6C63FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0;
    position: relative;
    z-index: 1;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108, 99, 255, 0.15);
    border: 1px solid rgba(108, 99, 255, 0.3);
    color: var(--accent-light);
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── Glass Card ── */
.glass-card {
    background: linear-gradient(145deg, rgba(22, 27, 34, 0.8), rgba(14, 17, 23, 0.9));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(108, 99, 255, 0.35);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
    transform: translateY(-2px);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(108, 99, 255, 0.1);
}

.card-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(59, 130, 246, 0.15));
    border: 1px solid rgba(108, 99, 255, 0.2);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

/* ── Caption Output ── */
.caption-result {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(59, 130, 246, 0.05));
    border: 1px solid rgba(108, 99, 255, 0.25);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}

.caption-result::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), #3B82F6, var(--accent));
    border-radius: 14px 14px 0 0;
}

.caption-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-light);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}

.caption-text {
    font-size: 1.35rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.5;
    margin: 0;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.status-pill.success {
    background: rgba(63, 185, 80, 0.1);
    border: 1px solid rgba(63, 185, 80, 0.25);
    color: var(--success);
}

.status-pill.loading {
    background: rgba(108, 99, 255, 0.1);
    border: 1px solid rgba(108, 99, 255, 0.25);
    color: var(--accent-light);
}

/* ── Upload Area ── */
[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, rgba(22, 27, 34, 0.6), rgba(14, 17, 23, 0.8));
    border: 2px dashed rgba(108, 99, 255, 0.25);
    border-radius: 16px;
    padding: 1rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(108, 99, 255, 0.5);
    background: linear-gradient(145deg, rgba(22, 27, 34, 0.8), rgba(14, 17, 23, 0.9));
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.45) !important;
    background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent) 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] {
    padding: 0.5rem 0;
}

.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Image Display ── */
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Metrics Row ── */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-top: 1rem;
}

.metric-item {
    flex: 1;
    background: rgba(22, 27, 34, 0.6);
    border: 1px solid rgba(108, 99, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-light);
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── How-It-Works Steps ── */
.step-container {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 0.9rem 0;
}

.step-number {
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(108, 99, 255, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--accent-light);
    font-family: 'JetBrains Mono', monospace;
}

.step-content {
    flex: 1;
}

.step-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin: 0 0 3px 0;
}

.step-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.45;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: var(--text-secondary);
    font-size: 0.8rem;
    border-top: 1px solid rgba(108, 99, 255, 0.08);
    margin-top: 3rem;
}

.app-footer a {
    color: var(--accent-light);
    text-decoration: none;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Toasts / Alerts override ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ── Placeholder / empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    color: var(--text-secondary);
}

.empty-state-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}

.empty-state-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.empty-state-desc {
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 400px;
    margin: 0 auto;
}

/* ── Animate caption fade in ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.caption-result {
    animation: fadeInUp 0.5s ease-out;
}

/* ── Sidebar custom styling ── */
.sidebar-section {
    background: rgba(22, 27, 34, 0.5);
    border: 1px solid rgba(108, 99, 255, 0.1);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.sidebar-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent-light);
    margin-bottom: 0.8rem;
}

.sidebar-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.sidebar-item span.label {
    color: var(--text-primary);
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_vocab():
    with open('vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_size=384,
        hidden_size=768,
        num_layers=1,
    )
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab


@st.cache_resource
def load_feature_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)


def extract_features(image, feature_extractor):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(1, -1)
    return features.squeeze(0)


# ─── Load Models ─────────────────────────────────────────────────────────────

models_loaded = False
try:
    caption_model, vocab = load_model_and_vocab()
    feature_extractor = load_feature_extractor()
    models_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #E6EDF3;">Vision Caption</div>
        <div style="font-size: 0.78rem; color: #8B949E; margin-top: 2px;">AI-Powered Image Captioning</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">Architecture</div>
        <div class="sidebar-item">🔍 <span class="label">Encoder</span>&nbsp; ResNet-50</div>
        <div class="sidebar-item">🧬 <span class="label">Decoder</span>&nbsp; LSTM</div>
        <div class="sidebar-item">📐 <span class="label">Embed Dim</span>&nbsp; 384</div>
        <div class="sidebar-item">📏 <span class="label">Hidden</span>&nbsp; 768</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">Training Data</div>
        <div class="sidebar-item">🖼️ <span class="label">Dataset</span>&nbsp; Flickr30k</div>
        <div class="sidebar-item">📊 <span class="label">Images</span>&nbsp; 31,783</div>
        <div class="sidebar-item">📝 <span class="label">Vocab</span>&nbsp; ~7,689 words</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">How It Works</div>
        <div class="step-container">
            <div class="step-number">1</div>
            <div class="step-content">
                <p class="step-title">Upload Image</p>
                <p class="step-desc">Drop a JPG or PNG file</p>
            </div>
        </div>
        <div class="step-container">
            <div class="step-number">2</div>
            <div class="step-content">
                <p class="step-title">Extract Features</p>
                <p class="step-desc">ResNet-50 encodes visual info</p>
            </div>
        </div>
        <div class="step-container">
            <div class="step-number">3</div>
            <div class="step-content">
                <p class="step-title">Generate Caption</p>
                <p class="step-desc">LSTM decodes into natural language</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding-top: 1rem; font-size: 0.75rem; color: #484F58;">
        Built with Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">✨ Powered by Deep Learning</div>
    <h1 class="hero-title">Vision Caption</h1>
    <p class="hero-subtitle">
        Transform any image into a vivid description. Upload a photo and watch our
        AI weave words from pixels — powered by ResNet-50 and LSTM on the Flickr30k dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# Status pill
if models_loaded:
    st.markdown("""
    <div class="status-pill success">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        Models loaded &amp; ready
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("""
    <div class="card-header">
        <div class="card-icon">🖼️</div>
        <p class="card-title">Upload Your Image</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or browse — JPG, JPEG, PNG supported",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        # Image info metrics
        w, h = image.size
        size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-item">
                <div class="metric-value">{w}×{h}</div>
                <div class="metric-label">Resolution</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{size_kb:.0f} KB</div>
                <div class="metric-label">File Size</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{uploaded_file.type.split('/')[-1].upper()}</div>
                <div class="metric-label">Format</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📷</div>
            <div class="empty-state-title">No image uploaded yet</div>
            <div class="empty-state-desc">
                Upload a photo above to get started. The AI will analyze it
                and generate a natural language caption.
            </div>
        </div>
        """, unsafe_allow_html=True)


with col_result:
    st.markdown("""
    <div class="card-header">
        <div class="card-icon">💬</div>
        <p class="card-title">Generated Caption</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Settings
        max_length = st.slider(
            "Maximum caption length",
            min_value=10,
            max_value=30,
            value=20,
            help="Control how long the generated caption can be.",
        )

        generate_btn = st.button("✦  Generate Caption", type="primary", use_container_width=True)

        if generate_btn:
            with st.spinner(""):
                # Progress animation
                progress_placeholder = st.empty()
                progress_placeholder.markdown("""
                <div class="status-pill loading">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10" opacity="0.3"/><path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></path></svg>
                    Analyzing image...
                </div>
                """, unsafe_allow_html=True)

                features = extract_features(image, feature_extractor)

                progress_placeholder.markdown("""
                <div class="status-pill loading">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10" opacity="0.3"/><path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></path></svg>
                    Generating caption...
                </div>
                """, unsafe_allow_html=True)

                caption = caption_model.generate_caption(
                    features, vocab, max_length=max_length
                )

                progress_placeholder.empty()

            # Display result
            st.markdown(f"""
            <div class="caption-result">
                <div class="caption-label">AI-Generated Caption</div>
                <p class="caption-text">" {caption.capitalize()} "</p>
            </div>
            """, unsafe_allow_html=True)

            # Copy-friendly text below
            st.code(caption.capitalize(), language=None)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">✨</div>
            <div class="empty-state-title">Caption will appear here</div>
            <div class="empty-state-desc">
                Once you upload an image and click Generate, the AI model
                will produce a descriptive caption for your photo.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    Vision Caption &nbsp;·&nbsp; Seq2Seq Image Captioning &nbsp;·&nbsp;
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> &amp;
    <a href="https://pytorch.org" target="_blank">PyTorch</a>
</div>
""", unsafe_allow_html=True)
