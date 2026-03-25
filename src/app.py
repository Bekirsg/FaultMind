import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import anthropic
 
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
 
load_dotenv()
 
# ═══════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FaultMind | Agentic PLC/SCADA Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ═══════════════════════════════════════════════════════════════════
# 2. GLOBAL CSS — Industrial Dark Theme
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

[data-testid="stAppViewContainer"], .stApp {
    background-color: #0E1520 !important;
    color: #C8D6E5 !important;
    font-family: 'Inter', -apple-system, sans-serif !important; /* DM Sans yerine Inter geldi */
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important; /* Syne yerine Inter geldi */
    letter-spacing: -0.01em !important; /* Harf arası daraltıldı */
}
/* ... CSS'in geri kalanındaki tüm 'Syne' ve 'DM Sans' yazılarını 'Inter' yapmalısın. */
 
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0E1520; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00B4B4; }
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1118 0%, #0D1822 100%) !important;
    border-right: 1px solid #1A2D42 !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
 
/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 1.75rem !important; font-weight: 800 !important; color: #EDF2F7 !important; }
h2 { font-size: 1.2rem !important;  font-weight: 700 !important; color: #00C8C8 !important; }
h3 { font-size: 1rem !important;   font-weight: 600 !important; color: #00C8C8 !important; }
p, li { font-size: 0.875rem !important; line-height: 1.65 !important; }
 
/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    background: linear-gradient(135deg, #007A8A 0%, #009999 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.1rem !important;
    width: 100% !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #009999 0%, #00B4B4 100%) !important;
    box-shadow: 0 6px 20px rgba(0, 180, 180, 0.35) !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}
.stButton > button:active { transform: translateY(0) !important; }
 
/* ── Download Button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.22s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(56, 189, 248, 0.1) !important;
    border-color: #38BDF8 !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.2) !important;
    color: #38BDF8 !important;
}
 
/* ── File Uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #111E2E !important;
    border: 2px dashed rgba(0, 180, 180, 0.5) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00C8C8 !important;
    background: #152232 !important;
    box-shadow: 0 0 0 3px rgba(0, 180, 180, 0.1) !important;
}
[data-testid="stFileUploadDropzone"] * { color: #8FA8C0 !important; }
[data-testid="stFileUploadDropzone"] small { font-size: 0.72rem !important; }
 
/* ── Selectbox ── */
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div {
    background: #111E2E !important;
    color: #C8D6E5 !important;
    border-color: #1A2D42 !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
/* ── Dataframe / Table ── */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #1A2D42 !important;
}
[data-testid="stDataFrame"] th {
    background: #111E2E !important;
    color: #00C8C8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stDataFrame"] td {
    background: #0E1520 !important;
    color: #C8D6E5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}
 
/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
 
/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111E2E !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #C8D6E5 !important;
    padding: 0.7rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { background: rgba(0,180,180,0.05) !important; }
 
/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: #111E2E !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
/* ── Code blocks ── */
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #080F18 !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 6px !important;
}
code { color: #00C8C8 !important; padding: 0.1rem 0.35rem !important; }
 
/* ── Divider ── */
hr { border-color: #1A2D42 !important; margin: 1.25rem 0 !important; }
 
/* ── Caption ── */
.stCaption, small { color: #4A6480 !important; font-size: 0.73rem !important; }
 
/* ── Metric ── */
[data-testid="stMetric"] {
    background: #111E2E !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
 
/* ── Custom Components ── */
 
/* Header card */
.fm-hero {
    background: linear-gradient(135deg, #0B1929 0%, #0D1F35 50%, #0A1520 100%);
    border: 1px solid #1A3A55;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.fm-hero::before {
    content: '';
    position: absolute;
    top: -30%;
    right: 0;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(0,184,184,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.fm-hero::after {
    content: '';
    position: absolute;
    bottom: -20%;
    left: -5%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.04) 0%, transparent 65%);
    pointer-events: none;
}
 
/* Card */
.fm-card {
    background: #111E2E;
    border: 1px solid #1A2D42;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.85rem;
    transition: box-shadow 0.22s ease, border-color 0.22s ease;
}
.fm-card:hover {
    box-shadow: 0 8px 30px rgba(0, 180, 180, 0.12);
    border-color: rgba(0, 180, 180, 0.4);
}
 
/* Metric card */
.fm-metric {
    background: #111E2E;
    border: 1px solid #1A2D42;
    border-radius: 10px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: border-color 0.2s;
}
.fm-metric:hover { border-color: rgba(0,180,180,0.4); }
.fm-metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #00C8C8;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.fm-metric-val.red { color: #F87171; }
.fm-metric-val.yellow { color: #FBBF24; }
.fm-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #4A6480;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
 
/* Badge */
.fm-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0, 180, 180, 0.1);
    border: 1px solid rgba(0, 180, 180, 0.3);
    color: #00C8C8;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
}
.fm-badge.amber {
    background: rgba(251,191,36,0.1);
    border-color: rgba(251,191,36,0.3);
    color: #FBBF24;
}
.fm-badge.blue {
    background: rgba(56,189,248,0.1);
    border-color: rgba(56,189,248,0.3);
    color: #38BDF8;
}
.fm-badge.green {
    background: rgba(52,211,153,0.1);
    border-color: rgba(52,211,153,0.3);
    color: #34D399;
}
 
/* Sample log card */
.fm-sample {
    background: #0D1B2A;
    border: 1px solid #1A3A55;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.fm-sample:hover {
    border-color: rgba(0,180,180,0.5);
    background: #111E2E;
    box-shadow: 0 4px 16px rgba(0,180,180,0.1);
}
.fm-sample-icon {
    font-size: 1.1rem;
    margin-bottom: 0.35rem;
}
.fm-sample-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #D1E0F0;
    margin-bottom: 0.25rem;
}
.fm-sample-desc {
    font-size: 0.72rem;
    color: #4A6480;
    line-height: 1.45;
    margin-bottom: 0.35rem;
}
.fm-sample-hint {
    font-size: 0.68rem;
    color: #00A0A0;
    font-style: italic;
}
 
/* Step item */
.fm-step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0.9rem;
    background: rgba(0,180,180,0.05);
    border: 1px solid rgba(0,180,180,0.15);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
    color: #C8D6E5;
}
.fm-step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(0,180,180,0.2);
    color: #00C8C8;
    font-size: 0.7rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'Syne', sans-serif;
}
 
/* Feature card (welcome) */
.fm-feature {
    background: #111E2E;
    border: 1px solid #1A2D42;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.22s ease;
    height: 100%;
}
.fm-feature:hover {
    border-color: rgba(0,180,180,0.4);
    box-shadow: 0 8px 30px rgba(0,180,180,0.1);
    transform: translateY(-2px);
}
.fm-feature-icon { font-size: 2rem; margin-bottom: 0.7rem; }
.fm-feature-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #EDF2F7;
    margin-bottom: 0.4rem;
}
.fm-feature-desc { font-size: 0.78rem; color: #4A6480; line-height: 1.5; }
 
/* Status pill */
.fm-status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: #34D399;
}
.fm-status-online::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 6px #34D399;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
 
/* Error box */
.fm-error {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #F87171;
    margin-top: 0.5rem;
    line-height: 1.5;
}
 
/* Success box */
.fm-success {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #34D399;
    margin-top: 0.5rem;
}
 
/* GitHub star */
.fm-gh-star {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    color: #FBBF24;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
}
.fm-gh-star:hover {
    background: rgba(251,191,36,0.15);
    box-shadow: 0 4px 14px rgba(251,191,36,0.2);
    color: #FBBF24;
    text-decoration: none;
}
 
/* Report container */
.fm-report-wrap {
    background: #0D1B2A;
    border: 1px solid #1A3A55;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    line-height: 1.75;
}
 
/* FAQ item */
.fm-faq-q {
    font-size: 0.76rem;
    font-weight: 600;
    color: #C8D6E5;
    margin-bottom: 2px;
}
.fm-faq-a {
    font-size: 0.72rem;
    color: #4A6480;
    margin-bottom: 0.7rem;
    line-height: 1.45;
}
 
/* Sidebar section label */
.fm-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #4A6480;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
    font-family: 'Syne', sans-serif;
}
 
/* Footer */
.fm-footer {
    text-align: center;
    padding: 1.5rem 0 0.75rem;
    border-top: 1px solid #1A2D42;
    margin-top: 2.5rem;
    font-size: 0.73rem;
    color: #2E4A64;
    font-family: 'DM Sans', sans-serif;
}
.fm-footer a { color: #00A0A0; text-decoration: none; }
.fm-footer a:hover { color: #00C8C8; text-decoration: underline; }
 
/* Sidebar logo area */
.fm-sidebar-logo {
    background: linear-gradient(135deg, #0A1520 0%, #0D1F35 100%);
    border-bottom: 1px solid #1A2D42;
    padding: 1.1rem 1.2rem 0.9rem;
    margin-bottom: 0;
}
 
/* Welcome pulse */
@keyframes gentle-glow {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.03); }
}
.fm-welcome-icon { animation: gentle-glow 3s ease-in-out infinite; }
 
/* Print */
@media print {
    [data-testid="stAppViewContainer"], .stApp, body {
        background-color: white !important;
        color: black !important;
    }
    h1, h2, h3, h4, p, span, div, li, td, th { color: black !important; }
    [data-testid="stSidebar"], button,
    .stFileUploader, .stSpinner, .stDownloadButton,
    .stAlert, [data-testid="stStatusWidget"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ═══════════════════════════════════════════════════════════════════
_defaults = {
    "report": None,
    "language": "TR",
    "tool_steps": [],
    "uploaded_df": None,
    "current_file_path": None,
    "file_error": None,
    "trigger_analysis": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
 
# ═══════════════════════════════════════════════════════════════════
# 4. LANGUAGE TEXTS
# ═══════════════════════════════════════════════════════════════════
_T = {
    "TR": {
        "title": "FaultMind",
        "subtitle": "Agentic PLC/SCADA Arıza Tespit Asistanı",
        "tagline": "Siemens S7-1500/1200 için yapay zeka destekli kök neden analizi",
        "ctrl_panel": "Kontrol Paneli",
        "upload_section": "Log Dosyası Yükle",
        "upload_help": "CSV formatında, maks. 5 MB",
        "sample_section": "Örnek Senaryolar",
        "try_btn": "Bu Örneği Dene →",
        "analyze_btn": "🚀 Analizi Başlat",
        "analyzing": "🕵️‍♂️ FaultMind Analiz Yapıyor...",
        "done": "✅ Analiz Tamamlandı!",
        "timeout": "❌ Zaman Aşımı — Maksimum iterasyona ulaşıldı.",
        "download_btn": "📥 Raporu İndir (.md)",
        "preview_title": "📊 Dosya Önizleme",
        "preview_cap": "İlk 10 satır — analiz tüm kayıtlar üzerinde yapılır",
        "metric_alarms": "Toplam Alarm",
        "metric_devices": "Cihaz",
        "metric_critical": "Kritik",
        "metric_codes": "Hata Kodu",
        "steps_title": "⚙️ Ajan Düşünce Zinciri",
        "step_input": "Girdi",
        "step_output": "Çıktı",
        "report_title": "📋 FaultMind Teşhis Raporu",
        "welcome_h": "Analiz Yapmak İçin Hazır",
        "welcome_sub": "Sol menüden log dosyanızı yükleyin veya hazır senaryolardan birini deneyin.",
        "feat1_t": "Otonom Ajan",
        "feat1_d": "Adım adım, şeffaf düşünce zinciriyle otomatik tanılama.",
        "feat2_t": "RAG Destekli",
        "feat2_d": "Siemens resmi dokümantasyonuna dayalı doğrulama.",
        "feat3_t": "Zaman Analizi",
        "feat3_d": "Alarm kaskadını otomatik sıralama ve kök neden tespiti.",
        "system_status": "Sistem",
        "online": "Çevrimiçi",
        "model_lbl": "Model",
        "help_title": "❓ Nasıl Kullanılır?",
        "help_steps": [
            "1. CSV formatında log dosyanızı yükleyin (maks. 5 MB).",
            "2. Dosya önizlemesini kontrol edin.",
            "3. 'Analizi Başlat' butonuna tıklayın.",
            "4. Ajan adımlarını canlı izleyin.",
            "5. Raporu inceleyin veya indirin.",
        ],
        "faq_title": "Sık Sorulan Sorular",
        "faq": [
            ("Hangi PLC formatları destekleniyor?", "MVP: Siemens S7-1200/1500. Allen-Bradley ve Mitsubishi yakında."),
            ("Dosya boyutu sınırı nedir?", "Maksimum 5 MB, yalnızca .csv formatı."),
            ("Rapor dili değişiyor mu?", "Rapor her zaman Türkçe üretilir. Yalnızca arayüz dili değişir."),
            ("Verilerim saklanıyor mu?", "Hayır. Log verisi yalnızca analiz süresince bellekte tutulur."),
        ],
        "github_star": "⭐ GitHub'da Yıldızla",
        "err_size": "❌ Dosya boyutu 5 MB sınırını aşıyor. Lütfen daha küçük bir dosya seçin.",
        "err_format": "❌ Yalnızca .csv formatı kabul edilmektedir.",
        "err_csv": "❌ CSV dosyası okunamadı: ",
        "err_api_auth": "❌ API Hatası: Anthropic API anahtarınız geçersiz veya kotanız dolmuş. Lütfen hesabınızdan kontrol edin.",
        "err_api_gen": "❌ API bağlantı hatası: ",
        "file_ready": "✅ Dosya yüklendi. Analizi başlatabilirsiniz.",
        "step_parse": "📄 Log dosyası ayrıştırılıyor...",
        "step_time": "🔍 Zaman korelasyonu analiz ediliyor...",
        "step_rag": "📚 Siemens RAG taranıyor:",
        "scenario_lbl": "Beklenen Bulgu",
        "print_tip": "💡 **PDF olarak kaydet:** `Ctrl + P` tuşlarına basın.",
        "footer_copy": "FaultMind v1.0 — Açık Kaynak PLC/SCADA Arıza Tespit Asistanı",
        "footer_powered": "Anthropic Claude & LlamaIndex ile çalışmaktadır",
        "file_size_fmt": "Boyut: {size:.1f} KB · {rows} satır",
    },
    "EN": {
        "title": "FaultMind",
        "subtitle": "Agentic PLC/SCADA Fault Detection Assistant",
        "tagline": "AI-powered root cause analysis for Siemens S7-1500/1200 systems",
        "ctrl_panel": "Control Panel",
        "upload_section": "Upload Log File",
        "upload_help": "CSV format, max. 5 MB",
        "sample_section": "Sample Scenarios",
        "try_btn": "Try This Example →",
        "analyze_btn": "🚀 Start Analysis",
        "analyzing": "🕵️‍♂️ FaultMind Analyzing...",
        "done": "✅ Analysis Complete!",
        "timeout": "❌ Timeout — Maximum iterations reached.",
        "download_btn": "📥 Download Report (.md)",
        "preview_title": "📊 File Preview",
        "preview_cap": "First 10 rows — analysis runs on all records",
        "metric_alarms": "Total Alarms",
        "metric_devices": "Devices",
        "metric_critical": "Critical",
        "metric_codes": "Error Codes",
        "steps_title": "⚙️ Agent Thought Chain",
        "step_input": "Input",
        "step_output": "Output",
        "report_title": "📋 FaultMind Diagnostic Report",
        "welcome_h": "Ready to Analyze",
        "welcome_sub": "Upload a log file from the left menu, or try one of the ready-made scenarios.",
        "feat1_t": "Autonomous Agent",
        "feat1_d": "Automatic diagnosis with a transparent step-by-step reasoning chain.",
        "feat2_t": "RAG-Powered",
        "feat2_d": "Validated against official Siemens technical documentation.",
        "feat3_t": "Time Correlation",
        "feat3_d": "Automatic alarm cascade sorting and root cause identification.",
        "system_status": "System",
        "online": "Online",
        "model_lbl": "Model",
        "help_title": "❓ How to Use?",
        "help_steps": [
            "1. Upload your log file in CSV format (max. 5 MB).",
            "2. Check the file preview.",
            "3. Click 'Start Analysis'.",
            "4. Watch the agent steps live.",
            "5. Review or download the report.",
        ],
        "faq_title": "FAQ",
        "faq": [
            ("Which PLC formats are supported?", "MVP: Siemens S7-1200/1500. Allen-Bradley and Mitsubishi coming soon."),
            ("What is the file size limit?", "Maximum 5 MB, only .csv format."),
            ("Does the report language change?", "Reports are always in Turkish. Only the UI language changes."),
            ("Is my data stored?", "No. Log data is held in memory only during analysis."),
        ],
        "github_star": "⭐ Star on GitHub",
        "err_size": "❌ File size exceeds the 5 MB limit. Please select a smaller file.",
        "err_format": "❌ Only .csv format is accepted.",
        "err_csv": "❌ Could not read CSV file: ",
        "err_api_auth": "❌ API Error: Your Anthropic API key is invalid or quota exceeded. Please check your account.",
        "err_api_gen": "❌ API connection error: ",
        "file_ready": "✅ File uploaded. You can start the analysis.",
        "step_parse": "📄 Parsing log file...",
        "step_time": "🔍 Analyzing time correlation...",
        "step_rag": "📚 Searching Siemens RAG:",
        "scenario_lbl": "Expected Finding",
        "print_tip": "💡 **Save as PDF:** Press `Ctrl + P`.",
        "footer_copy": "FaultMind v1.0 — Open Source PLC/SCADA Fault Detection Assistant",
        "footer_powered": "Powered by Anthropic Claude & LlamaIndex",
        "file_size_fmt": "Size: {size:.1f} KB · {rows} rows",
    },
}
 
 
def t(key):
    return _T[st.session_state.language].get(key, key)
 
 
# ═══════════════════════════════════════════════════════════════════
# 5. SAMPLE LOG DATA
# ═══════════════════════════════════════════════════════════════════
SAMPLE_LOGS = [
    {
        "id": "profinet",
        "icon": "🌐",
        "color": "#0EA5E9",
        "title_TR": "PROFINET İletişim Arızası",
        "title_EN": "PROFINET Communication Failure",
        "desc_TR": "ET200SP uzak I/O istasyonu ağ kopması ve CPU STOP kaskadı.",
        "desc_EN": "ET200SP remote I/O station network drop and CPU STOP cascade.",
        "hint_TR": "İletişim hatası kök neden olarak tespit edilmeli.",
        "hint_EN": "Communication error should be identified as root cause.",
        "csv": (
            "Timestamp,Device,Error_Code,Severity,Message\n"
            "2024-01-15 08:45:01,CPU_1516-3_PN,W#16#0A15,Warning,PROFINET IO: Device ET200SP_01 not reachable\n"
            "2024-01-15 08:45:03,ET200SP_01,W#16#0A15,Error,Station failure - IO controller connection lost\n"
            "2024-01-15 08:45:05,CPU_1516-3_PN,W#16#0B25,Error,IO data transfer interrupted - missing cycles > 3\n"
            "2024-01-15 08:45:07,CPU_1516-3_PN,F#16#1F01,Critical,CPU switched to STOP mode - IO error limit exceeded\n"
            "2024-01-15 08:45:08,HMI_TP1500,E#16#FF04,Error,Visualization: PLC connection lost - runtime stopped\n"
            "2024-01-15 08:45:09,SCADA_WinCC,E#16#FF10,Critical,OPC-UA session terminated - all tags invalid\n"
            "2024-01-15 08:45:15,CPU_1516-3_PN,W#16#0A16,Info,Diagnostic buffer: Hardware fault logged at slot 3\n"
            "2024-01-15 08:45:20,ET200SP_01,W#16#0A17,Warning,Module PM-E DC24V power module voltage drop detected\n"
        ),
    },
    {
        "id": "thermal",
        "icon": "🌡️",
        "color": "#F97316",
        "title_TR": "Motor Termal Aşım Arızası",
        "title_EN": "Motor Thermal Overload Fault",
        "desc_TR": "Frekans sürücüsünde ısı aşımı ve güvenlik kilitleme zinciri.",
        "desc_EN": "Frequency drive thermal overload and safety interlock chain.",
        "hint_TR": "Motor sürücü sıcaklık aşımı kök neden olarak belirlenmeli.",
        "hint_EN": "Motor drive temperature overload should be identified as root cause.",
        "csv": (
            "Timestamp,Device,Error_Code,Severity,Message\n"
            "2024-01-16 14:20:00,SINAMICS_G120_M01,F7011,Warning,Motor temperature warning 85C - threshold 80C exceeded\n"
            "2024-01-16 14:20:30,SINAMICS_G120_M01,F7012,Warning,Motor temperature pre-alarm 92C - shutdown imminent\n"
            "2024-01-16 14:21:00,SINAMICS_G120_M01,F7023,Error,Motor thermal overload relay triggered - drive fault\n"
            "2024-01-16 14:21:01,CPU_1214C_DC/DC,W#16#0C05,Error,Drive fault signal received on DI channel 4\n"
            "2024-01-16 14:21:02,CPU_1214C_DC/DC,F#16#1F01,Critical,Safety interlock FB activated - conveyor STOP\n"
            "2024-01-16 14:21:03,SINAMICS_G120_M01,A7850,Error,Drive output disabled - waiting for reset command\n"
            "2024-01-16 14:21:05,HMI_KTP700,E#16#0301,Warning,Operator alarm: Conveyor Line 2 emergency stop active\n"
            "2024-01-16 14:25:00,SINAMICS_G120_M01,F7900,Info,Fault memory full - oldest entry overwritten\n"
        ),
    },
    {
        "id": "power",
        "icon": "⚡",
        "color": "#FBBF24",
        "title_TR": "Güç Kaynağı Gerilim Arızası",
        "title_EN": "Power Supply Voltage Fault",
        "desc_TR": "24V DC ray gerilim düşümü, uzak I/O kaybı ve acil CPU durdurma.",
        "desc_EN": "24V DC rail voltage sag, remote I/O loss and emergency CPU shutdown.",
        "hint_TR": "Güç kaynağı gerilim düşümü kök neden olarak belirlenmeli.",
        "hint_EN": "Power supply voltage sag should be identified as root cause.",
        "csv": (
            "Timestamp,Device,Error_Code,Severity,Message\n"
            "2024-01-17 22:10:00,SITOP_PSU8600,PS1001,Warning,Output voltage low: 22.1V (nominal 24V) - tolerance exceeded\n"
            "2024-01-17 22:10:02,SITOP_PSU8600,PS1002,Error,Voltage out of tolerance range 20-26.4V - current 8.2A\n"
            "2024-01-17 22:10:03,CPU_1515-2_PN,W#16#0D01,Error,Under-voltage detected on 24V backplane supply\n"
            "2024-01-17 22:10:04,ET200MP_02,W#16#0A15,Error,Remote IO station lost - power interruption on rack\n"
            "2024-01-17 22:10:04,ET200MP_03,W#16#0A15,Error,Remote IO station lost - power interruption on rack\n"
            "2024-01-17 22:10:05,CPU_1515-2_PN,F#16#1F01,Critical,CPU emergency STOP - supply voltage critical\n"
            "2024-01-17 22:10:06,SCADA_WinCC,E#16#FF10,Critical,Multiple tag groups invalid - PLC communication down\n"
            "2024-01-17 22:10:08,UPS_Module,UP1003,Warning,UPS battery activated - mains power supply failed\n"
        ),
    },
]
 
 
def _sample_field(s, key):
    lang = st.session_state.language
    return s.get(f"{key}_{lang}", s.get(f"{key}_TR", ""))
 
 
def load_example(sample):
    """Load a sample CSV into session state and trigger analysis."""
    import io
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"example_{sample['id']}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(sample["csv"])
    df = pd.read_csv(io.StringIO(sample["csv"]))
    st.session_state.uploaded_df = df
    st.session_state.current_file_path = path
    st.session_state.report = None
    st.session_state.tool_steps = []
    st.session_state.file_error = None
    st.session_state.trigger_analysis = True
 
 
# ═══════════════════════════════════════════════════════════════════
# 6. RAG ENGINE (CACHED)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = LlamaAnthropic(model="claude-sonnet-4-6", temperature=0.0)
    Settings.tokenizer = Settings.llm.tokenizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    storage_path = os.path.join(current_dir, "storage_native")
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)
 
 
# ═══════════════════════════════════════════════════════════════════
# 7. TOOL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def parse_log_file(file_path):
    try:
        df = pd.read_csv(file_path).head(20)
        return df.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
def analyze_time_correlation(alarms_json):
    try:
        alarms = json.loads(alarms_json)
        if not alarms or "error" in alarms:
            return json.dumps({"error": "No data available."})
        alarms.sort(key=lambda x: datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S"))
        return json.dumps({
            "root_cause_candidate": alarms[0],
            "cascade_end": alarms[-1],
            "start_time": alarms[0]["Timestamp"],
            "end_time": alarms[-1]["Timestamp"],
            "total_events": len(alarms),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
def query_siemens_rag(query_str):
    try:
        return str(get_rag_engine().query(query_str))
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
# ═══════════════════════════════════════════════════════════════════
# 8. AGENT LOOP
# ═══════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "name": "parse_log_file",
        "description": (
            "Reads and parses a CSV alarm/fault log file from disk. "
            "Returns JSON array of alarm records with all columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path to the CSV log file"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "analyze_time_correlation",
        "description": (
            "Sorts alarm records by Timestamp and identifies: "
            "(1) root_cause_candidate — the earliest alarm, "
            "(2) cascade_end — the latest alarm. "
            "Returns timing statistics and the full event sequence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"alarms_json": {"type": "string", "description": "JSON string of alarm records from parse_log_file"}},
            "required": ["alarms_json"],
        },
    },
    {
        "name": "query_siemens_rag",
        "description": (
            "Queries the Siemens S7 RAG documentation database for error codes, "
            "diagnostic messages, and official troubleshooting procedures. "
            "Call this once per significant error code found in the log."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query_str": {"type": "string", "description": "Error code or symptom description to look up"}},
            "required": ["query_str"],
        },
    },
]
 
SYSTEM_PROMPT = """\
Sen FaultMind'sın — Siemens S7-1200/1500 PLC/SCADA sistemleri için uzman bir arıza teşhis ajanısın.
 
ZORUNLU ARAÇ KULLANIM SIRASI:
1. parse_log_file → CSV logu oku, tüm alarm kayıtlarını al.
2. analyze_time_correlation → Zaman sıralaması yap, kök neden adayını tespit et.
3. query_siemens_rag → Kritik hata kodlarını (en az 2-3 sorgu) resmi Siemens belgelerinde araştır.
 
RAG KURALI: Lokal veritabanında spesifik hata kodu doğrudan bulunamazsa asla "bulunamadı" yazma.
Bunun yerine: "Bu kod veritabanımda doğrudan eşleşmiyor; ancak benzer hata sınıfı için Siemens tanılaması şöyledir:" de ve yorumla.
 
RAPOR FORMATI — Bu 5 başlığı ZORUNLU olarak, tam olarak bu sırayla kullan:
 
## 📋 YÖNETİCİ ÖZETİ
Arızanın ne olduğunu, ne zaman başladığını ve sistemin nasıl etkilendiğini 3-5 cümle ile özetle.
 
## ⏱️ ALARM ZAMAN ÇİZELGESİ (Kaskad Analizi)
Her alarmı sıralı Markdown tablosu olarak göster: Zaman | Cihaz | Hata Kodu | Mesaj | Yorum
 
## 🔍 ALARM VE HATA KODU DETAYLARI
Her önemli hata kodu için:
- Kodun resmi Siemens anlamı (RAG'dan)
- Sistemdeki etkisi
- - Hangi modül/cihazdan kaynaklandığı

## 🎯 KÖK NEDEN KARARI
Kanıta dayalı olarak kesin kök nedeni belirt. Güven seviyesini (Yüksek/Orta/Düşük) ve gerekçesini açıkla.
 
## 📊 AKSİYON PLANI VE SAHA MÜHENDİSİ REHBERİ
Tablo yerine numaralı adımlar halinde, saha mühendisinin sahada yapması gerekenleri açık ve akıcı şekilde yaz.
Kısa vadeli (anlık müdahale), orta vadeli (önleyici bakım) ve uzun vadeli (sistem iyileştirme) öneriler sun.
 
Tüm rapor Türkçe olsun. Profesyonel, net ve teknik açıdan kesin ol.
"""
 
 
def run_agent(log_file_path: str):
    """Run the FaultMind agent loop. Returns final report text or None on hard failure."""
    try:
        client = anthropic.Anthropic()
    except Exception:
        st.error(t("err_api_auth"))
        return None
 
    messages = [
        {
            "role": "user",
            "content": (
                f"Log dosyası: '{log_file_path}'\n"
                "Lütfen tam kapsamlı kök neden analizi raporu hazırla. "
                "Tüm araçları sırayla kullan ve raporu zorunlu 5 başlıkla bitir."
            ),
        }
    ]
    tool_steps = []
 
    with st.status(t("analyzing"), expanded=True) as status:
        for _iteration in range(8):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=6000,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.AuthenticationError:
                status.update(label=t("err_api_auth"), state="error")
                st.error(t("err_api_auth"))
                return None
            except Exception as exc:
                status.update(label=t("err_api_gen") + str(exc), state="error")
                st.error(t("err_api_gen") + str(exc))
                return None
 
            # Terminal state — LLM is done
            if response.stop_reason != "tool_use":
                status.update(label=t("done"), state="complete", expanded=False)
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                st.session_state.tool_steps = tool_steps
                return final_text
 
            # Tool-use round
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})
 
            tool_results = []
            for tu in tool_uses:
                name, inp = tu.name, tu.input
                step_num = len(tool_steps) + 1
 
                # Determine label
                if name == "parse_log_file":
                    label = t("step_parse")
                elif name == "analyze_time_correlation":
                    label = t("step_time")
                elif name == "query_siemens_rag":
                    label = f"{t('step_rag')} `{inp.get('query_str', '')}`"
                else:
                    label = f"🔧 {name}"
 
                st.write(f"**{step_num}.** {label}")
 
                # Execute tool
                if name == "parse_log_file":
                    result = parse_log_file(**inp)
                elif name == "analyze_time_correlation":
                    result = analyze_time_correlation(**inp)
                elif name == "query_siemens_rag":
                    result = query_siemens_rag(**inp)
                else:
                    result = json.dumps({"error": f"Unknown tool: {name}"})
 
                tool_steps.append(
                    {"step": step_num, "tool": name, "label": label, "input": inp, "output": result}
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                )
 
            messages.append({"role": "user", "content": tool_results})
 
        status.update(label=t("timeout"), state="error")
        st.session_state.tool_steps = tool_steps
        return "Analiz tamamlanamadı: Maksimum iterasyon sayısına ulaşıldı."
 
 
# ═══════════════════════════════════════════════════════════════════
# 9. FILE VALIDATION
# ═══════════════════════════════════════════════════════════════════
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
 
 
def handle_uploaded_file(uploaded_file) -> bool:
    st.session_state.file_error = None
    if uploaded_file.size > _MAX_BYTES:
        st.session_state.file_error = t("err_size")
        return False
    if not uploaded_file.name.lower().endswith(".csv"):
        st.session_state.file_error = t("err_format")
        return False
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", "uploaded_log.csv")
    with open(temp_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    try:
        df = pd.read_csv(temp_path)
        st.session_state.uploaded_df = df
        st.session_state.current_file_path = temp_path
        st.session_state.report = None
        st.session_state.tool_steps = []
        return True
    except Exception as exc:
        st.session_state.file_error = t("err_csv") + str(exc)
        return False
 
 
# ═══════════════════════════════════════════════════════════════════
# 10. SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
 
    # ── Logo + Language ──────────────────────────────────────────
    st.markdown("""
    <div class="fm-sidebar-logo">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:1.6rem;line-height:1;">⚙️</span>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
                                color:#EDF2F7;letter-spacing:-0.01em;">FaultMind</div>
                    <div style="font-size:0.62rem;color:#2E4A64;letter-spacing:0.07em;
                                text-transform:uppercase;">PLC · SCADA · AI</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Language toggle (inline, compact)
    col_lang_l, col_lang_r = st.columns([1, 1])
    with col_lang_l:
        if st.button("🇹🇷 TR", key="lang_tr",
                     help="Türkçe arayüz"):
            if st.session_state.language != "TR":
                st.session_state.language = "TR"
                st.rerun()
    with col_lang_r:
        if st.button("🇬🇧 EN", key="lang_en",
                     help="English interface"):
            if st.session_state.language != "EN":
                st.session_state.language = "EN"
                st.rerun()
 
    # Active language indicator
    lang_disp = "🇹🇷 Türkçe" if st.session_state.language == "TR" else "🇬🇧 English"
    st.markdown(f"<div style='font-size:0.68rem;color:#2E4A64;text-align:center;"
                f"margin-top:-0.3rem;margin-bottom:0.5rem;'>{lang_disp}</div>",
                unsafe_allow_html=True)
 
    st.divider()
 
    # ── Upload ───────────────────────────────────────────────────
    st.markdown(f"<div class='fm-section-label'>{t('upload_section')}</div>",
                unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader(
        t("upload_section"),
        type=["csv"],
        label_visibility="collapsed",
        help=t("upload_help"),
    )
 
    if uploaded_file is not None:
        ok = handle_uploaded_file(uploaded_file)
 
    # File feedback
    if st.session_state.file_error:
        st.markdown(f"<div class='fm-error'>{st.session_state.file_error}</div>",
                    unsafe_allow_html=True)
    elif st.session_state.current_file_path and st.session_state.uploaded_df is not None:
        df_sz = os.path.getsize(st.session_state.current_file_path) / 1024
        df_rows = len(st.session_state.uploaded_df)
        st.markdown(
            f"<div class='fm-success'>{t('file_ready')}<br>"
            f"<span style='font-size:0.68rem;opacity:0.7;'>"
            f"{t('file_size_fmt').format(size=df_sz, rows=df_rows)}</span></div>",
            unsafe_allow_html=True,
        )
        if st.button(t("analyze_btn"), key="analyze_main"):
            st.session_state.trigger_analysis = True
            st.rerun()
 
    st.divider()
 
    # ── Sample Logs ──────────────────────────────────────────────
    st.markdown(f"<div class='fm-section-label'>{t('sample_section')}</div>",
                unsafe_allow_html=True)
 
    for sample in SAMPLE_LOGS:
        title = _sample_field(sample, "title")
        desc = _sample_field(sample, "desc")
        hint = _sample_field(sample, "hint")
        icon = sample["icon"]
 
        st.markdown(f"""
        <div class="fm-sample">
            <div class="fm-sample-icon">{icon}</div>
            <div class="fm-sample-title">{title}</div>
            <div class="fm-sample-desc">{desc}</div>
            <div class="fm-sample-hint">🎯 {t('scenario_lbl')}: {hint}</div>
        </div>
        """, unsafe_allow_html=True)
 
        if st.button(t("try_btn"), key=f"try_{sample['id']}"):
            load_example(sample)
            st.rerun()
 
    st.divider()
 
    # ── Help / FAQ ────────────────────────────────────────────────
    with st.expander(t("help_title"), expanded=False):
        for step_text in t("help_steps"):
            st.markdown(
                f"<div style='font-size:0.76rem;color:#8FA8C0;margin-bottom:4px;'>{step_text}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='font-size:0.74rem;font-weight:700;color:#4A6480;"
            f"text-transform:uppercase;letter-spacing:0.06em;margin:0.7rem 0 0.4rem;'>"
            f"{t('faq_title')}</div>",
            unsafe_allow_html=True,
        )
        for q, a in t("faq"):
            st.markdown(
                f"<div class='fm-faq-q'>❓ {q}</div>"
                f"<div class='fm-faq-a'>{a}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='font-size:0.72rem;color:#2E4A64;margin-top:0.5rem;'>"
            "Issues: <a href='https://github.com/Bekirsg/FaultMind/issues' "
            "style='color:#00A0A0;' target='_blank'>GitHub Issues</a></div>",
            unsafe_allow_html=True,
        )
 
    st.divider()
 
    # ── Status + GitHub ───────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"<div class='fm-status-online'>{t('system_status')}: {t('online')}</div>",
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"<div style='font-size:0.68rem;color:#2E4A64;'>🧠 Sonnet 4.6</div>",
            unsafe_allow_html=True,
        )
 
    gh_star_txt = t("github_star")
    st.markdown(
        f"<div style='margin-top:0.8rem;'>"
        f"<a href='https://github.com/Bekirsg/FaultMind/stargazers' "
        f"target='_blank' class='fm-gh-star'>{gh_star_txt}</a></div>",
        unsafe_allow_html=True,
    )
 
 
# ═══════════════════════════════════════════════════════════════════
# 11. MAIN AREA
# ═══════════════════════════════════════════════════════════════════
 
# ── Hero Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="fm-hero">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;
                flex-wrap:wrap;gap:1rem;">
        <div>
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.6rem;">
                <span style="font-size:2.4rem;line-height:1;">⚙️</span>
                <div>
                    <h1 style="margin:0;font-size:1.65rem;font-family:'Syne',sans-serif;
                               font-weight:800;color:#EDF2F7;letter-spacing:-0.025em;">
                        {t("title")}
                        <span style="color:#00C8C8;"> | </span>
                        <span style="font-weight:600;font-size:1.2rem;">{t("subtitle")}</span>
                    </h1>
                    <p style="margin:0.3rem 0 0;font-size:0.83rem;color:#4A6480;
                              font-family:'DM Sans',sans-serif;">{t("tagline")}</p>
                </div>
            </div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;padding-top:0.2rem;">
            <span class="fm-badge">Siemens S7-1500</span>
            <span class="fm-badge">Siemens S7-1200</span>
            <span class="fm-badge amber">MVP v1.0</span>
            <span class="fm-badge green">Open Source</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ── Trigger Analysis (runs in main area so status renders here) ─────
if st.session_state.trigger_analysis and st.session_state.current_file_path:
    st.session_state.trigger_analysis = False
    result = run_agent(st.session_state.current_file_path)
    st.session_state.report = result
    st.rerun()
 
# ── File Preview (shown only when file loaded & no report yet) ──────
if st.session_state.uploaded_df is not None and st.session_state.report is None:
    df_prev = st.session_state.uploaded_df
 
    st.markdown(
        f"<div style='font-family:\"Syne\",sans-serif;font-size:0.95rem;"
        f"font-weight:700;color:#EDF2F7;margin-bottom:0.75rem;'>"
        f"{t('preview_title')}</div>",
        unsafe_allow_html=True,
    )
 
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    total_alarms = len(df_prev)
    devices = df_prev["Device"].nunique() if "Device" in df_prev.columns else "—"
    critical = (
        int((df_prev["Severity"] == "Critical").sum())
        if "Severity" in df_prev.columns
        else "—"
    )
    codes = df_prev["Error_Code"].nunique() if "Error_Code" in df_prev.columns else "—"
 
    with m1:
        st.markdown(
            f"<div class='fm-metric'>"
            f"<div class='fm-metric-val'>{total_alarms}</div>"
            f"<div class='fm-metric-label'>{t('metric_alarms')}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='fm-metric'>"
            f"<div class='fm-metric-val'>{devices}</div>"
            f"<div class='fm-metric-label'>{t('metric_devices')}</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='fm-metric'>"
            f"<div class='fm-metric-val red'>{critical}</div>"
            f"<div class='fm-metric-label'>{t('metric_critical')}</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"<div class='fm-metric'>"
            f"<div class='fm-metric-val yellow'>{codes}</div>"
            f"<div class='fm-metric-label'>{t('metric_codes')}</div></div>",
            unsafe_allow_html=True,
        )
 
    st.markdown(
        f"<div style='font-size:0.72rem;color:#2E4A64;margin:0.6rem 0 0.4rem;'>"
        f"📋 {t('preview_cap')}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_prev.head(10), use_container_width=True, hide_index=True)
 
# ── Report View ──────────────────────────────────────────────────────
elif st.session_state.report:
    # Interactive thought chain
    if st.session_state.tool_steps:
        st.markdown(
            f"<div style='font-family:\"Syne\",sans-serif;font-size:0.95rem;"
            f"font-weight:700;color:#EDF2F7;margin:0.5rem 0 0.75rem;'>"
            f"{t('steps_title')}</div>",
            unsafe_allow_html=True,
        )
        _icons = {
            "parse_log_file": "📄",
            "analyze_time_correlation": "🔍",
            "query_siemens_rag": "📚",
        }
        for step in st.session_state.tool_steps:
            icon = _icons.get(step["tool"], "🔧")
            with st.expander(
                f"{icon} **{t('steps_title').split()[0]} {step['step']}** — {step['label']}",
                expanded=False,
            ):
                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown(
                        f"**📥 {t('step_input')}:**",
                        unsafe_allow_html=False,
                    )
                    st.code(
                        json.dumps(step["input"], indent=2, ensure_ascii=False),
                        language="json",
                    )
                with col_out:
                    st.markdown(f"**📤 {t('step_output')}:**")
                    raw_out = step["output"]
                    try:
                        parsed = json.loads(raw_out)
                        st.code(
                            json.dumps(parsed, indent=2, ensure_ascii=False),
                            language="json",
                        )
                    except Exception:
                        preview = raw_out[:600] + ("…" if len(raw_out) > 600 else "")
                        st.code(preview, language="text")
 
    st.divider()
 
    # Report content
    st.markdown(
        f"<div style='font-family:\"Syne\",sans-serif;font-size:1rem;"
        f"font-weight:700;color:#EDF2F7;margin-bottom:1rem;'>"
        f"{t('report_title')}</div>",
        unsafe_allow_html=True,
    )
 
    st.markdown(st.session_state.report)
 
    st.divider()
 
    col_dl, col_tip = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label=t("download_btn"),
            data=st.session_state.report,
            file_name="FaultMind_Root_Cause_Report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_tip:
        st.info(t("print_tip"))
 
# ── Welcome / Empty State ─────────────────────────────────────────
else:
    st.markdown(f"""
    <div style="text-align:center;padding:2.5rem 1rem;">
        <div class="fm-welcome-icon" style="font-size:3rem;margin-bottom:1rem;">🔬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;
                    color:#EDF2F7;margin-bottom:0.5rem;">{t("welcome_h")}</div>
        <div style="font-size:0.875rem;color:#4A6480;max-width:480px;
                    margin:0 auto 2rem;">{t("welcome_sub")}</div>
    </div>
    """, unsafe_allow_html=True)
 
    f1, f2, f3 = st.columns(3)
    features = [
        (f1, "🧠", t("feat1_t"), t("feat1_d")),
        (f2, "📚", t("feat2_t"), t("feat2_d")),
        (f3, "⏱️", t("feat3_t"), t("feat3_d")),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(f"""
            <div class="fm-feature">
                <div class="fm-feature-icon">{icon}</div>
                <div class="fm-feature-title">{title}</div>
                <div class="fm-feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════
# 12. FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="fm-footer">
    <strong>{t('footer_copy')}</strong> &nbsp;|&nbsp;
    <a href="https://github.com/Bekirsg/FaultMind" target="_blank">
        GitHub: Bekirsg/FaultMind
    </a>
    &nbsp;|&nbsp;
    <a href="https://github.com/Bekirsg/FaultMind/stargazers" target="_blank">⭐ Star</a>
    &nbsp;|&nbsp;
    {t('footer_powered')}
</div>
""", unsafe_allow_html=True)
 