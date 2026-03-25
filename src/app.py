import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# Çevresel değişkenleri yükle
load_dotenv()

# Sayfa Ayarları
st.set_page_config(page_title="FaultMind - AI SCADA Asistanı", page_icon="⚙️", layout="centered")

# CSS ile Endüstriyel Tema Dokunuşları
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #00FFCA; }
    .stAlert { background-color: #1E2329; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ FaultMind: Agentic Arıza Tespit Asistanı")
st.markdown("*Siemens S7-1500 PLC/SCADA sistemleri için kanıta dayalı (RAG) kök neden analizi.*")

# ──────────────────────────────────────────────
# VERİTABANI YÜKLEME (CACHE İLE HIZLANDIRILMIŞ)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="AI Beyni (RAG) Belleğe Yükleniyor...")
def load_rag_system():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Anthropic(model="claude-3-haiku-20240307", temperature=0.0)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    storage_path = os.path.join(project_root, "storage_native")
    
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)

query_engine = load_rag_system()

# ──────────────────────────────────────────────
# ARAYÜZ VE DÜŞÜNCE ZİNCİRİ (AGENTIC WORKFLOW)
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader("SCADA/PLC Alarm Logunu Yükleyin (.csv)", type=["csv"])

if uploaded_file is not None:
    if st.button("🚀 Analizi Başlat", use_container_width=True):
        
        # st.status ile Düşünce Zinciri (Chain of Thought) Animasyonu
        with st.status("Agentic İş Akışı Başlatıldı...", expanded=True) as status:
            
            # Adım 1: Log Parse
            st.write("📄 Log dosyası parse ediliyor (Pandas)...")
            df = pd.read_csv(uploaded_file)
            time.sleep(1) # Sadece UI animasyonu için ufak bekleme
            
            # Adım 2: Zaman Korelasyonu
            st.write("🔍 Zaman damgaları analiz ediliyor ve korelasyon kuruluyor...")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values(by='Timestamp')
            first_error = df.iloc[0] # Kök neden adayı
            last_error = df.iloc[-1] # Sonuç (Örn: CPU Stop)
            time.sleep(1)
            
            # Adım 3: Prompt Hazırlığı ve RAG Sorgusu
            st.write("📚 Kök neden için Siemens RAG veritabanı sorgulanıyor...")
            prompt = f"""
            Sen uzman bir Siemens otomasyon mühendisisin. Aşağıdaki SCADA log zincirini analiz et:
            - İlk Hata (Kök Neden Adayı): {first_error['Error_Code']} - {first_error['Message']} (Cihaz: {first_error['Device']})
            - Son Durum: {last_error['Error_Code']} - {last_error['Message']}
            
            Kendi veri tabanındaki (RAG) resmi dokümanları kullanarak:
            1. Bu arıza zinciri tam olarak ne anlama geliyor?
            2. Sahadaki mühendis CPU'yu tekrar RUN moduna almak için sırasıyla hangi adımları izlemeli?
            
            Cevabını Markdown formatında, net, profesyonel ve Türkçe olarak ver. Eğer RAG veritabanında bu hata kodu yoksa, "Dokümantasyonda bulunamadı, uzman müdahalesi gerekli" de (Halüsinasyon yapma).
            """
            
            # Adım 4: Yapay Zeka Sentezi
            st.write("🧠 Claude 3 Haiku verileri sentezliyor ve çözüm planı üretiyor...")
            response = query_engine.query(prompt)
            
            status.update(label="✅ Analiz Tamamlandı! Rapor Hazır.", state="complete", expanded=False)
        
        # ──────────────────────────────────────────────
        # SONUÇ RAPORU (UI GÖSTERİMİ)
        # ──────────────────────────────────────────────
        st.success("Arıza Kök Nedeni Başarıyla Tespit Edildi!")
        
        with st.expander("Ham Log Verisi (İlk 5 Satır)"):
            st.dataframe(df.head())
            
        st.markdown("---")
        st.markdown("### 📋 FaultMind Teşhis Raporu")
        st.markdown(response.response)
        
        # Rapor İndirme Butonu
        st.download_button(
            label="💾 Raporu İndir (.md)",
            data=response.response,
            file_name="FaultMind_Rapor.md",
            mime="text/markdown",
            use_container_width=True
        )