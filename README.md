
# ⚙️ FaultMind: Agentic PLC/SCADA Arıza Tespit Asistanı

**FaultMind**, Siemens S7-1200/1500 sistemlerinde yaşanan zincirleme arızaların **kök nedenini otonom olarak tespit eden** açık kaynaklı, şeffaf ve düşük maliyetli bir AI asistanıdır.

Sahadaki otomasyon mühendisleri ve bakım teknisyenleri için geliştirildi:  
Yüzlerce satırlık PLC/SCADA logunu **1 dakikada** analiz eder, zaman korelasyonu kurar ve **adım adım müdahale planı** sunar.

🔗 **Canlı Demo** → [FaultMind Demo](https://faultmind-i9ee6rjkerjqqggpbcznh5.streamlit.app/)  
🐙 **Geliştirici** → [Bekir Samet Güzlek](https://www.linkedin.com/in/bekirsametguzlek/) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/bekirsametguzlek/)

---

## 🌟 Neden FaultMind?

- **🕵️‍♂️ Otonom Kök Neden Analizi**  
  Karmaşık logları tarar, olayların zaman sırasını kurar ve “İlk patlayan nokta neresiydi?” sorusunu net cevaplar.

- **💡 Tam Şeffaf Düşünce Zinciri**  
  Yapay zekanın her adımını (log parse → zaman analizi → RAG sorgusu → karar) arayüzde canlı olarak izleyebilirsiniz.

- **🛡️ Güvenilir ve Kanıta Dayalı**  
  Resmi Siemens teknik dokümanlarından beslenen RAG sistemiyle çalışır. Bilmediği bir hata kodu gördüğünde uydurmaz, “Uzman müdahalesi gerekli” der.

- **⚡ Hafif ve Tak-Çalıştır**  
  Ağır veritabanı veya karmaşık kurulum gerektirmez. Her Windows makinesinde minimum kaynakla anında çalışır.

---

## 🏗️ Nasıl Çalışır?

1. **Yükle** → Fabrikadan aldığınız arıza logunu (.csv) arayüze sürükleyin.  
2. **Analiz Et** → FaultMind logu parse eder, zaman korelasyonu yapar ve RAG ile dokümanlarda arama yapar.  
3. **Çöz** → Kök neden hipotezi + sahada yapılması gereken **adım adım müdahale planını** size sunar.

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.9 veya üzeri
- Anthropic API Anahtarı ([console.anthropic.com](https://console.anthropic.com/))

### Kurulum Adımları

```bash
# 1. Repoyu klonlayın
git clone https://github.com/Bekirsg/FaultMind.git
cd FaultMind

# 2. Sanal ortam oluşturun (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Gerekli paketleri kurun
pip install -r requirements.txt
```

### Yapılandırma

1. **PDF Dokümanlarını Ekleyin**  
   `docs/` klasörüne kendi indirdiğiniz Siemens S7-1500/1200 System Manual ve Diagnostics PDF’lerini koyun.  
   (Telif hakları nedeniyle bu dosyalar repoda yer almamaktadır.)

2. **API Anahtarını Ayarlayın**  
   Ana dizinde `.env` dosyası oluşturun ve içine şu satırı ekleyin:

   ```env
   ANTHROPIC_API_KEY=sk-ant-api...
   ```

3. **RAG Veritabanını Oluşturun** (tek seferlik)
   ```bash
   python src/build_rag.py
   ```

4. **Uygulamayı Başlatın**
   ```bash
   streamlit run src/app.py
   ```

Tarayıcınızda FaultMind arayüzü otomatik olarak açılacaktır.

---

## ⚖️ Yasal Uyarı

Bu proje bağımsız, açık kaynaklı bir topluluk çalışmasıdır.  
**Siemens**, **S7-1200** ve **S7-1500** Siemens AG’nin tescilli ticari markalarıdır.  
FaultMind’ın Siemens AG ile hiçbir resmi bağı, sponsorluğu veya ortaklığı bulunmamaktadır.