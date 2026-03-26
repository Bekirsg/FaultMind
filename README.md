
# ⚙️ FaultMind: Agentic PLC/SCADA Arıza Tespit Asistanı 



**FaultMind**, endüstriyel tesislerde (Siemens S7-1200/1500) yaşanan zincirleme otomasyon arızalarının kök nedenini otonom olarak bulan açık kaynaklı bir yapay zeka asistanıdır.



Sahadaki mühendislerin arıza ararken harcadığı saatleri, yapay zekanın **1 dakikalık** kesin ve kanıta dayalı analizine dönüştürür.



🔗 **Canlı Demo:** [FaultMind Demo Sitesi](https://faultmind-i9ee6rjkerjqqggpbcznh5.streamlit.app/)

🐙 **Geliştirici:** [Bekir Samet Güzlek](https://www.linkedin.com/in/bekirsametguzlek/) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/bekirsametguzlek)



---## 🌟 Neden FaultMind?-
**🕵️‍♂️ Otonom Kök Neden Analizi** – Yüzlerce satırlık karmaşık SCADA loglarını tarar, olayların zaman akışını (zaman korelasyonu) kurar ve "İlk patlayan nokta neresiydi?" sorusunu cevaplar.-
**💡 Şeffaf Düşünce Zinciri** – Yapay zekanın "kara kutu" olmasını engeller. Asistanın logları nasıl okuduğunu, hangi dökümana başvurduğunu ve nasıl karar verdiğini arayüzde adım adım izlersiniz.-
**🛡️ Güvenilir Bilgi (Sıfır Tahmin)** – Sistemin RAG (Retrieval-Augmented Generation) beyni, resmi teknik dokümanlardan beslenir. Bilmediği bir kod gördüğünde uydurmaz, uzman müdahalesi talep eder.-
**⚡ Tak-Çalıştır Endüstriyel Mimari** – Ağır veritabanları gerektirmez. Minimum sistem gereksinimiyle her Windows/Endüstriyel PC'de anında çalışır.



---## 🏗️ Nasıl Çalışır?
1. **Yükle** – Fabrikadan alınan arıza logunu (.csv) sisteme sürükleyin.
2. **Analiz Et** – FaultMind veriyi okur, zaman damgalarını sıralar ve asıl donanım/haberleşme problemini tespit eder.
3. **Çöz** – Teknik dökümanlara göre "Şu an sahada yapılması gereken ilk 3 işlem" listesini size sunar.



---## 🚀 Hızlı Başlangıç### Gereksinimler- Python 3.9 veya üzeri- [Anthropic API Anahtarı](https://console.anthropic.com/)

```markdown
### Kurulum Adımları

```bash
# Repoyu klonlayın
git clone [https://github.com/Bekirsg/FaultMind.git](https://github.com/Bekirsg/FaultMind.git)
cd FaultMind

# Sanal ortam oluşturun (Windows)
python -m venv venv
venv\Scripts\activate

# Gerekli kütüphaneleri kurun
pip install -r requirements.txt
```

### Yapılandırma ve Veritabanı (RAG) İnşası

1. **PDF Dokümanlarını Ekleyin:** Telif hakları gereği Siemens'e ait PDF dosyaları bu repoda bulunmamaktadır. Sistemi kullanmak için kendi indirdiğiniz Siemens S7-1500/1200 "System Manual" PDF'lerini `docs/` klasörüne koymanız gerekmektedir.
2. **API Anahtarını Ayarlayın:** Ana dizinde bir `.env` dosyası oluşturun ve içine Anthropic API anahtarınızı girin:
   ```env
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```
3. **RAG Veritabanını Oluşturun:** `docs/` klasörüne koyduğunuz PDF'lerden AI beynini inşa edin (Bu işlem tek seferliktir):
   ```bash
   python src/build_rag.py
   ```

### Uygulamayı Başlatın

```bash
streamlit run src/app.py
```
*Tarayıcınızda otomatik açılan sekmede FaultMind arayüzünü görebilirsiniz.*

⚖️ Yasal Uyarı / Disclaimer

Bu proje bağımsız, açık kaynaklı bir topluluk çalışmasıdır. 'Siemens', 'S7-1200' ve 'S7-1500' Siemens AG'nin tescilli ticari markalarıdır. FaultMind'ın Siemens AG ile hiçbir resmi bağı, sponsorluğu veya ortaklığı bulunmamaktadır.