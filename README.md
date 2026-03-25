
# ⚙️ FaultMind: AI-Powered PLC/SCADA Diagnostic Assistant



![FaultMind Demo](assets/demo.gif)**FaultMind**, endüstriyel tesislerde (Siemens S7-1200/1500) yaşanan zincirleme otomasyon arızalarının kök nedenini otonom olarak bulan açık kaynaklı bir yapay zeka asistanıdır.



Sahadaki mühendislerin arıza ararken harcadığı saatleri, yapay zekanın **1 dakikalık** kesin ve kanıta dayalı analizine dönüştürür.



🔗 **Canlı Demo:** [FaultMind Demo Sitesi](https://your-streamlit-app.streamlit.app) 

🐙 **Geliştirici:** [Bekir Samet Güzlek](https://github.com/yourusername)



---## 🌟 Neden FaultMind?* **🕵️‍♂️ Otonom Kök Neden Analizi:** Yüzlerce satırlık karmaşık SCADA loglarını tarar, olayların zaman akışını (zaman korelasyonu) kurar ve "İlk patlayan nokta neresiydi?" sorusunu cevaplar.* **💡 Şeffaf Düşünce Zinciri:** Yapay zekanın "kara kutu" olmasını engeller. Asistanın logları nasıl okuduğunu, hangi dokümana başvurduğunu ve nasıl karar verdiğini arayüzde adım adım izlersiniz.* **🛡️ Güvenilir Bilgi (Sıfır Tahmin):** Sadece resmi Siemens System Manual dokümanlarından beslenir. Bilmediği bir kod gördüğünde uydurmaz, uzman müdahalesi talep eder.* **⚡ Tak-Çalıştır Endüstriyel Mimari:** Ağır veritabanları gerektirmez. Minimum sistem gereksinimiyle her Windows/Endüstriyel PC'de anında çalışır.



---## 🏗️ Nasıl Çalışır?1. **Yükle:** Fabrikadan alınan arıza logunu (.csv) sisteme sürükleyin.2. **Analiz Et:** FaultMind veriyi okur, zaman damgalarını sıralar ve asıl donanım/haberleşme problemini tespit eder.3. **Çöz:** Resmi Siemens dokümanlarına göre "Şu an sahada yapılması gereken ilk 3 işlem" listesini size sunar.



---## 🚀 Hızlı Başlangıç



Sistemi kendi bilgisayarınızda anında ayağa kaldırmak için:```bash

# Repoyu klonlayın ve klasöre girin

git clone [https://github.com/yourusername/FaultMind.git](https://github.com/yourusername/FaultMind.git)

cd FaultMind



# Gerekli kütüphaneleri kurun

pip install -r requirements.txt



# Veritabanını tek seferlik oluşturun (docs/ klasöründeki PDF'lerden)

python src/build_rag.py



# Arayüzü başlatın!

streamlit run src/app.py

(Not: Projenin çalışması için ana dizine .env dosyası açıp ANTHROPIC_API_KEY=sk-... anahtarınızı girmeniz gerekmektedir.)

🛠️ Tech Stack

Dil & Arayüz: Python, Streamlit, Pandas

AI Core: Anthropic Claude 3 (Agentic Logic), LlamaIndex (Native Storage)

Endüstriyel veri gizliliğini (NDA) korumak amacıyla projede gerçek fabrika verileri değil, yapay zeka ile üretilmiş sentetik arıza logları (sample_log.csv) kullanılmaktadır.