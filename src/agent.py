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

# =====================================================================
# 🛠️ TOOL FONKSİYONLARI
# =====================================================================

query_engine = None

def get_rag_engine():
    global query_engine
    if query_engine is None:
        print("   [Sistem] 📚 LlamaIndex Veritabanı RAM'e yükleniyor...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = LlamaAnthropic(model="claude-sonnet-4-6", temperature=0.0)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        storage_path = os.path.join(os.path.dirname(current_dir), "storage_native")
        
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3)
    return query_engine

def parse_log_file(file_path="data/sample_log.csv"):
    print(f"   [Araç Kullanımı] 📄 Log dosyası okunuyor: {file_path}")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(os.path.dirname(current_dir), file_path)
        df = pd.read_csv(full_path)
        df = df.head(20)
        return df.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": str(e)})

def analyze_time_correlation(alarms_json):
    print("   [Araç Kullanımı] ⏱️ Zaman korelasyonu analizi yapılıyor...")
    try:
        alarms = json.loads(alarms_json)
        if not alarms or "error" in alarms:
            return json.dumps({"error": "Geçerli alarm verisi bulunamadı."})
        
        alarms.sort(key=lambda x: datetime.strptime(x["Timestamp"], "%Y-%m-%d %H:%M:%S"))
        root_candidate = alarms[0]
        return json.dumps({
            "root_cause_candidate": root_candidate,
            "total_alarms_analyzed": len(alarms),
            "timeline_start": alarms[0]["Timestamp"],
            "timeline_end": alarms[-1]["Timestamp"]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

def query_siemens_rag(query_str):
    print(f"   [Araç Kullanımı] 🔍 RAG'a soruluyor: '{query_str}'")
    try:
        engine = get_rag_engine()
        response = engine.query(query_str)
        return str(response)
    except Exception as e:
        return json.dumps({"error": str(e)})

# =====================================================================
# 🧠 CLAUDE AJAN (AGENT) DÖNGÜSÜ
# =====================================================================

def run_faultmind_agent():
    print("\n" + "="*60)
    print("🚀 FAULTMIND AJANI BAŞLATILIYOR...")
    print("="*60)

    client = anthropic.Anthropic()
    
    tools = [
        {
            "name": "parse_log_file",
            "description": "Reads the factory CSV log file and returns the alarms.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the log file. Use 'data/sample_log.csv'."}
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "analyze_time_correlation",
            "description": "Analyzes the JSON alarm data to find the earliest root cause candidate based on timestamps.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alarms_json": {"type": "string", "description": "The JSON string of alarms obtained from parse_log_file."}
                },
                "required": ["alarms_json"]
            }
        },
        {
            "name": "query_siemens_rag",
            "description": "Queries the local Siemens documentation database (RAG) to understand specific error codes (like 16#xxxx) or diagnostic steps.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_str": {"type": "string", "description": "The specific question to ask the documentation, e.g., 'What does error code 16#02C7 mean?'"}
                },
                "required": ["query_str"]
            }
        }
    ]

    system_prompt = (
        "Sen FaultMind'sın. Sadece Siemens S7-1200/1500 PLC arızalarını inceleyen kıdemli bir endüstriyel uzmansın.\n"
        "GÖREVİN:\n"
        "1. Önce parse_log_file aracıyla logları oku.\n"
        "2. Sonra analyze_time_correlation aracıyla zaman akışını ve kök nedeni bul.\n"
        "3. Kök nedendeki veya önemli gördüğün Hata Kodlarını (Error_Code) query_siemens_rag aracıyla dokümantasyonda ara.\n"
        "4. En son, tüm bu verileri birleştirerek Markdown formatında profesyonel bir kök neden analiz raporu yaz.\n\n"
        "KATI KURAL: Teşhis yaparken SADECE query_siemens_rag aracından dönen bilgiyi kullan. Eğer bilgi yoksa "
        "'Dokümantasyonda bulunamadı, uzman müdahalesi gerekli' de. Asla halüsinasyon görme."
    )

    messages = [{"role": "user", "content": "Sistemde bir arıza var. Lütfen 'data/sample_log.csv' dosyasını inceleyip kök nedeni bul ve bana çözüm raporu sun."}]

    max_iterations = 6 # Bol bol sorsun diye limiti 6'ya çıkardık
    
    for i in range(max_iterations):
        print(f"\n🔄 [Döngü {i+1}] Claude Düşünüyor...")
        
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            temperature=0.0,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        # Eğer tool_use yoksa, rapor hazırdır
        if response.stop_reason != "tool_use":
            print("\n" + "="*60)
            print("✅ NİHAİ RAPOR HAZIRLANDI:")
            print("="*60)
            for block in response.content:
                if block.type == 'text':
                    print(block.text)
            break

        # SİHİRLİ DOKUNUŞ: Claude'un gönderdiği TÜM araç kullanımlarını bul (Paralel İşlem)
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        
        # Claude'un kendi isteğini mesaja ekle
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results_content = []

        # Her bir aracı sırayla çalıştır ve cevapları biriktir
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            
            print(f"🧠 [Düşünce Zinciri] Claude karar verdi: '{tool_name}' aracını kullanacak.")
            
            tool_result_content = ""
            if tool_name == "parse_log_file":
                tool_result_content = parse_log_file(**tool_input)
            elif tool_name == "analyze_time_correlation":
                tool_result_content = analyze_time_correlation(**tool_input)
            elif tool_name == "query_siemens_rag":
                tool_result_content = query_siemens_rag(**tool_input)
            else:
                tool_result_content = f"Error: Unknown tool '{tool_name}'"

            # Topladığımız her cevabı aynı paketin içine koyuyoruz
            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": tool_result_content
            })

        # Claude'un 5 sorusu varsa, 5 cevabı da aynı anda ona geri gönderiyoruz
        messages.append({
            "role": "user", 
            "content": tool_results_content
        })

    else:
        print("\n❌ Maksimum iterasyon sınırına (6) ulaşıldı. Döngü sonlandırıldı.")

if __name__ == "__main__":
    run_faultmind_agent()