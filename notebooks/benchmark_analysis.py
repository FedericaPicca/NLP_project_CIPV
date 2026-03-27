import json
import glob
import os
import time
import pandas as pd
from tqdm import tqdm
from core.graph_rag_engine import run_graph_rag_analysis 

# --- CONFIGURATION ---
CHATS_FOLDER = r"C:\Users\feder\OneDrive\Desktop\NLP\PROGETTO\CIPV_SIIA\DA PULIRE PER ADRIANO\outputs\chats" 
OUTPUT_EXCEL = "MultiModel_Benchmark_Results.xlsx" 
WINDOW_SIZE = 3  
STEP_SIZE = 2    

MODELS_TO_COMPARE = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b"
]

def get_generation_strategy(filename):
    if "single_prompt" in filename: return "Single_Prompt"
    elif "1_model" in filename: return "1_Model_Roleplay"
    elif "2_models" in filename: return "2_Models_Roleplay"
    return "Standard_Synthetic"

def normalize_text(text):
    if not text: return ""
    clean = text.replace("<think>", "").replace("</think>", "")
    return clean.strip()

def main():
    json_files = glob.glob(os.path.join(CHATS_FOLDER, "*.json"))
    batch_results = []
    MAX_WINDOWS = 15
    windows_processed = 0

    for filepath in tqdm(json_files, desc="Processing Chats"):
        if windows_processed >= MAX_WINDOWS: break

        filename = os.path.basename(filepath)
        strategy = get_generation_strategy(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_messages = data.get("conversation", data)
        formatted_dialogue = []
        for msg in raw_messages:
            content = normalize_text(msg.get('content', msg.get('text', '')))
            if content: formatted_dialogue.append(f"{msg.get('sender', 'Partner')}: {content}")

        for i in range(WINDOW_SIZE - 1, len(formatted_dialogue), STEP_SIZE):
            if windows_processed >= MAX_WINDOWS: break
            chat_context = " | ".join(formatted_dialogue[i - WINDOW_SIZE + 1 : i + 1])
            
            result_row = {"Strategy": strategy, "File_Source": filename, "Context_Window": chat_context}

            for model in MODELS_TO_COMPARE:
                result_row[f"Analysis_{model.replace('/', '_')}"] = run_graph_rag_analysis(chat_context, model_id=model)
                time.sleep(3) # Anti-Rate Limit
            
            batch_results.append(result_row)
            windows_processed += 1

    pd.DataFrame(batch_results).to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n[SUCCESS] Benchmark completato! Salvo in {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()