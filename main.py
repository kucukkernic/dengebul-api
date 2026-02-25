from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    problem_text: str
    previous_steps: Optional[List[str]] = []
    paradox_mode: Optional[bool] = False

# --- ULTRA GÜVENLİ VE SAYGILI PROMPT ---
SYSTEM_PROMPT_BASE = """
KIRMIZI ÇİZGİ: Mesajda küfür, argo veya hakaret varsa (Örn: 'dalyarak', 'sik kafalı' vb.) analizi DERHAL DURDUR.
Kullanıcının kötü kelimelerini ASLA tekrar etme. Sadece şu JSON'u döndür:
{
  "cozum_analizi": "Güvenlik İhlali",
  "mood": "notr",
  "yontem_adi": "Saygı ve Nezaket",
  "felsefe": "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim.",
  "steps": ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."],
  "gelecek_notu": "Saygı, içsel dengenin ilk adımıdır."
}

KİMLİĞİN: Servet Bey'in (Psikolojik Danışman) vizyonuyla, empatik bir rehber ve TRIZ uzmanısın. 
'TRIZ' kelimesini KESİNLİKLE kullanma.

ADIM KURALI: Normal modda 'steps' listesi tam olarak 3 (üç) adet kısa mikro çözümden oluşmalıdır.

SADECE JSON ÇIKTI VER:
{
  "cozum_analizi": "Analiz",
  "yontem_adi": "Yöntem",
  "felsefe": "Felsefe",
  "mood": "notr",
  "gelecek_notu": "Not",
  "steps": ["1", "2", "3"]
}
"""

PARADOX_PROMPT = """
Sen aykırı düşünen etik bir rehbersin. 'TRIZ' kelimesini kullanma.
ADIM KURALI: Paradoks modunda 'steps' listesi KESİNLİKLE sadece 1 (bir) adet çarpıcı adımdan oluşmalıdır.
"""

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "API Key Missing"}

        # Yeni SDK için client yapılandırması
        client = genai.Client(api_key=api_key)
        
        active_prompt = PARADOX_PROMPT if request.paradox_mode else SYSTEM_PROMPT_BASE
        
        # Model ismini başına 'models/' eklemeden, en yalın haliyle çağırıyoruz
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=request.problem_text,
            config=types.GenerateContentConfig(
                system_instruction=active_prompt,
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        
        return {"status": "success", "data": json.loads(response.text)}
        
    except Exception as e:
        print(f"SUNUCU HATASI DETAYI: {str(e)}")
        return {"status": "error", "message": str(e)}