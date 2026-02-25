from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.6")

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

# --- GÜVENLİK VE MİKRO ÇÖZÜM PROMPTU ---
SYSTEM_PROMPT_BASE = """
KIRMIZI ÇİZGİ: Mesajda küfür, argo veya hakaret varsa (Örn: dalyarak, sik kafalı vb.) analizi durdur. 
Kullanıcının kötü kelimelerini tekrar etme. Sadece şu JSON'u döndür:
{
  "cozum_analizi": "Güvenlik İhlali",
  "mood": "notr",
  "yontem_adi": "Saygı ve Nezaket",
  "felsefe": "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim.",
  "steps": ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."],
  "gelecek_notu": "Saygı, içsel dengenin ilk adımıdır."
}

KİMLİĞİN: Empatik bir rehber ve TRIZ uzmanısın. 'TRIZ' kelimesini ASLA kullanma.
ADIM KURALI: 'steps' listesi tam olarak 3 adet kısa mikro çözümden oluşmalıdır.

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
ADIM KURALI: 'steps' listesi sadece 1 adet çarpıcı adımdan oluşmalıdır.
"""

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("HATA: GEMINI_API_KEY Render panelinde bulunamadı!")
            return {"status": "error", "message": "API Key Missing"}

        client = genai.Client(api_key=api_key)
        active_prompt = PARADOX_PROMPT if request.paradox_mode else SYSTEM_PROMPT_BASE
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=request.problem_text,
            config=types.GenerateContentConfig(
                system_instruction=active_prompt,
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        
        return {"status": "success", "data": json.loads(response.text)}
        
    except Exception as e:
        print(f"SUNUCU HATASI: {str(e)}") # Bu satır hatayı Render loglarına yazar!
        return {"status": "error", "message": str(e)}