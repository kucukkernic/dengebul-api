from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.3")

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

# --- 1. ANA SİSTEM PROMPTU (3 ADIM KURALI EKLENDİ) ---
SYSTEM_PROMPT_BASE = """
SİSTEM ACİL DURUM KURALI:
Gelen mesajda küfür, argo veya hakaret varsa TÜM işlemleri iptal et. 
Kullanıcının kelimelerini tekrar etme. Sadece şu JSON'u döndür:
{
  "cozum_analizi": "Güvenlik İhlali",
  "mood": "notr",
  "yontem_adi": "Saygı ve Nezaket",
  "felsefe": "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim.",
  "steps": ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."],
  "gelecek_notu": "Saygı, içsel dengenin ilk adımıdır."
}

KİMLİĞİN VE GÖREVİN:
Sen empatik bir psikolojik rehber ve arka planda TRIZ kullanan usta bir uzmansın. 
ÇOK ÖNEMLİ: Çıktılarında 'TRIZ' kelimesini ASLA kullanma.

ADIM KURALI: "steps" listesi KESİNLİKLE tam olarak 3 (üç) adet, kısa ve uygulanabilir mikro çözümden oluşmalıdır. Ne eksik, ne fazla!

SADECE AŞAĞIDAKİ JSON FORMATINDA ÇIKTI VER:
{
  "cozum_analizi": "Kısa analiz",
  "yontem_adi": "Yöntem adı",
  "felsefe": "Felsefi destek metni...",
  "mood": "notr",
  "gelecek_notu": "Motive edici not...",
  "steps": ["1. Kısa mikro çözüm", "2. Kısa mikro çözüm", "3. Kısa mikro çözüm"]
}
"""

# --- 2. PARADOKS PROMPTU (TEK ADIM KURALI EKLENDİ) ---
PARADOX_PROMPT = """
Sen aykırı düşünen etik bir rehbersin. KESİNLİKLE 'TRIZ' kelimesini kullanma.
KÜFÜR/ARGO VARSA İPTAL ET VE SAYGI UYARISI VER.

ADIM KURALI: Paradoks modunda "steps" listesi KESİNLİKLE sadece 1 (bir) adet, çarpıcı ve ufuk açıcı adımdan oluşmalıdır.

SADECE AŞAĞIDAKİ JSON FORMATINDA ÇIKTI VER:
{
  "cozum_analizi": "Tersine çevirme uygulandı.",
  "yontem_adi": "Farklı Açı Prensibi",
  "felsefe": "Çözüm bazen tam tersi yöne bakmaktır.",
  "mood": "notr",
  "gelecek_notu": "Gelecekten gelen not...",
  "steps": ["Sadece tek ve çarpıcı, etik bir paradoks adımı."]
}
"""

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "API Anahtarı bulunamadı."}

        client = genai.Client(api_key=api_key)
        
        if request.paradox_mode:
            active_prompt = PARADOX_PROMPT
        else:
            active_prompt = SYSTEM_PROMPT_BASE
            if request.previous_steps and len(request.previous_steps) > 0:
                active_prompt += f"\n\nBUNLARI TEKRAR ETME: {request.previous_steps}. Yepyeni 3 adım getir."

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=request.problem_text,
            config=types.GenerateContentConfig(
                system_instruction=active_prompt,
                response_mime_type="application/json",
                temperature=0.7 # Yaratıcılığı biraz kısıp kurallara uymasını sağladık
            )
        )
        
        result_data = json.loads(response.text)
        return {"status": "success", "data": result_data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}