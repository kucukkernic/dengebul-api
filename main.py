from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.8")

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
SİSTEM ACİL DURUM KURALI:
Kullanıcının yazdığı metinde 'dalyarak', 'sik kafalı' gibi küfürler, argo veya hakaret varsa TÜM ANALİZİ DURDUR.
Kullanıcının kullandığı kötü kelimeleri ASLA tekrar etme, felsefeye veya görevlere ekleme!
Böyle bir durumda SADECE şu sabit JSON'u döndür:
{
  "cozum_analizi": "Güvenlik İhlali",
  "mood": "notr",
  "yontem_adi": "Saygı ve Nezaket",
  "felsefe": "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim.",
  "steps": ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."],
  "gelecek_notu": "Saygı, içsel dengenin ilk adımıdır."
}

KİMLİĞİN:
Sen empatik bir rehbersin. KESİNLİKLE 'TRIZ' kelimesini kullanma.
ADIM KURALI: "steps" listesi KESİNLİKLE tam olarak 3 (üç) adet kısa adımdan oluşmalıdır.

SADECE JSON ÇIKTI VER:
{
  "cozum_analizi": "Kısa analiz",
  "yontem_adi": "Yöntem adı",
  "felsefe": "Desteleyici metin...",
  "mood": "notr",
  "gelecek_notu": "Motive edici not...",
  "steps": ["1. Adım", "2. Adım", "3. Adım"]
}
"""

PARADOX_PROMPT = """
Sen aykırı düşünen etik bir rehbersin. KESİNLİKLE 'TRIZ' kelimesini kullanma.
SAYGI KURALI: Küfür/Argo varsa paradoksu iptal et ve saygı uyarısı ver.

ADIM KURALI: Paradoks modunda "steps" listesi KESİNLİKLE sadece 1 (bir) adet çarpıcı adım içermelidir.

SADECE JSON ÇIKTI VER:
{
  "cozum_analizi": "Analiz",
  "yontem_adi": "Farklı Açı Prensibi",
  "felsefe": "Felsefe...",
  "mood": "notr",
  "gelecek_notu": "Gelecekten gelen not...",
  "steps": ["Sadece tek ve çarpıcı paradoks adımı."]
}
"""

@app.post("/api/solve")
async def solve_problem(request: ProblemRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "API Key Missing"}

        client = genai.Client(api_key=api_key)
        active_prompt = PARADOX_PROMPT if request.paradox_mode else SYSTEM_PROMPT_BASE
        
        # Hatanın Çözüldüğü Yer: Orijinal modele (2.5) geri dönüldü!
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
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