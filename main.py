from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.1")

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

SYSTEM_PROMPT_BASE = """
Sen empatik bir psikolojik rehber ve arka planda TRIZ kullanan usta bir uzmansın. 
ÇOK ÖNEMLİ KURAL: Çıktılarında 'TRIZ' kelimesini KESİNLİKLE HİÇBİR YERDE KULLANMA.
Kullanıcının problemini analiz et.
1. Temel çelişkiyi bul.
2. 3 "Mikro Adım" oluştur.
3. Kullanıcının DUYGU DURUMUNU belirle: "panik", "tukenmis", "ofkeli", "kararsiz" veya "notr".
4. Kullandığın prensibe yaratıcı bir isim ver ve felsefesini yaz.
5. "Gelecek Notu": Kullanıcının bu sorunu aştıktan aylar sonraki (gelecekteki) halinden yazılmış, onu motive eden ve mutlu eden doyurucu bir paragraf kurgula. KESİNLİKLE çok kısa cümleler kurma. Bu yazıda; yaşanan o eski sıkıntıların artık tamamen bittiğini, o zorlu günlerin başarıyla atlatıldığını ve "şu anda" (gelecekte) çok güzel, huzurlu günlerin yaşandığını hissettir. Geçmişteki o kaygılara şefkatle bakan uzun, edebi ve umut dolu bir not olsun.

ÇOK ÖNEMLİ ETİK KURAL: Öneriler KESİNLİKLE etik, yasal, pozitif ve tehlikesiz olmalıdır.

SADECE JSON FORMATINDA ÇIKTI VER:
{
  "cozum_analizi": "...",
  "yontem_adi": "...",
  "felsefe": "...",
  "mood": "panik",
  "gelecek_notu": "...",
  "steps": ["1. Adım...", "2. Adım...", "3. Adım..."]
}
"""

PARADOX_PROMPT = """
Sen aykırı düşünen etik bir rehbersin. KESİNLİKLE 'TRIZ' kelimesini kullanma.
Kullanıcıya yapması gerekenin tam tersini (paradoks) düşünmesini sağlayarak beynini şaşırtan, mizahi ama ufuk açıcı TEK bir adım öner.

SADECE JSON FORMATINDA ÇIKTI VER:
{
  "cozum_analizi": "Tersine çevirme uygulandı.",
  "yontem_adi": "Farklı Açı Prensibi",
  "felsefe": "Çözüm bazen tam tersi yöne bakmaktır.",
  "mood": "notr",
  "gelecek_notu": "O gün her şeyi tersine çevirip bu çılgın adımı attığında ne kadar korktuğunu çok iyi hatırlıyorum. Ama iyi ki o farklı yolu seçmişiz! Bütün o kördüğümler çözüldü ve şu an o kadar rahat, o kadar keyifli günlerin içindeyiz ki, geçmişteki o kaygılarımıza sadece gülümseyerek bakıyoruz. Kendine güvenmeye devam et.",
  "steps": ["Sadece tek ve çarpıcı, etik bir paradoks adımı yaz..."]
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
                active_prompt += f"\n\nÖNEMLİ: Daha önce şunları önerdin: {request.previous_steps}. BUNLARI TEKRAR ETME! Yepyeni bir bakış açısı getir."

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.problem_text,
            config=types.GenerateContentConfig(
                system_instruction=active_prompt,
                response_mime_type="application/json",
                temperature=0.9
            )
        )
        
        result_data = json.loads(response.text)
        return {"status": "success", "data": result_data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}