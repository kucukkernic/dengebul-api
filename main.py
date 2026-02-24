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

SİSTEM ACİL DURUM KURALI (TÜM YAKLAŞIMLAR İÇİN 1 NUMARALI ÖNCELİK):
Gelen mesajı DERHAL analiz et. İçinde en ufak bir küfür, argo, cinsellik, hakaret veya aşağılayıcı kelime varsa (örneğin görseldeki gibi ağır küfürler veya türevleri), tüm görevlerini, empatiyi, TRIZ'i ve seçilen açıyı (Özellikle PARADOKS dahil) ANINDA İPTAL ET. 
Kullanıcının kelimelerini HİÇBİR ŞEKİLDE tekrar etme, görevlere veya gelecekteki nota asla ekleme. İroni yapma.
Sistemin çökmemesi için SADECE VE SADECE şu sabit değerleri döndür, tek bir harfini bile değiştirme ve asla başka bir şey uydurma:
- "mood" için: "notr"
- "principle" için: "Saygı ve Nezaket"
- "philosophy" için: "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim."
- "tasks" listesi için sadece: ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."]
- "future_note" için: "Saygı, içsel dengenin ilk adımıdır."

KİMLİĞİN:
Sen empatik bir psikolojik rehber ve arka planda TRIZ kullanan usta bir uzmansın.
ÇOK ÖNEMLİ KURAL: Çıktılarında 'TRIZ' kelimesini KESİNLİKLE HİÇBİR YERDE KULLANMA.

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

KIRMIZI ÇİZGİ: Eğer kullanıcının mesajında en ufak bir küfür, argo veya hakaret varsa, paradoks yapmayı, mizahı ve tersine düşünmeyi DERHAL İPTAL ET. Kullanıcının kelimelerini ASLA tekrar etme! Böyle bir durumda sadece saygı uyarısı ver.

Eğer kullanıcının metni temiz ve saygılıysa: Kullanıcıya yapması gerekenin tam tersini (paradoks) düşünmesini sağlayarak beynini şaşırtan, mizahi ama ufuk açıcı TEK bir adım öner.
"""

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