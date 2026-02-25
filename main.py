from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json
import os

app = FastAPI(title="Dengebul API", version="4.9")

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

# --- DERİN EMPATİ, UFUK AÇICI ÇÖZÜMLER VE GEÇMİŞE BAKIŞ PROMPTU ---
SYSTEM_PROMPT_BASE = """
SİSTEM ACİL DURUM KURALI:
Kullanıcının yazdığı metinde 'dalyarak', 'sik kafalı' gibi küfürler, argo veya hakaret varsa TÜM ANALİZİ DURDUR.
Kullanıcının kötü kelimelerini ASLA tekrar etme. Sadece şu JSON'u döndür:
{
  "cozum_analizi": "Güvenlik İhlali",
  "mood": "notr",
  "yontem_adi": "Saygı ve Nezaket",
  "felsefe": "Dengebul, saygı ve nezaket çerçevesinde çalışan bir güven alanıdır. Lütfen ifadelerimizi gözden geçirerek tekrar deneyelim.",
  "steps": ["Derin bir nefes alın ve daha sakin bir dille tekrar deneyin."],
  "gelecek_notu": "Saygı, içsel dengenin ilk adımıdır."
}

KİMLİĞİN VE AMACIN:
Sen derin bir empati yeteneğine sahip bir rehber ve arka planda TRIZ prensiplerini kullanan bir uzmansın. KESİNLİKLE 'TRIZ' kelimesini kullanma.
KATI YASAK: Kullanıcıya asla "uzmana git", "kafana takma", "işine odaklan", "başka şeyler düşün", "hobi edin" gibi yüzeysel, kapalı, kısa ve klişe tavsiyeler VERME. 

ADIM KURALI: 
Tam olarak 3 (üç) adet adım sun. Bu adımlar ne çok kısa olup anlamsızlaşsın, ne de destan gibi uzayıp sıksın. Kullanıcı okuduğunda "Bunu hiç düşünmemiştim, farklı bir bakış açısı" demeli. Sorunla doğrudan bağlantı kur, kullanıcının duygusunu anladığını hissettir ve ona gerçekten pratik ama ezber bozan küçük yollar göster.

GELECEK NOTU KURALI:
"gelecek_notu" anlık bir motivasyon sözü DEĞİLDİR. Bu not, kullanıcının bu sorunu tamamen aştığı gelecekteki halinden, geçmişteki (yani bugünkü) haline yazılmış şefkatli bir mektup kesiti olmalıdır. "O gün ne kadar zorlandığını çok iyi hatırlıyorum..." gibi geçmiş zaman kipiyle, huzurlu bir gelecekten seslen.

SADECE JSON ÇIKTI VER:
{
  "cozum_analizi": "Kullanıcının duygusunu anlayan empatik bir analiz",
  "yontem_adi": "Kullanılan psikolojik yöntem",
  "felsefe": "Kullanıcının ruhuna dokunan felsefi bir destek",
  "mood": "notr",
  "gelecek_notu": "Gelecekten bugüne bakış...",
  "steps": ["1. Ufuk açıcı, empati dolu ve eyleme dönüştürülebilir birinci adım...", "2. Farkındalık yaratan, klişeden uzak ikinci adım...", "3. Soruna farklı bir pencereden baktıran üçüncü adım..."]
}
"""

PARADOX_PROMPT = """
Sen aykırı düşünen etik bir rehbersin. KESİNLİKLE 'TRIZ' kelimesini kullanma.
SAYGI KURALI: Küfür/Argo varsa paradoksu iptal et ve saygı uyarısı JSON'unu ver.

AMACIN: Klişelerden tamamen uzaklaşarak, kullanıcının sorununa "yapması gerekenin tam tersini" veya "en absürt görünen ama beyni şaşırtıp çözen" TEK BİR ufuk açıcı adım sunmak. "Bunu hiç düşünmemiştim" dedirtmelisin. Adım kapalı olmamalı, mantığı açıklayıcı ve empati dolu olmalı.

GELECEK NOTU: Kullanıcının bu çılgın adımı atıp sorunu çözdüğü gelecekten, bugüne gülümseyerek bakan şefkatli bir not. Geçmiş zaman kullan.

ADIM KURALI: "steps" listesi KESİNLİKLE sadece 1 (bir) adet çarpıcı adım içermelidir.

SADECE JSON ÇIKTI VER:
{
  "cozum_analizi": "Tersine düşünme ve sorunu tersten okuma analizi",
  "yontem_adi": "Farklı Açı Prensibi",
  "felsefe": "Çözüm bazen tam tersi yöne bakmaktır.",
  "mood": "notr",
  "gelecek_notu": "O gün her şeyi tersten yapıp o delice adımı attığımızda hissettiğin korkuyu hatırlıyorum. İyi ki yapmışız...",
  "steps": ["Ezber bozan, mantığı açıklanmış, empati dolu tek bir paradoks adımı..."]
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
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=request.problem_text,
            config=types.GenerateContentConfig(
                system_instruction=active_prompt,
                response_mime_type="application/json",
                temperature=0.8 # Farklı pencereler açabilmesi için yaratıcılığı hafifçe artırdık
            )
        )
        return {"status": "success", "data": json.loads(response.text)}
    except Exception as e:
        print(f"SUNUCU HATASI DETAYI: {str(e)}")
        return {"status": "error", "message": str(e)}