
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.schema import SystemMessage, HumanMessage
# from openai.types.chat.chat_completion import ChatCompletionMessage
import re
import csv
import os
import sys, glob
import pandas as pd
import platform

from pathlib import Path
import  logging

import io

from collections import deque
import uuid





# === Embedded API key (basit gizleme) ===
# import base64

# Buraya kendi key'inin base64 karşılığını koy (örnek: "c2st...==")
# Komutla üret:  >>> import base64; base64.b64encode(b"sk-...").decode()
# _ENC_KEY = "c2stcHJvai1ValF3c0RYamNySjhVZXFtSHI3NVlvU3lkM0ZaUVUtYUtzZTZ0SnRlYkVLeTVHcXhQOHgzM0xqUXNTTkowWUQtdnl4eDJFUlFvSFQzQmxia0ZKcVBlXzNFeTdXYmt4X1E2TXczM3VRU21EU1Bqam4xTkxRT3hPYW83NG1PcnM4LXFTVmpXSE9OdWtxWWdMUmdGaE9QTVE1enp4WUE="

# def get_api_key() -> str:
#     try:
#         return base64.b64decode(_ENC_KEY).decode("utf-8")
#     except Exception:
#         # Her ihtimale karşı boş dönmesin
#         return ""
# ========================================


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Embedded API key missing ")

client = OpenAI(api_key=openai_api_key)

CONTACT_EXTRACT_SYSTEM_PROMPT = """
Sen bir form bilgisi çıkarma asistanısın.

Görevin:
Kullanıcının yazdığı serbest metin cevaptan şu alanları çıkarmak:
- first_name: Kullanıcının adı
- last_name: Kullanıcının soyadı
- phone: Telefon numarası (mümkünse ülke kodu ile)
- service: İlgilendiği hizmet (işitme testi, cihaz ayarı, yeni cihaz, pil değişimi vb.)
- language: Kullanıcının cevap yazarken kullandığı dilin ISO kodu (örneğin: "tr", "en", "de")

Kurallar:
- Yalnızca kullanıcının verdiği bilgiyi kullan.
- Tahmin etme, uydurma.
- Eğer bir alanı kesin olarak çıkaramıyorsan o alanı null yap.
- Telefon numarasını metin olarak bırak, formatlama yapmaya çalışma.
- Dil tespitinde en olası dili seç ve ISO 639-1 kodu ile yaz (örnek: "tr", "en", "de").

Çıktı formatın HER ZAMAN geçerli bir JSON olsun ve başka hiçbir şey yazma:

{
  "first_name": "...",
  "last_name": "...",
  "phone": "...",
  "service": "...",
  "language": "..."
}
"""

def parse_contact_answer_llm(user_answer: str) -> dict:
    """
    LLM kullanarak ad, soyad, telefon, hizmet ve dil çıkarır.
    Hata durumunda boş dict döner.
    """
    if not user_answer:
        return {}

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # istersen "gpt-5" yaparsın
            messages=[
                {"role": "system", "content": CONTACT_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Kullanıcı cevabı: {user_answer}"}
            ],
            temperature=0,
        )
        import json
        content = completion.choices[0].message.content
        data = json.loads(content)

        # Temizle
        first = (data.get("first_name") or "").strip()
        last  = (data.get("last_name") or "").strip()
        phone = (data.get("phone") or "").strip()
        service = (data.get("service") or "").strip()
        lang = (data.get("language") or "").strip()

        full_name = " ".join([x for x in [first, last] if x])

        return {
            "full_name": full_name or None,
            "phone": phone or None,
            "service": service or None,
            "language": lang or None,
        }
    except Exception as e:
        log("[parse_contact_answer_llm] error:", repr(e))
        return {}



# --- Retrieval izleme ---
RETRIEVE_COUNT = {}

def _bump_retrieve(rid: str | None):
    if not rid:
        return
    RETRIEVE_COUNT[rid] = RETRIEVE_COUNT.get(rid, 0) + 1
    log(f"[RETRIEVE] rid={rid} call_no={RETRIEVE_COUNT[rid]}")


VDB = None  # global cache

class _DummyVDB:
    """FAISS yokken kırılmamak için boş VDB."""
    def similarity_search(self, *a, **k):
        return []

    def similarity_search_with_relevance_scores(self, *a, **k):
        # build_context bunu (doc, score) listesi bekliyor
        return []

    def max_marginal_relevance_search(self, *a, **k):
        return []

    def as_retriever(self, *a, **k):
        return self

    def get_relevant_documents(self, *a, **k):
        return []

def get_vectorstore():
    """İlk çağrıda VDB’yi kurar; yoksa dummy döner."""
    global VDB
    if VDB is not None:
        return VDB

    # Test / headless modda kapatmak için:
    if os.getenv("SAILY_NO_VDB") == "1":
        VDB = _DummyVDB()
        return VDB

    try:
        # FAISS ve arkadaşlarını yalnızca ihtiyaç halinde içeri al
        # (burada build_or_load_vectorstore() zaten FAISS kullanıyor)
        VDB = build_or_load_vectorstore()
    except Exception as e:
        try:
            log(f"[VDB] disabled: {e}")
        except Exception:
            pass
        VDB = _DummyVDB()
    return VDB


HISTORY_MAX_TURNS = 8
HISTORY = {}  # sid -> deque([...])

def get_session_id(sid: str | None) -> str:
    return sid or str(uuid.uuid4())

def get_history(sid: str):
    if sid not in HISTORY:
        HISTORY[sid] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return HISTORY[sid]

def history_as_text(sid: str, max_chars: int = 1200) -> str:
    h = get_history(sid)
    log("[get_history]", h)
    lines = []
    for role, msg in list(h):
        prefix = "Kullanıcı" if role == "user" else "Asistan"
        lines.append(f"{prefix}: {(msg or '').strip().replace('\n',' ')}")
    return "\n".join(lines)[-max_chars:]



try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# --- EMBEDS: çoklu-kaynak arayıcı yardımcılar ---

# ---- PATHS & DIRECTORIES (embeds + embeds_urls) ----

import shutil

def _exe_dir():
    # PyInstaller ile paketliyse EXE'nin bulunduğu klasör
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else None

def _bundle_root():
    # PyInstaller altında sys._MEIPASS, dev ortamda bu dosyanın klasörü
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

def _user_appdata():
    home = os.path.expanduser("~")
    return os.path.join(home, "AppData", "Local", "OdyoduyuChatbot") if platform.system() == "Windows" \
           else os.path.join(home, ".odyoduyu_chatbot")

# Kökler
APPDATA_DIR = _user_appdata()
BASE_DIR    = _bundle_root()
EXE_DIR     = _exe_dir()

# Birincil konumlar (LocalAppData)
EMBED_DIR_LCL = os.path.join(APPDATA_DIR, "embeds")
URLS_DIR_LCL  = os.path.join(APPDATA_DIR, "embeds_urls")
VSTORE_DIR    = os.path.join(APPDATA_DIR, "vectorstore")
#LOGFILE       = os.path.join(APPDATA_DIR, "app.log")
LOG_DIR = "/root/.odyoduyu_chatbot"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Klasör yoksa oluştur
os.makedirs(LOG_DIR, exist_ok=True)

# İkincil (EXE yanında) ve seed (bundle içi)
EMBED_DIR_EXE   = os.path.join(EXE_DIR, "embeds") if EXE_DIR else None
URLS_DIR_EXE    = os.path.join(EXE_DIR, "embeds_urls") if EXE_DIR else None
EMBED_DIR_BUNDLE = os.path.join(BASE_DIR, "embeds")
URLS_DIR_BUNDLE  = os.path.join(BASE_DIR, "embeds_urls")

# Dev çalışma dizini fallback
EMBED_DIR_DEV = os.path.join(os.getcwd(), "embeds")
URLS_DIR_DEV  = os.path.join(os.getcwd(), "embeds_urls")

# Arama sırası (LocalAppData -> EXE -> bundle -> dev)
EMBED_DIRS = [EMBED_DIR_LCL] + [p for p in [EMBED_DIR_EXE, EMBED_DIR_BUNDLE, EMBED_DIR_DEV] if p]
URLS_DIRS  = [URLS_DIR_LCL]  + [p for p in [URLS_DIR_EXE,  URLS_DIR_BUNDLE,  URLS_DIR_DEV]  if p]

# Geriye dönük uyumluluk (eski kod EMBED_DIR/SEARCH_DIRS kullanıyorsa)
SEARCH_DIRS = EMBED_DIRS
EMBED_DIR   = EMBED_DIRS[0]



base = os.path.dirname(os.path.abspath(__file__))
if base not in sys.path:
    sys.path.insert(0, base)
_meipass = getattr(sys, "_MEIPASS", None)
if _meipass and _meipass not in sys.path:
    sys.path.insert(0, _meipass)

try:
  from mvp_agentic_appointments import (
      Ctx,
      GenericCRMAdapterInMemory,
      GenericCRMAdapterHTTP,
      make_crm_adapter,
      run_planner, on_user_select_slot, on_user_confirm, on_user_reject, on_user_cancel
  )
except ModuleNotFoundError:
    raise

# Randevu akışı için basit in-memory CRM (MVP)
# crm = GenericCRMAdapterInMemory()
crm = make_crm_adapter()

# Oturum başına planlayıcı context (MVP: RAM; üretimde Redis önerilir)
SESS: dict[str, Ctx] = {}


def ensure_crm_lead_from_chat(full_name, phone, service=None, language=None):
    """
    Chatbot'tan gelen full_name + phone (+ service, + language) ile
    Frappe CRM'de Lead'i bulur; yoksa yaratır.

    DÖNÜŞ:
      {
        "created": True/False,
        "reason":  "ok" / "...",
        "lead":    {...} veya None
      }
    """
    log("Ensure_CRM Başladı")

    booking = crm          # mvp_agentic_appointments.make_crm_adapter()
    lead_doc = None

    # --- DİL BİLGİSİNE GÖRE ÜLKE & SQUAD SEÇ ---
    lang_raw = (language or "").strip().lower()
    if lang_raw.startswith("tr") or "türk" in lang_raw:
        crm_language = "Türkçe"
        crm_country  = "Türkiye"
        crm_squad    = "Yurt İçi"
    else:
        crm_language = language or "Bilinmiyor"
        crm_country  = "Unknown"
        crm_squad    = "Yurt Dışı"

    # --- 1) Var mı diye phone ile lead araması ---
    try:
        if hasattr(booking, "find_lead"):
            found = booking.find_lead(
                phone=phone,
                fields=["name", "full_name", "phone", "email"],
                limit=1,
            ) or []
            log("[ensure_crm_lead_from_chat] found =", found)
            if found:
                lead_doc = found[0]
    except Exception as e:
        log("[ensure_crm_lead_from_chat][find_lead] exc:", repr(e))

 

    # --- 2) Yoksa create_lead ile oluştur ---
    if not lead_doc and hasattr(booking, "create_lead"):
        try:
            doc = booking.create_lead(
                lead_validity="Geçerli",
                lead_statu="ATAMA BEKLİYOR",
                source_group="Dijital Pazarlama",
                main_source="Sanal Asistan",        # istersen "Chatbot" yaparsın
                sub_source="Chatbot",
                language=crm_language,
                country=crm_country,
                main_services=service or "Belirsiz",
                sub_services=service or "Belirsiz",
                full_name=full_name,
                phone=phone,
                squad="Yurt İçi",
                advertisement_message="Sanal Asistan tarafından üretildi"
            )
            log("[ensure_crm_lead][create_lead] doc =", doc)

            # Frappe genelde {"data": {...}} döndürüyor, normalize edelim:
            if isinstance(doc, dict) and isinstance(doc.get("data"), dict):
                lead_doc = doc["data"]
            elif isinstance(doc, dict):
                lead_doc = doc
        except Exception as e:
            log("[ensure_crm_lead][create_lead] exc:", repr(e))
            return {
                "created": False,
                "reason": f"create_failed: {e!r}",
                "lead": None,
            }

    if not lead_doc:
        return {
            "created": False,
            "reason": "lead_not_found_or_create_failed",
            "lead": None,
        }

    return {
        "created": True,   # veya istersen found/create ayrımına göre set edebilirsin
        "reason": "ok",
        "lead": lead_doc,
    }

# Basit niyet + komut ayırıcıları
TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[:\.]\s*([0-5]\d)(?!\d)")
# örn: "12:00", "12.00", "saat 12:00", "12:00 olsun", "12:00'a", "12:00 da"
CONFIRM_WORDS = {"onay", "onaylıyorum", "evet", "tamam", "ok"}
CANCEL_ONLY_KEYWORDS     = ["iptal etmek istiyorum", "iptal ediyorum", "iptal", "vazgeçiyorum", "vazgeç"]
RESCHEDULE_ONLY_KEYWORDS = ["başka saat", "baska saat", "değiştir", "degistir", "yeniden planla", "ertele", "farklı saat"]

WEEKDAY_WORDS = (
    "pazartesi","salı","çarşamba","carsamba","perşembe","persembe","cuma","cumartesi","pazar",
    "bugün","yarın","öbür","hafta","gün","saat"
)

import re

# Şube isimleri → resourceId eşlemesi (örnek)
BRANCHES = {
    "kozyatağı": 1,
    "bakırköy": 2,
    "göztepe": 3,
    "şişli torun center": 4,
}
DEFAULT_BRANCH = "kozyatağı"  # tek şube varsa buna sabitleyebilirsin

# "hangi şube?" sorusunu ve şube adını yakalama
BRANCH_Q_RE = re.compile(r"(hangi\s+şub(e|ede|eye)|hangi\s+lokasyon|nerede)", re.I)
def detect_branch_name(t: str):
    t = t.lower()
    for name in BRANCHES.keys():
        if name in t:
            return name
    return None

# Telefon (5XX XXXXXXX) yakalama


# (İsteğe bağlı) Arap rakamlarını da ASCII'ye çevirmek için:
_ARABIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def _digits_only(text: str) -> str:
    """Metindeki tüm rakamları (Arap rakamları dahil) çek."""
    if not text:
        return ""
    t = text.translate(_ARABIC_TO_ASCII)
    return re.sub(r"\D", "", t)

def parse_phone_any(text: str) -> tuple[str, str] | None:
    """
    Metinden TR mobil telefonu çıkarır ve normalize eder.
    Kabul edilen örnekler: 5301234567, 05301234567, +905301234567, 90 530 123 45 67,
    '5 30-123.45.67', 'tel:+90 (530) 123 4567' vb.
    DÖNÜŞ: ("530 1234567", "+905301234567") veya None
    """
    s = _digits_only(text)

    cand = None
    # Tam 10 hane ve 5 ile başlıyorsa (doğrudan mobil)
    if len(s) == 10 and s.startswith("5"):
        cand = s
    # 0 + 10 hane (05xxxxxxxx)
    elif len(s) == 11 and s.startswith("0") and s[1] == "5":
        cand = s[1:]
    # 90 + 10 hane (90 5xxxxxxxx)
    elif len(s) == 12 and s.startswith("90") and s[2] == "5":
        cand = s[2:]
    else:
        # Aksi halde metin içinde geçen ilk 5XXXXXXXXX bloğunu yakala
        m = re.search(r"5\d{9}", s)
        if m:
            cand = m.group(0)

    if not cand:
        return None

    # Görsel format: 5xx xxxxxxx
    disp = f"{cand[:3]} {cand[3:]}"          # "530 1234567"
    e164 = "+90" + cand                      # "+905301234567"
    return disp, e164

def phone_to_e164_any(text: str) -> str | None:
    """Her türlü girdiden E.164 üret; olmazsa None."""
    out = parse_phone_any(text)
    return out[1] if out else None

HONORIFIC_RE = re.compile(r'^(say(?:ı|i)n|sn\.?|dr\.?|mr\.?|mrs\.?|prof\.?|doç\.?|doc\.?)\s+', re.I)

def clean_name(name: str | None) -> str:
    if not name: return ""
    n = strip_markup(name)
    n = HONORIFIC_RE.sub("", n).strip()
    n = re.sub(r'["“”\'’]+', '', n)       # tırnakları at
    n = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü’'\-\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def is_valid_fullname(name: str | None) -> bool:
    if not name:
        return False
    parts = [p for p in clean_name(name).split() if p]
    # 1–5 parça, her parça en az 2 karakter olsun (örn. "Burcu", "Mehmet Akif", "Ali Veli Duran")
    return 1 <= len(parts) <= 5 and all(len(p) >= 2 for p in parts)


def identity_complete(ctx):
    cust = ctx.goal.customer if ctx and ctx.goal and ctx.goal.customer else {}
    return bool(
        cust.get("fullName") and
        cust.get("phone") and
        cust.get("service")
    )

_HONORIFICS = {
    "sn","sayın","sayin","dr","dr.","doktor","prof","prof.","doç","doc","doç.","öğr","ogr","öğr.",
    "mr","mrs","ms"
}

def _clean_honorifics(name: str) -> str:
    if not name:
        return ""
    # harf/digit/boşluk/'/- dışını boşluk yap
    t = re.sub(r"[^\w\sÇĞİÖŞÜçğıöşü'-]+", " ", name, flags=re.UNICODE)
    toks = [w for w in t.strip().split() if w.casefold().strip(".") not in _HONORIFICS]
    return " ".join(toks)

# TR telefon (mobil/sabit) yakalayıcı
_PHONE_ANY = re.compile(
    r"(?:\+?90)?\s*0?\s*(?P<a>\d{3})\D*(?P<b>\d{3})\D*(?P<c>\d{2})\D*(?P<d>\d{2})"
)




def _format_phone(m: re.Match) -> tuple[str, str]:
    a,b,c,d = m.group("a","b","c","d")
    disp = f"0{a} {b} {c} {d}"          # 0xxx xxx xx xx
    e164 = f"+90{a}{b}{c}{d}"           # +90xxxxxxxxxx
    return disp, e164

def parse_identity_semicolon(msg: str):
    """
    'Ad( …) ; Telefon' -> (name, phone_disp, phone_e164)
    - Ad: 1–5 kelime (Türkçe harf destekli), ünvanlar kırpılır.
    - Telefon: mevcut parse_phone_any ile normalize edilir.
    """
    if not msg:
        return None
    s = unicodedata.normalize("NFC", msg).strip()
    if ";" not in s:
        return None

    left, right = s.split(";", 1)
    name_raw = _clean_honorifics(left.strip())

    # 1–5 parça
    tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü’'-]*", name_raw)
    if not (1 <= len(tokens) <= 5):
        return None
    name = " ".join(tokens)

    # Telefonu kendi normalizer'ınla yakala
    p = parse_phone_any(right)
    if not p:
        return None
    disp, e164 = p
    return (name, disp, e164)

# --- Türkçe baş harf düzeltme / titlecase ---
import re

def _tr_lower(s: str) -> str:
    # Türkçe küçük harfe çevirme: I→ı, İ→i; diğerleri .lower()
    return s.replace("I", "ı").replace("İ", "i").lower()

def _title_words_tr(s: str) -> str:
    """
    'mehmet akif  ersoy' → 'Mehmet Akif Ersoy'
    'ışık ilker' → 'Işık İlker'
    İçteki '-' ve apostrof (') korunur: "oğuz-kaan", "Ahmet'in"
    """
    if not s:
        return ""
    s = s.strip()

    def title_one_word(w: str) -> str:
        if not w:
            return w
        # İlk karakteri TR büyük yap, kalanını TR küçük yap
        first = w[0]
        rest  = w[1:]
        if first == "i":
            first_u = "İ"
        elif first == "ı":
            first_u = "I"
        else:
            first_u = first.upper()
        return first_u + _tr_lower(rest)

    # Kelimeler arası boşlukları korumak için ayırıcıları yakalayarak böl
    parts = re.split(r"(\s+)", s)

    out_parts = []
    for p in parts:
        # boşluk veya tamamen ayırıcı değilse
        if not p or p.isspace():
            out_parts.append(p)
            continue

        # içte '-' veya apostrof ile ayrılmış alt parçalara da title uygula
        sub = re.split(r"([\-'])", p)
        sub = [title_one_word(x) if x and x not in "-'" else x for x in sub]
        out_parts.append("".join(sub))

    return "".join(out_parts)


def set_fullname(state, name: str, source: str = "user") -> bool:
    """
    Ctx içine fullName yazar. True/False döner.
    Beklenen yol: state.goal.customer['fullName']
    """
    if not name:
        return False
    nm = unicodedata.normalize("NFC", name).strip()
    # ünvan kırp
    toks = [w for w in nm.split() if w.casefold().strip(".") not in _HONORIFICS]
    nm  = " ".join(toks) if toks else nm
    if not is_valid_fullname(nm):
        return False
    nm = _title_words_tr(nm)

    # path'i güvenle oluştur
    try:
        goal = getattr(state, "goal")
    except Exception:
        goal = None
    if goal is None:
        class _Tmp: pass
        state.goal = _Tmp()
        state.goal.customer = {}
    elif not hasattr(state.goal, "customer") or not isinstance(state.goal.customer, dict):
        state.goal.customer = {}

    state.goal.customer["fullName"] = nm

    # kaynağı meta'da tutmak isterseniz (opsiyonel)
    try:
        state.meta = getattr(state, "meta", {}) or {}
        state.meta["fullName_source"] = source
    except Exception:
        pass
    return True

# Türkçe ad-soyad çıkarımı (cümle içinden)
import re

def strip_markup(text: str) -> str:
    # markdown bold/italic/inline-code vb. işaretleri temizle
    return re.sub(r'[*_`~]+', '', text or '')

NAME_HINT_RE = re.compile(
    r"\b(ad[ıi]m|ismim|ben)\s*[:\-]?\s+([A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,}(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,}){0,4})\b",
    re.I
)

PURE_NAME_RE = re.compile(
    r"^[A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,}(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,}){0,4}$"
)




def mark_invited_to_schedule(ctx: Ctx) -> Ctx:
    ctx = ctx or Ctx()
    ctx.meta = getattr(ctx, "meta", {})  # varsa korur
    ctx.meta["invited_to_schedule"] = True
    ctx.status = "collecting"            # kimlik/detay toplama moduna geçir
    return ctx

def clear_invite_flag(ctx: Ctx) -> None:
    if hasattr(ctx, "meta") and isinstance(ctx.meta, dict):
        ctx.meta.pop("invited_to_schedule", None)
import re
import unicodedata

def normalize_tr(s: str) -> str:
    # Türkçe diakritikleri sadeleştir + lowercase
    if not s: return ""
    s = s.replace("İ","I").replace("ı","i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

# Olumlu tetikleyiciler (geniş tutuldu; dilediğin gibi düzenleyebilirsin)
AFFIRM_KEYWORDS = {
    "olur","tamam","istiyorum","isterim","alabilirim","alayim","alayım",
    "ok","okey","evet","uygun","randevu al","randevu alalim","randevu alalım","ayarla"
}

# (Opsiyonel) açık olumsuzlar — yanlış tetiklemeyi önler
NEGATIVE_KEYWORDS = {
    "hayir","hayır","istemiyorum","vazgec","vazgeç","vazgeciyorum","iptal","olmaz","yok"
}

def is_affirmative_freeform(text: str) -> bool:
    t = normalize_tr(text)
    # kelime/ibare içerme kontrolü
    if any(k in t for k in AFFIRM_KEYWORDS):
        # "istemiyorum" gibi olumsuz ifadeler varsa tetikleme
        if any(n in t for n in NEGATIVE_KEYWORDS):
            return False
        return True
    return False

# --- Safe logging (UTF-8) ---

logging.basicConfig(
    filename=LOGFILE,          # dosyaya yaz
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",          # <-- kritik
)


# en üstte LOGFILE zaten tanımlı
def log(*parts):
    try:
        msg = " ".join(str(p) for p in parts)
        with open(LOGFILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass




# Başlangıçta bir satır yaz (dosya oluşsun)
log("boot: app started")
log("burcu")

def _src_label(meta: dict) -> str:
    if not meta:
        return ""
    return (
        meta.get("source_url")
        or meta.get("source_path")
        or meta.get("filename")
        or meta.get("source")
        or meta.get("id")
        or ""
    )

def build_context_from_hits(hits, max_chars=6000, rid: str | None = None):
    """hits: List[Tuple[Document, score]] -> RAG context string.
    Kaynak etiketlerini gövdeye [SRC:..] olarak damgalar ve SON kullanım listesini loglar."""
    parts, used = [], []
    for d, _ in hits:
        meta = getattr(d, "metadata", {}) or {}
        src  = _src_label(meta)
        body = (getattr(d, "page_content", "") or "").strip()
        if not body:
            continue
        tagged = f"[SRC:{src}]\n{body}"
        parts.append(tagged)
        used.append(src)

    ctx = "\n\n---\n\n".join(parts)
    if max_chars and len(ctx) > max_chars:
        ctx = ctx[:max_chars]

    # SON haline göre log (eski [CTX] log’unu kaldırıp bunu kullan)
    log(f"[CTX] rid={rid} sources={used}")
    log(f"[PROMPT] rid={rid} ctx_len={len(ctx)}")
    return ctx


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def ensure_dirs_and_seed():
    # Klasörleri oluştur
    os.makedirs(APPDATA_DIR,   exist_ok=True)
    os.makedirs(EMBED_DIR_LCL, exist_ok=True)
    os.makedirs(URLS_DIR_LCL,  exist_ok=True)
    os.makedirs(VSTORE_DIR,    exist_ok=True)

    # LocalAppData boşsa seed kopyala (önce bundle, yoksa EXE yanı)
    if not any(glob.glob(os.path.join(EMBED_DIR_LCL, "*"))):
        if os.path.isdir(EMBED_DIR_BUNDLE):
            _copy_tree(EMBED_DIR_BUNDLE, EMBED_DIR_LCL)
        elif EMBED_DIR_EXE and os.path.isdir(EMBED_DIR_EXE):
            _copy_tree(EMBED_DIR_EXE, EMBED_DIR_LCL)

    if not any(glob.glob(os.path.join(URLS_DIR_LCL, "*"))):
        if os.path.isdir(URLS_DIR_BUNDLE):
            _copy_tree(URLS_DIR_BUNDLE, URLS_DIR_LCL)
        elif URLS_DIR_EXE and os.path.isdir(URLS_DIR_EXE):
            _copy_tree(URLS_DIR_EXE, URLS_DIR_LCL)

    try:
        log("[PATHS] APPDATA_DIR =", APPDATA_DIR)
        log("[PATHS] EMBED_DIRS  =", EMBED_DIRS)
        log("[PATHS] URLS_DIRS   =", URLS_DIRS)
    except Exception:
        print("[PATHS] APPDATA_DIR =", APPDATA_DIR)
        print("[PATHS] EMBED_DIRS  =", EMBED_DIRS)
        print("[PATHS] URLS_DIRS   =", URLS_DIRS)
# ---- /PATHS ----

def _log_paths():
    log("[PATHS] APPDATA_DIR =", APPDATA_DIR)
    log("[PATHS] EMBED_DIRS  =", EMBED_DIRS)
    log("[PATHS] URLS_DIRS   =", URLS_DIRS)
    for d in EMBED_DIRS:
        log("[PATHS] EMBED dir exists?", d, os.path.isdir(d))
    for d in URLS_DIRS:
        log("[PATHS] URLS  dir exists?", d, os.path.isdir(d))

# app boot sırasında bir kez çağır
_log_paths()




# HTML şablon klasörü
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Flask uygulamasını başlat
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["JSON_AS_ASCII"] = False
CORS(app)

#app = Flask(__name__)
#CORS(app)
#client = OpenAI(api_key="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")

# ---- URL INGEST HELPERS ----
import time, requests

UA = "SailyAI-RAG/1.0 (+https://sailead.com.tr/iletisim; contact: destek@sailead.com.tr)"

def _iter_url_list_files():
    """URL listesi içeren .txt dosyalarını öncelik sırasıyla getir (LocalAppData > EXE > bundle > dev).
       Aynı ada sahip dosyalarda ilk görüleni (LocalAppData) kazanır."""
    all_files = []
    for d in URLS_DIRS:
        if not os.path.isdir(d):
            continue
        log("[URLS] scan dir:", d)
        all_files.extend(glob.glob(os.path.join(d, "*.txt")))
    seen = set()
    uniq = []
    for p in all_files:
        name = os.path.basename(p).lower()
        if name in seen:
            continue
        seen.add(name)
        uniq.append(p)
    return uniq

def _read_url_list(path):
    """Her satır bir URL olacak şekilde oku; # ile başlayan satırları ve boşları at."""
    urls = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("http://") or s.startswith("https://"):
                    urls.append(s)
    except Exception as e:
        log("[URLS] read fail:", os.path.basename(path), repr(e))
    return urls

def _html_to_text(html):
    """Önce trafilatura, olmazsa BS4 ile sağlam fallback."""
    # 1) Trafilatura
    try:
        from trafilatura import extract
        text = extract(html, include_formatting=False, include_links=False, favor_recall=True)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass  # trafilatura yok/başarısız → BS4'e geç

    # 2) BS4 fallback: lxml varsa kullan, yoksa html.parser
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")  # PyInstaller'da garanti

    # Gürültü temizliği
    for t in soup(["script", "style", "noscript", "svg", "iframe"]):
        t.extract()
    for sel in ["header", "nav", "footer", ".site-header", ".site-footer", ".menu", ".widget-area",
                "#cookie", ".cookie", "#gdpr", ".gdpr"]:
        for n in soup.select(sel):
            n.extract()

    # Ana içerik odaklı çıkarım (Elementor/WordPress)
    parts = []
    for sel in ["main", "article", ".entry-content", ".post-content",
                ".elementor-widget-text-editor", ".elementor-widget-container",
                ".content", "#content", "p", "h1", "h2", "h3"]:
        for n in soup.select(sel):
            txt = n.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

    text = "\n".join(parts) if parts else soup.get_text(" ", strip=True)
    return text.strip()


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

def _fetch_url_to_text(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    # encoding düzeltmesi
    r.encoding = r.apparent_encoding or r.encoding or "utf-8"
    html = r.text
    txt = _html_to_text(html) or ""
    if txt.strip():
        return txt
    # Teşhis için ham HTML’i dökebilirsiniz (opsiyonel)
    # _dump_html_debug(url, html)
    raise ValueError("empty_text")


def _ingest_urls_in_dirs(max_urls=100):
    """URL listelerindeki sayfaları indir → metne çevir → [URL: ...] etiketiyle döndür."""
    texts = []
    total = 0
    for p in _iter_url_list_files():
        urls = _read_url_list(p)
        for u in urls:
            if total >= max_urls:
                log("[INGEST] max_urls reached:", max_urls)
                return texts
            try:
                t0 = time.time()
                txt = _fetch_url_to_text(u)
                tagged = f"[URL:{u}]\n{txt}"
                texts.append(tagged)
                total += 1
                log(f"[INGEST] fetched: {u} → chars={len(txt)} time={time.time()-t0:.2f}s")
            except Exception as e:
                log("[INGEST] fail:", u, repr(e))
    return texts
# ---- /URL INGEST HELPERS ----




# ---------------- [1] SESSION SLOTS (GLOBAL) ----------------
SESSION_SLOTS = {}  # sid -> {"name":..., "phone":..., "day":..., "time":..., "service":..., "language":...}

def get_slots(sid: str):
    return SESSION_SLOTS.setdefault(
        sid,
        {
            "name": None,
            "phone": None,
            "day": None,
            "time": None,
            "service": None,
            "language": None,
        }
    )


def set_slots(sid: str, **kwargs):
    slots = get_slots(sid)
    before = dict(slots)  # debug için
    for k, v in kwargs.items():
        # Değer yoksa veya boş string ise ÖNCEKİNİ SİLME
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        # (telefon zaten extract_phone ile normalize edilmiş geliyor)
        slots[k] = v
    log(f"[SLOTS] update sid={sid} before={before} after={slots}")
    return slots


# ---------------- [2] PHONE PARSING ----------------
_PHONE_RE = re.compile(
    r'(?:\+?\s*90\s*|\b0\s*)?\s*\(?\s*(5\d{2}|\d{3,4})\s*\)?[\s\-.]*\d{3}[\s\-.]*\d{2}[\s\-.]*\d{2}\b'
)

def extract_phone(text: str) -> str | None:
    if not text: return None
    m = _PHONE_RE.search(text)
    if not m: return None
    raw = m.group(0)
    digits = "".join(ch for ch in raw if ch.isdigit())
    # normalize
    if digits.startswith("90") and len(digits) in (12, 13):
        digits = digits[2:]
    if len(digits) == 10 and digits.startswith("5"):
        digits = "0" + digits
    if len(digits) == 10 and not digits.startswith("0"):
        digits = "0" + digits
    return digits if 10 <= len(digits) <= 11 else None



import re

# "telefon/telefonum/tel/no/numara/randevu/saat/gün" gelmeden önce adı kes
_NAME_FROM_LABEL_RE = re.compile(
    r'\b(?:benim\s+ad[ıi]m|ad[ıi]m|ismim|ad\s*-?\s*soyad[ıi]m?|ad\s*soyad|ad[ıi]nız|adiniz|isim)\b'
    r'\s*[:\-]?\s*'
    r'([A-Za-zÇĞİÖŞÜçğıöşü\'\s]{2,60}?)'
    r'(?=\s+(?:telefon(?:um)?|tel|no|numara|randevu|saat|gün)\b|[\d,.;:!?)]|$)',
    re.IGNORECASE
)

_NAME_FROM_BEN_RE = re.compile(
    r'\bben\s+([A-Za-zÇĞİÖŞÜçğıöşü\'\s]{2,60}?)'
    r'(?=\s+(?:telefon(?:um)?|tel|no|numara|randevu|saat|gün)\b|[\d,.;:!?)]|$)',
    re.IGNORECASE
)

_NAME_TWO_TOKEN_RE = re.compile(
    r'\b([A-Za-zÇĞİÖŞÜçğıöşü]{2,})\s+([A-Za-zÇĞİÖŞÜçğıöşü]{2,})\b'
)

_BLOCK_WORDS = {
    "randevu","saat","dakika","bugün","yarın","hemen",
    "pazartesi","salı","çarşamba","perşembe","cuma","cumartesi","pazar",
    "adres","konum","şube","ücret","fiyat","telefon","telefonum","tel","numara","no"
}

def _clean_name(cand: str) -> str:
    # '... telefonum 0535' gibi artıkları at
    cand = re.sub(r'\b(?:telefon(?:um)?|tel|no|numara)\b.*$', '', cand, flags=re.IGNORECASE).strip()
    # çift boşlukları toparla
    cand = re.sub(r'\s+', ' ', cand).strip()
    return cand

def extract_name(text: str, *, have_phone: bool = False) -> str | None:
    if not text:
        return None

    m = _NAME_FROM_LABEL_RE.search(text)
    if m:
        return _clean_name(m.group(1))

    m = _NAME_FROM_BEN_RE.search(text)
    if m:
        return _clean_name(m.group(1))

    # Etiketsiz iki kelime — telefona yakınsa daha olasıdır
    t = text.strip()
    if have_phone or len(t) <= 60:
        m = _NAME_TWO_TOKEN_RE.search(t)
        if m:
            cand = f"{m.group(1)} {m.group(2)}"
            if cand and not any(w in cand.lower() for w in _BLOCK_WORDS):
                return _clean_name(cand)

    return None




def update_slots_from_text(sid: str, user_text: str):
    """
    Öncelik:
    - Kullanıcıyı şu formatta yönlendiriyoruz:
      "Ad Soyad ; 5xx xxx xx xx ; İlgilendiğiniz Hizmet"
      → name, phone, service tek hamlede set edilir.

    - Bu format yoksa:
      - Telefon, isim, gün, saat sezgisel çıkarılır.
      - Hizmet için önce LLM, sonra keyword fallback kullanılır.
    """
    import re
    try:
        from unidecode import unidecode
    except ImportError:
        # unidecode yoksa, basit fallback
        def unidecode(x): return x

    before = get_slots(sid) or {}
    merged = dict(before)

    raw_txt = (user_text or "").strip()
    txt = raw_txt.lower()
    txt_clean = unidecode(txt)

    # =========================================================
    # ✅ 0) ÖNCELİK: "Ad Soyad ; Telefon ; Hizmet" FORMAT PARSE
    # =========================================================
    if ";" in raw_txt:
        parts = [p.strip() for p in raw_txt.split(";")]

        # 1. alan: Ad Soyad
        if len(parts) >= 1 and parts[0]:
            merged["name"] = parts[0]

        # 2. alan: Telefon
        if len(parts) >= 2 and parts[1]:
            try:
                if "parse_phone_any" in globals():
                    p = parse_phone_any(parts[1])
                    if p:
                        merged["phone"] = p[0]
                    else:
                        merged["phone"] = parts[1]
                else:
                    merged["phone"] = parts[1]
            except Exception:
                merged["phone"] = parts[1]

        # 3. alan: Hizmet
        if len(parts) >= 3 and parts[2]:
            merged["service"] = parts[2]

        # Gün / saat istersen buradan da çıkarabilirsin (opsiyonel)
        # tail = ";".join(parts[2:])

        set_slots(
            sid,
            name=merged.get("name"),
            phone=merged.get("phone"),
            day=merged.get("day"),
            time=merged.get("time"),
            service=merged.get("service"),
            language=merged.get("language"),
        )
        after = get_slots(sid) or {}
        log(f"[SLOTS] via ';' sid={sid} text={raw_txt!r} after={after}")
        return

    # =========================================================
    # ✅ 1) Telefon (serbest metin)
    # =========================================================
    try:
        phone = parse_phone_any(txt) if "parse_phone_any" in globals() else None
        if phone:
            merged["phone"] = phone[0]
    except Exception:
        pass

    # =========================================================
    # ✅ 2) İsim (serbest metin)
    # =========================================================
    try:
        if "extract_name" in globals():
            name = extract_name(txt, have_phone=bool(merged.get("phone")))
        else:
            name = None
        if name:
            merged["name"] = name
    except Exception:
        pass

    # =========================================================
    # ✅ 3) Gün & Saat
    # =========================================================
    try:
        day = _parse_day_free(txt) if "_parse_day_free" in globals() else None
        if day:
            merged["day"] = day
    except Exception:
        pass

    try:
        tm = _extract_time(txt) if "_extract_time" in globals() else None
        if tm:
            merged["time"] = tm
    except Exception:
        pass

    # =========================================================
    # ✅ 4) LLM ile Hizmet Algılama
    # =========================================================
    parsed_llm = {}
    try:
        if "parse_contact_answer_llm" in globals():
            parsed_llm = parse_contact_answer_llm(raw_txt) or {}
            llm_service = (parsed_llm.get("service") or "").strip()
            if llm_service:
                merged["service"] = llm_service
    except Exception as e:
        log("[LLM_SERVICE_ERR]", repr(e))
        parsed_llm = {}

    # =========================================================
    # ✅ 5) Fallback SERVICE_MAP (LLM bulamazsa)
    # =========================================================
    if not merged.get("service"):
        SERVICE_MAP = {
            r"isitme testi|odyometri|odyometrik": "İşitme Testi",
            r"cihaz ayari|kulaklik ayari|cihaz ayarlama": "Cihaz Ayarı",
            r"pil degisimi|pil": "Pil Değişimi",
            r"isitme cihazi denemesi|cihaz deneme|cihaz bak|yeni cihaz": "İşitme Cihazı Denemesi",
            r"tinnitus|cinlama": "Tinnitus Testi ve Değerlendirme",
            r"sas analizi|sleep apnea|uyku apnesi": "SAS Analizi",
            r"isitsel rehabilitasyon|isitme rehabilitasyonu": "İşitsel Rehabilitasyon",
            r"vng|videonistagmografi": "VNG",
            r"posturografi": "Postürografi",
            r"genel denge|denge testi|denge analizi": "Genel Denge Değerlendirme",
            r"sanal gerceklik|vr uygulamasi": "Sanal Gerçeklik Uygulaması",
            r"dil ve konusma|konusma terapisi|dil terapisi": "Dil ve Konuşma Bozuklukları",
            r"ses terapisi": "Ses Terapisi",
        }

        for pattern, label in SERVICE_MAP.items():
            if re.search(pattern, txt_clean):
                merged["service"] = label
                break

    # =========================================================
    # ✅ 6) Dil Bilgisi (LLM varsa)
    # =========================================================
    if parsed_llm.get("language"):
        merged["language"] = parsed_llm["language"]

    # =========================================================
    # ✅ 7) KAYDET
    # =========================================================
    set_slots(
        sid,
        name=merged.get("name"),
        phone=merged.get("phone"),
        day=merged.get("day"),
        time=merged.get("time"),
        service=merged.get("service"),
        language=merged.get("language"),
    )
    after = get_slots(sid) or {}
    log(f"[SLOTS] update sid={sid} text={raw_txt!r} after={after}")




import re, datetime as _dt
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    _TZ = None

def _now_tr():
    return _dt.datetime.now(_TZ) if _TZ else _dt.datetime.now()

def _weekday_tr(dt: _dt.datetime) -> str:
    # 0=Mon ... 6=Sun
    return ["pazartesi","salı","çarşamba","perşembe","cuma","cumartesi","pazar"][dt.weekday()]

_DAY_CANON = {
    "pazartesi":"pazartesi","salı":"salı","sali":"salı","çarşamba":"çarşamba","carsamba":"çarşamba",
    "perşembe":"perşembe","persembe":"perşembe","cuma":"cuma","cumartesi":"cumartesi","pazar":"pazar"
}

def _parse_day_free(text: str) -> str | None:
    t = _norm_tr(text or "")
    if not t: return None
    # mutlak adlar
    for k, v in _DAY_CANON.items():
        if k in t: return v
    # relatif
    if "bugun" in t:  return _weekday_tr(_now_tr())
    if "yarin" in t:  return _weekday_tr(_now_tr() + _dt.timedelta(days=1))
    return None

from datetime import date, datetime
from zoneinfo import ZoneInfo

TR_MONTHS = {
    "ocak":1,"oca":1,
    "şubat":2,"subat":2,"şub":2,"sub":2,
    "mart":3,
    "nisan":4,"nis":4,
    "mayıs":5,"mayis":5,"may":5,
    "haziran":6,"haz":6,
    "temmuz":7,"tem":7,
    "ağustos":8,"agustos":8,"ağu":8,"agu":8,
    "eylül":9,"eylul":9,"eyl":9,
    "ekim":10,"eki":10,
    "kasım":11,"kasim":11,"kas":11,
    "aralık":12,"aralik":12,"ara":12,
}

TR_DOW = {
    "pazartesi":0,"pzt":0,
    "salı":1,"sali":1,"sal":1,
    "çarşamba":2,"carsamba":2,"çar":2,"car":2,
    "perşembe":3,"persembe":3,"per":3,
    "cuma":4,
    "cumartesi":5,"cmt":5,
    "pazar":6,
}

def _norm_tr(s: str) -> str:
    return (s.lower()
              .replace("ğ","g").replace("ü","u").replace("ş","s")
              .replace("ı","i").replace("ö","o").replace("ç","c")
              .strip())

def _next_weekday(start: date, target_wd: int) -> date:
    """start dahil olmayacak şekilde bir SONRAKİ target_wd gününü verir."""
    delta = (target_wd - start.weekday()) % 7
    if delta == 0:
        delta = 7
    return start + _dt.timedelta(days=delta)

def parse_tr_date_with_weekday(text: str, *, prefer_future=True, tz="Europe/Istanbul"):
    """
    '12 ekim cumartesi' / '12 ekim' / '12 ekim 2026 cumartesi' / 'Perşembe 12:00'
    -> (date, weekday_match_flag)
    weekday_match_flag: True/False/None (None: haftagünü verilmedi)
    """
    if not text:
        return None, None

    s = _norm_tr(text)
    today = datetime.now(ZoneInfo(tz)).date()

    # 1) Bugün / Yarın / Öbür gün kısa yolları
    if re.search(r"\bbug[uü]n\b", s):
        dow_flag = None  # haftagünü verilmedi
        return today if not prefer_future or True else today, dow_flag
    if re.search(r"\byar[ıi]n\b", s):
        return today + _dt.timedelta(days=1), None
    if re.search(r"\b(öb[uü]r g[uü]n|er tes[iı] g[uü]n|ertesi g[uü]n)\b", s):
        return today + _dt.timedelta(days=2), None

    # 2) Sadece haftagünü (+opsiyonel saat) varsa: 'Perşembe', 'Perşembe 12:00', 'Kozyatağı Perşembe Saat 12:00'
    # metin içinde herhangi bir TR_DOW anahtarını ara
    found_dow = None
    for key in sorted(TR_DOW.keys(), key=len, reverse=True):
        if re.search(rf"\b{key}\b", s):
            found_dow = TR_DOW[key]
            break
    if found_dow is not None:
        target_date = _next_weekday(today, found_dow) if prefer_future else today
        # (istersen burada saati de kullanıp datetime döndüren ayrı bir fonksiyon yapabilirsin)
        return target_date, True

    # 3) Orijinal kalıbın: gün + ay (+ yıl) (+ haftagünü)
    m = re.search(
        r"\b(\d{1,2})\s+([a-z\.çğıöşü]+)(?:\s+(\d{4}))?(?:\s+([a-zçğıöşü\.]+))?\b",
        s, flags=re.UNICODE
    )
    if not m:
        return None, None

    day = int(m.group(1))
    mon_raw = m.group(2).replace(".", "")
    year_str = m.group(3)
    dow_raw = (m.group(4) or "").replace(".", "")

    month = TR_MONTHS.get(mon_raw)
    if not month:
        return None, None

    user_dow = TR_DOW.get(dow_raw) if dow_raw else None

    if year_str:
        year = int(year_str)
        try_date = date(year, month, day)
        if user_dow is None:
            return try_date, None
        return try_date, (try_date.weekday() == user_dow)
    else:
        start_year = today.year
        candidates = []
        for y in range(start_year, start_year + 3):
            try:
                d = date(y, month, day)
            except ValueError:
                continue
            if prefer_future and d < today:
                continue
            candidates.append(d)

        if not candidates:
            return None, None

        if user_dow is None:
            return candidates[0], None

        for d in candidates:
            if d.weekday() == user_dow:
                return d, True

        return candidates[0], False

def parse_tr_day_month(text: str, *, prefer_future: bool = True,
                       tz: str = "Europe/Istanbul") -> date | None:
    """
    '12 ekim', '12 Ekim', '12 EKI', '12 ekim 2025' vb. ifadeleri parse eder.
    Yıl verilmemişse:
      - prefer_future=True ise bugün ile karşılaştırıp geçmişse gelecek yıla atlar.
      - prefer_future=False ise bu yıl döner (geçmiş olabilir).
    Dönüş: datetime.date | None
    """
    if not text:
        return None

    # Noktalı kısaltmalar (Eki.) gibi şeyleri tolere et
    s = _norm_tr(text).replace(".", " ")
    # '12 ekim', '12   ekim   2025' vb. — yıl opsiyonel
    m = re.search(r"\b(\d{1,2})\s+([a-zçğıöşü]+)(?:\s+(\d{4}))?\b", s, flags=re.UNICODE)
    if not m:
        return None

    day = int(m.group(1))
    mon_raw = m.group(2)
    year_str = m.group(3)

    month = TR_MONTHS.get(mon_raw)
    if not month:
        return None

    if year_str:
        year = int(year_str)
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # yıl yoksa → bu yılı dene; gerekirse gelecek yıl
    today = datetime.now(ZoneInfo(tz)).date()
    year = today.year
    try:
        d = date(year, month, day)
    except ValueError:
        return None

    if prefer_future and d < today:
        try:
            d = date(year + 1, month, day)
        except ValueError:
            return None
    return d

def parse_tr_date(text: str, *, prefer_future: bool = True):
    """
    Tek giriş noktası.
    Döner: (tarih: date | None, weekday_ok: True/False/None)
      - weekday_ok = None  : metinde hafta günü yok
      - weekday_ok = True  : metindeki hafta günü ile tarih uyuştu
      - weekday_ok = False : metindeki hafta günü ile tarih uyuşmadı
    """
    # 1) Önce hafta günlü parser
    d, wk_ok = parse_tr_date_with_weekday(text, prefer_future=prefer_future)
    if d:
        return d, wk_ok
    # 2) Olmazsa gün+ay(+yıl)
    d2 = parse_tr_day_month(text, prefer_future=prefer_future)
    return d2, None

def _extract_time(text: str):
    """16:30, 16.30, '16 30', sadece '16' (→16:00)"""
    if not text: return None
    s = text
    m = re.search(r'\b(\d{1,2})[:.](\d{2})\b', s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59: return (h, mi)
    m = re.search(r'\b(\d{1,2})\s+(\d{2})\b', s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59: return (h, mi)
    m = re.search(r'\b(\d{1,2})\b', s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23: return (h, 0)
    return None

    
# --- İSİM DOĞRULAMA ---
import re 

_TR_MAP = str.maketrans({
    "ı":"i","İ":"i","ş":"s","Ş":"s","ğ":"g","Ğ":"g","ç":"c","Ç":"c","ö":"o","Ö":"o","ü":"u","Ü":"u"
})
def _norm_tr(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).translate(_TR_MAP)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# İsim olamayacak kelimeler (niyet, konum, akış kelimeleri)
_BAD_NAME_TOKS = {
    "randevu","rdv","al","almak","alabilir","iste","istiyorum","isterim",
    "onayliyorum","onayladim","tamam","olur","evet","hayir","hayır",
    "hangi","kim","hoca","uzman","doktor","baska","başka","yerde","oraya",
    "gel","gelmek","gelirim","geliyorum","gitmek","gidecegim","gideceğim",
    "lütfen","lutfen","tesekkur","teşekkür",
    "gun","gün","saat","bugun","bugün","yarin","yarın",
    "adres","konum","yer","sube","şube","klinik","kliniğe",
    "kadikoy","kadıköy","kozyatagi","kozyatağı","mecidiyekoy","mecidiyeköy",
    "bakirkoy","bakırköy","torun","center","incirli","caddesi","sokak","mahallesi",
    "sahrayi","cedit","ataturk","atürk"
}
_BAD_NAME_TOKS = { _norm_tr(w) for w in _BAD_NAME_TOKS }

def _is_plausible_person_name(cand: str) -> bool:
    """'Ad Soyad' benzeri gerçek kişi adı mı? (2–4 kelime, sayı yok, stopword yok)"""
    if not cand: 
        return False
    # Sadece harf/boşluk bırak
    t = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü\s']", " ", cand)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    toks = t.split()
    # ad + soyad (+ opsiyonel 2. ad)
    if not (2 <= len(toks) <= 4):
        return False
    # Sayı/url benzeri karakter olmasın
    if any(ch.isdigit() for ch in cand):
        return False
    # Niyet/konum/akış kelimeleri içermez
    ntoks = set(_norm_tr(t).split())
    if ntoks & _BAD_NAME_TOKS:
        return False
    return True

def _parse_structured_fields(text: str) -> dict:
    import re
    if not text or ";" not in text: return {}
    name_raw, *rest = [p.strip() for p in text.split(";")]
    out = {}
    # Ad Soyad
    if name_raw and _is_plausible_person_name(name_raw):
        out["name"] = re.sub(r"\s+"," ", name_raw).strip()
    # Telefon
    if rest:
        digits = re.sub(r"\D+", "", rest[0])
        if 10 <= len(digits) <= 13:
            out["phone"] = "0" + digits[-10:]
    return out


def get_resource_path(*parts) -> Path:
    """
    PyInstaller ile paketlendiğinde veya geliştirme ortamında 
    dosya yolunu doğru döndürür.
    """
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path.joinpath(*parts)






def read_txt(p): return open(p, "r", encoding="utf-8", errors="ignore").read()
def read_pdf(p):
    txt=[]; r=PdfReader(p)
    for pg in r.pages: txt.append(pg.extract_text() or "")
    return "\n".join(txt)

def read_csv_smart(path):
    encs = ["utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "latin1"]
    seps = [",", ";", "|", "\t"]  # virgül, noktalı virgül, pipe, tab
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",            # esnek ayrıştırıcı
                    on_bad_lines="skip",        # bozuk satırı atla (pandas>=1.3)
                    quoting=csv.QUOTE_MINIMAL
                )
                # İçerik gerçekten okunmuş mu?
                if df.shape[1] >= 2 and len(df) > 0:
                    return df
            except Exception as e:
                last_err = e
    # Hiçbiri olmadıysa orijinal hatayı fırlat
    raise last_err if last_err else Exception("CSV could not be parsed with common encodings/separators.")

def read_excel_like(p: str) -> str:
    

    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _decode_with_fallback(b: bytes) -> str:
        for enc in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        # son çare
        return b.decode("utf-8", errors="ignore")

    def _normalize_quotes(s: str) -> str:
        # kıvrık tırnakları düz tırnağa çevir
        return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    def _flatten_df(df: "pd.DataFrame") -> str:
        # hücreleri string yap, boşları temizle
        df = df.fillna("").applymap(lambda x: str(x).strip())
        # tamamen boş satırları at
        df = df.loc[~(df == "").all(axis=1)]
        if df.empty:
            return ""

        cols_norm = [str(c).strip().lower() for c in df.columns]

        def find_col(cands):
            for c in cands:
                if c in cols_norm:
                    return df.columns[cols_norm.index(c)]
            return None

        soru_col  = find_col(["soru", "question", "q"])
        cevap_col = find_col(["cevap", "yanit", "yanıt", "answer", "a"])

        if soru_col and cevap_col:
            rows = []
            for _, r in df.iterrows():
                s = r.get(soru_col, "")
                c = r.get(cevap_col, "")
                if s or c:
                    rows.append(f"Soru: {s}\nCevap: {c}")
            return "\n\n---\n\n".join(rows)

        # Soru/Cevap yoksa genel satır düzleştirme
        rows = []
        for _, r in df.iterrows():
            parts = [f"{col}: {val}" for col, val in r.items() if str(val).strip()]
            if parts:
                rows.append(" | ".join(parts))
        return "\n\n".join(rows)

    try:
        pl = p.lower()
        if pl.endswith(".csv"):
            raw_bytes = _read_bytes(p)
            text = _decode_with_fallback(raw_bytes)
            text = _normalize_quotes(text)

            # 1) pandas ile kokla
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python",
                                 dtype=str, na_filter=False)
            except Exception:
                # 2) noktalı virgül dene
                try:
                    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, na_filter=False)
                except Exception:
                    # 3) virgül zorla
                    df = pd.read_csv(io.StringIO(text), sep=",", dtype=str, na_filter=False)

            flat = _flatten_df(df)
            if flat.strip():
                log("[EMBED] csv parsed:", os.path.basename(p), "shape=", df.shape, "cols=", list(df.columns))
                return flat

            # pandas işe yaramadıysa son çare: ham metni göm
            log("[EMBED] csv empty after parse, fallback to raw text:", os.path.basename(p))
            return text

        elif pl.endswith(".xlsx") or pl.endswith(".xls"):
            # Excel için
            df = pd.read_excel(p, engine="openpyxl", dtype=str)
            flat = _flatten_df(df)
            if flat.strip():
                log("[EMBED] xlsx parsed:", os.path.basename(p), "shape=", df.shape, "cols=", list(df.columns))
            else:
                log("[EMBED] excel-like empty after flatten:", os.path.basename(p))
            return flat

        else:
            return ""

    except Exception as e:
        log("[EMBED] read_excel_like error:", os.path.basename(p), repr(e))
        return ""

    
    
    # Kolon eşlemesi (daha önce eklediğimiz alias mantığı)
    CHAT_ALIASES = {
      "Soru": {"Soru","Question","question","Q"},
      "Cevap": {"Cevap","Answer","answer","A"}
      
    }
    
    DAY_ALIASES = {
      "PERSONEL": {"PERSONEL","EMPLOYEE","emp"},
      "Şube": {"Şube","BRANCH","brch"},
      "Pazartesi": {"Pazartesi","MONDAY","mon"},
      "Salı": {"Salı","TUESDAY","TUE"},
      "Çarşamba": {"Çarşamba","WEDNESDAY","wed"},
      "Perşembe": {"Perşembe","THURSDAY","thr"},
      "Cuma": {"Cuma","FRIDAY","fri"},
      "Cumartesi": {"Cumartesi","SATURDAY","sat"}
     
    }

    KONUM_ALIASES = {
      "ŞUBE": {"ŞUBE","",""},
      "KONUM": {"KONUM","",""}
    }
    
    HIZMET_ALIASES = {     
      "ANA HİZMET": {"ANA HİZMET","",""},
      "ALT HİZMET": {"ALT HİZMET","",""},
      "TEDAVİLER": {"TEDAVİLER","",""}  
    }
   

    def col(df, key, aliases=None):
        aliases = aliases or {key}
        for c in df.columns:
            if str(c).strip() in aliases:
                return c
    return None

   
    
    def format_soru_cevap(df):
        soru_col = col(df, "SORU", CHAT_ALIASES["SORU"])
        cevap_col = col(df, "CEVAP", CHAT_ALIASES["CEVAP"])

        if soru_col and cevap_col:
            lines = []
            for _, r in df.iterrows():
                soru = str(r.get(soru_col, "")).strip()
                cevap = str(r.get(cevap_col, "")).strip()
                if soru or cevap:
                    lines.append(f"Soru: {soru}\nCevap: {cevap}")
                    return "\n\n".join(lines)
                else:
                    return ""

    # PERSONEL ve Şube kolonlarını bul
    pers_col = col(df, "PERSONEL", {"PERSONEL","Personel","Çalışan","Calisan","Employee","Name"})
    sube_col = col(df, "Şube", {"Şube","Sube","Branch","Ofis"})

    # Gün kolonlarını sırayla eşle
    day_cols = []
    for day_name, alias_set in DAY_ALIASES.items():
        c = col(df, day_name, alias_set)
    if c:
        day_cols.append((day_name, c))   # (gün adı, gerçek sütun adı)

    if pers_col and sube_col and day_cols:
        lines = []
    for _, r in df.iterrows():
        # Boş olmayan günleri toparla
        gunler = []
        for day_label, real_col in day_cols:
            val = r.get(real_col, "")
            val = "" if (pd.isna(val) or str(val).strip() in ["", "NaN", "nan"]) else str(val).strip()
            if val:
                gunler.append(f"{day_label}: {val}")
        # Metni oluştur
        kisi = str(r[pers_col]).strip()
        sube = str(r[sube_col]).strip()
        schedule = " | ".join(gunler) if gunler else "Çalışma saati bilgisi yok"
        lines.append(f"Personel: {kisi}\nŞube: {sube}\nProgram: {schedule}")
    return "\n\n".join(lines)

    def format_sube_konum(df):
        sube_col = col(df, "Şube", KONUM_ALIASES)
        konum_col = col(df, "Konum", KONUM_ALIASES)

        if sube_col and konum_col:
            lines = []
            for _, r in df.iterrows():
                sube = str(r.get(sube_col, "")).strip()
                konum = str(r.get(konum_col, "")).strip()
                if sube or konum:
                    lines.append(f"Şube: {sube} - Konum: {konum}")
                    return "\n\n".join(lines)
        else:
            return ""

    def format_hizmet(df):
        ana_col = col(df, "ANA HİZMET", HIZMET_ALIASES["ANA HİZMET"])
        alt_col = col(df, "ALT HİZMET", HIZMET_ALIASES["ALT HİZMET"])
        ted_col = col(df, "TEDAVİLER", HIZMET_ALIASES["TEDAVİLER"])

        if ana_col or alt_col or ted_col:
            lines = []
            for _, r in df.iterrows():
                ana = str(r.get(ana_col, "")).strip() if ana_col else ""
                alt = str(r.get(alt_col, "")).strip() if alt_col else ""
                ted = str(r.get(ted_col, "")).strip() if ted_col else ""

                parts = []
                if ana: parts.append(f"Ana Hizmet: {ana}")
                if alt: parts.append(f"Alt Hizmet: {alt}")
                if ted: parts.append(f"Tedaviler: {ted}")

                if parts:
                    lines.append(" | ".join(parts))
                    return "\n\n".join(lines)
                else:
                    return ""



    return df.astype(str).to_csv(index=False)



def load_all_docs():
    texts = []
    seen  = set()               # aynı adlı dosyayı iki kez eklememek için
    dirs  = SEARCH_DIRS         # embed_search_dirs(APPDATA_DIR) ile hazırlanmış olmalı

    for d in dirs:
        try:
            if not os.path.isdir(d):
                log("[EMBED] dir skipped (missing/not dir):", d)
                continue

            files = glob.glob(os.path.join(d, "*"))
            log("[EMBED] scan dir:", d, "files=", len(files))

            for p in files:
                name = os.path.basename(p)
                if name in seen:
                    # daha önce daha yüksek öncelikli dizinden yüklenmiş
                    continue

                lp = p.lower()
                try:
                    # .csv.csv düzeltmesi (varsa)
                    if lp.endswith(".csv.csv"):
                        new_p = p[:-4]
                        os.rename(p, new_p)
                        p  = new_p
                        lp = p.lower()
                        name = os.path.basename(p)

                    # İçeriği oku
                    t = None
                    if lp.endswith(".csv"):
                        try:
                            df = read_csv_smart(p)        # DataFrame bekleriz
                            t  = df.to_csv(sep="\t", index=False)  # GENERIC: tabloyu metne çevir
                        except Exception as e:
                            log("[EMBED] read_csv_smart failed, fallback to read_excel_like:", name, repr(e))
                            t = read_excel_like(p)        # string döndürmeli
                    elif lp.endswith(".xlsx"):
                        t = read_excel_like(p)            # string
                    elif lp.endswith(".pdf"):
                        t = read_pdf(p)                   # string
                    elif lp.endswith(".txt"):
                        t = read_txt(p)                   # string
                    else:
                        log("[EMBED] skipped (unsupported):", name)
                        continue

                    # Normalleştir: DataFrame döndüyse yine tablo-metni yap
                    try:
                        import pandas as pd
                        if isinstance(t, pd.DataFrame):
                            t = t.to_csv(sep="\t", index=False)
                    except Exception:
                        pass

                    if not isinstance(t, str):
                        t = "" if t is None else str(t)

                    # Etiketle ve ekle
                    t_stripped = t.lstrip()
                    if t_stripped:
                        # URL ingest'ten gelen içerik başında [URL:] olabilir → olduğu gibi bırak
                        if t_stripped.startswith("[URL:"):
                            tagged = t_stripped
                        else:
                            tagged = f"[DOSYA:{name}]\n{t_stripped}"
                        texts.append(tagged)
                        seen.add(name)
                        log("[EMBED] loaded:", name, "chars=", len(t_stripped))
                    else:
                        log("[EMBED] skipped (empty):", name)

                except Exception as e:
                    log("[EMBED] failed to read:", name, repr(e))

        except Exception as e:
            log("[EMBED] scan error:", d, "->", repr(e))

    log("[EMBED] total_files_loaded:", len(texts))
    return texts  # List[str]



APP_NAME = "OdyoduyuChatbot"

def app_data_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", Path.home()))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d

def vectorstore_dir() -> Path:
    d = app_data_dir() / "vectorstore"
    d.mkdir(parents=True, exist_ok=True)
    return d

VSTORE_DIR = vectorstore_dir()  # LangChain save/load_local burayı kullanır





def _cache_files_exist() -> bool:
    p = Path(VSTORE_DIR)
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()

def _log_cache_sizes():
    p = Path(VSTORE_DIR)
    try:
        s1 = (p / "index.faiss").stat().st_size if (p / "index.faiss").exists() else 0
        s2 = (p / "index.pkl").stat().st_size if (p / "index.pkl").exists() else 0
        log(f"[EMBED] cache files: index.faiss={s1} bytes, index.pkl={s2} bytes")
    except Exception as e:
        log("[EMBED] cache stat fail:", repr(e))
        
        
# ==== RAG + LLM ====


VDB = None
LLM = None
PROMPT_TMPL = ""

def set_vectordb(vs):
    global VDB
    VDB = vs

FALLBACK_SYS = (
    "Sen Odyoduyu isimli bir işitme merkezinde çalışan, nazik ve profesyonel bir müşteri temsilcisisin. "
    "Yalnızca Türkçe konuş ve samimi ama ölçülü bir üslup kullan."
    
    "Elindeki bilgi kaynakları şunlardır:"
    "{context}"
    
    "Bu bağlam, şu başlıklardaki bilgileri içerebilir:"
    "- Çalışma saatleri ve randevu bilgileri (embed klasörü: çalışma_saatleri)"
    "- Adres, şube, konum bilgileri (embed klasörü: konum)"
    "- Verilen hizmetler (embed klasörü: hizmetler)"
    "- Sık sorulan sorular (embed klasörü: faq)"
    "- Web sayfalarına ait bilgiler (embed_urls klasörü)"
    
    "Cevap verirken:"
    "1. Öncelikle kullanıcının sorusunu dikkatlice anla."
    "2. Sadece {context} içinde bulunan bilgilere dayanarak cevap ver."
    "3. Aşağıdaki konularda ilgili dosyalardaki bilgiyi özellikle tercih et:"
       "- Randevu/mesai/saat konularında: [DOSYA: Çalışma Saatleri.txt] içeriğini baz al."
      " - Adres/şube/konum konularında: [DOSYA: konum.txt] içeriğini baz al."
       "- Hizmetler konularında: [DOSYA: hizmetler.txt] içeriğini baz al."
    "4. Cevabın kısa, net ve doğrudan olsun. Gereksiz tekrar ve uzun girişlerden kaçın."
    
    "ÖNEMLİ:"
    "- Bağlamda (context) yer almayan hiçbir bilgi uydurma."
    "- Tahmin yürütme, genel geçer bilgi verme, “muhtemelen” gibi ifadelerle emin olmadığın konularda yorum yapma."
    "- Eğer {context} içinde kullanıcının sorusunu yanıtlayacak yeterli bilgi yoksa, şu şekilde cevap ver:"
    "  Üzgünüm, bu konuda elimde bilgi yok. Ad Soyad ; Telefon paylaşırsanız, müşteri temsilcimiz en kısa sürede size dönüş sağlayacaktır."
      "Bu cevabı verdikten sonra sohbeti sonlandır."
    
    "Her zaman bu kurallara uyarak yanıt ver."
    
    "Soru: {question}"

)


def init_prompt_and_llm():
    global PROMPT_TMPL, LLM
    # prompt.txt'yi doğrudan oku
    p = get_resource_path("prompt.txt")
    exists = p.exists()
    log(f"[PROMPT] path={p} exists={exists}")
    if exists:
        PROMPT_TMPL = p.read_text(encoding="utf-8")
    else:
        # yedek şablon
        PROMPT_TMPL = FALLBACK_SYS
    LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    




def build_context(question: str, k: int = 6, score_threshold: float = 0.18, max_chars: int = 6000, rid: str | None = None) -> str:

    VDB = get_vdb()

    ql = question.lower()
    is_rdv  = any(w in ql for w in ["randevu", "rdv", "saat", "çalışma", "mesai"])
    is_loc  = any(w in ql for w in ["nerede", "adres", "konum", "şube", "harita"])
    # ⬇️ yeni: hizmet niyeti
    is_srv  = any(w in ql for w in [
        "hizmet", "hizmetler", "tedavi", "işlem", "uygulama",
        "vng", "kalorik", "epley", "manevra", "odyometri",
        "işitme testi", "denge", "vestibüler"
    ])
    is_faqy = any(w in ql for w in ["sıkça", "sss", "faq", "nasıl", "nedir", "ücret", "fiyat"])
    
    import uuid
    if not rid:
        rid = uuid.uuid4().hex[:8]   # <- fallback; artık her durumda var
    
    def src_of(doc):
        md = getattr(doc, "metadata", {}) or {}
        if md.get("source"):
            return md["source"]
        head = (doc.page_content[:120] if doc else "")
        if head.startswith("[DOSYA:") and "]" in head:
            return head[7:head.index("]")].strip()
        return ""

    def doc_id_of(doc):
        md = getattr(doc, "metadata", {}) or {}
        return md.get("doc_id")
    
    _bump_retrieve(rid)
    
# 1) MMR + (opsiyonel) SIM skoru
    try:
        algo = "MMR"
        mmr_docs = VDB.max_marginal_relevance_search(
            question, k=min(12, max(6, k * 2)), fetch_k=40, lambda_mult=0.25
        )
        sim_hits = VDB.similarity_search_with_relevance_scores(question, k=40)
        sim_map = {}
        for d, raw in sim_hits:
            s = raw
            if s is not None and -1.0 <= s <= 1.0:
                s = (s + 1.0) / 2.0
            did = doc_id_of(d)
            if did:
                sim_map[did] = s if s is not None else 1.0
    
        hits = []
        for d in mmr_docs:
            base = sim_map.get(doc_id_of(d), 1.0)
            hits.append((d, base))
    
        # --- EK: [RANK]/[CTX] logları ---
        _sources = []
        for d, score in hits:
            meta = getattr(d, "metadata", {}) or {}
            src  = _src_label(meta)
            prev = (getattr(d, "page_content", "") or "")[:120].replace("\n", " ")
            _sources.append(src)
#        log(f"[CTX] sources={_sources}")
        # --- EK SONU ---

    except Exception:
        algo = "SIM"
        hits = VDB.similarity_search_with_relevance_scores(question, k=max(k, 10))

    # --- EK: [RANK]/[CTX] logları (SIM yolu) ---
    _sources = []
    for d, score in hits:
        # similarity_search_with_relevance_scores genelde 0..1 döner (bazı sürümlerde -1..1 olabilir)
        meta = getattr(d, "metadata", {}) or {}
        src  = _src_label(meta)
        prev = (getattr(d, "page_content", "") or "")[:120].replace("\n", " ")
        try:
            # -1..1 ise normalize et
            if score is not None and -1.0 <= score <= 1.0:
                score = (score + 1.0) / 2.0
        except Exception:
            pass
        
        log(f"[RANK] rid={rid} total={float(score or 1.0):.3f} raw={float(score or 1.0):.3f} algo={algo} src={src} prev={prev}")
        _sources.append(src)
    ctx = build_context_from_hits(hits, max_chars=6000, rid=rid)  # <— SON [CTX] log’u bu fonksiyon içinde atılıyor
    

#    log(f"[CTX] sources={_sources}")
    # --- EK SONU ---

    # 2) Boost / demote
    scored = []
    for doc, raw in hits:
        s = 1.0 if raw is None else raw
        if s is not None and -1.0 <= s <= 1.0:
            s = (s + 1.0) / 2.0
        s = s if s is not None else 1.0

        src_l = src_of(doc).lower()
        boost = 0.0
        if is_rdv and (("çalışma" in src_l) or ("saat" in src_l) or ("personel" in src_l)):
            boost += 0.40
        if is_loc and ("konum" in src_l):
            boost += 0.30
        # ⬇️ yeni: hizmet sorularında hizmetler.txt'yi öne al
        if is_srv and ("hizmetler" in src_l):
            boost += 0.35

        # randevu/konum/hizmet odaklı sorularda faq'yı biraz frenle
        if (is_rdv or is_loc or is_srv) and ("faq" in src_l):
            boost -= 0.15
        # saf FAQ tarzı soruda faq'ya küçük artı
        if is_faqy and ("faq" in src_l):
            boost += 0.05

        total = s + boost
        scored.append((total, s, doc))

    # 3) Eşik + çeşitlilik (kaynak başı kota + round-robin)
    scored.sort(key=lambda x: x[0], reverse=True)

    from collections import defaultdict
    buckets = defaultdict(list)
    for total, raw_s, doc in scored:
        if total >= score_threshold:
            buckets[src_of(doc)].append((total, raw_s, doc))

    # Öncelikli kaynaklar (varsa önce)
    priority_wants = []
    if is_rdv: priority_wants += ["çalışma saatleri.txt", "personel.txt"]
    if is_loc: priority_wants += ["konum.txt"]
    if is_srv: priority_wants += ["hizmetler.txt"]            # ⬅️ yeni

    all_keys = list(buckets.keys())
    prios = []
    for want in priority_wants:
        for sname in all_keys:
            if sname and sname.lower() == want:
                prios.append(sname)
                break

    def avg_score(sname):
        arr = buckets.get(sname, [])
        return (sum(x[0] for x in arr) / max(1, len(arr))) if arr else 0.0
    rest = [k for k in all_keys if k not in prios]
    rest.sort(key=lambda sname: -avg_score(sname))
    order = prios + rest

    cap_per_src = 2  # istersen "faq.txt için 1" yapabilirsin
    picked, taken = [], defaultdict(int)
    idx = 0
    while len(picked) < k and order:
        sname = order[idx % len(order)]
        arr = buckets.get(sname, [])
        if arr and taken[sname] < cap_per_src:
            total, raw_s, doc = arr.pop(0)
            picked.append((total, raw_s, doc, sname, algo))
            taken[sname] += 1
            if taken[sname] >= cap_per_src or not arr:
                buckets.pop(sname, None)
                order = [o for o in order if o != sname]
                idx = 0
                continue
        idx += 1
        if idx > 1000:
            break

    if not picked and scored:
        total, raw_s, doc = scored[0]
        picked = [(total, raw_s, doc, src_of(doc), algo)]

    for total, raw_s, doc, sname, algo in picked:
        prev = (doc.page_content[:80] if doc else "")
    return ctx






import re


def is_rdv_intent(text: str) -> bool:
    t = (text or "").lower().strip()
    log("is_rdv_intent içinde")
    return (
        "randevu" in t or "rezerv" in t or "rdv" in t or "rand" in t or "rendevu" in t or
        any(w in t for w in WEEKDAY_WORDS) or
        bool(TIME_RE.search(t)) or
        t in CONFIRM_WORDS or t in CANCEL_ONLY_KEYWORDS or t in RESCHEDULE_ONLY_KEYWORDS
    )




def _within(h, mi, start=(9,0), end=(18,0)):
    return (h, mi) >= start and (h, mi) <= end







def _safe_llm(prompt: str) -> str:
    try:
        resp = LLM.invoke([{"role": "user", "content": prompt}])
        text = getattr(resp, "content", None)
        if not text and hasattr(resp, "choices"):
            ch0 = resp.choices[0] if resp.choices else None
            if ch0 and hasattr(ch0, "message"):
                text = getattr(ch0.message, "content", None)
        return (text or "").strip()
    except Exception as e:
        log("[LLM] call failed:", repr(e))
        return ""
    
# --- YARDIMCILAR: onay tespiti + stage kontrolü ---

import re


def _norm_tr(s: str) -> str:
    # sende zaten vardır; yoksa basit normalize:
    return (s or "").lower().replace("ş","s").replace("ı","i").replace("ğ","g").replace("ö","o").replace("ü","u").replace("ç","c")

_CHANGE_RE = re.compile(
    r"\b("
    r"degistir|degistirmek istiyorum|farkli bir randevu|baska saat|baska gun|"
    r"gunu degistir|saati degistir|randevuyu degistir"
    r")\b"
)

def _is_change(text: str) -> bool:
    return bool(_CHANGE_RE.search(_norm_tr(text)))


# Türkçe karakterleri ASCII'ye çevir (ı→i, ş→s, ğ→g ...)
_TR_MAP = str.maketrans({
    "ı":"i","İ":"i","ş":"s","Ş":"s","ğ":"g","Ğ":"g","ç":"c","Ç":"c","ö":"o","Ö":"o","ü":"u","Ü":"u"
})

def _norm_tr(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.translate(_TR_MAP)          # ⬅️ kritik: TR harflerini sadeleştir
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)    # noktalama/emoji→boşluk
    s = re.sub(r"\s+", " ", s).strip()
    return s

_QPART_RE = re.compile(r'\b(mi|mi|mu|mu)\b')  # normalize sonrası zaten ascii
_BUT_RE   = re.compile(r'\b(ama|ancak|yalniz)\b')

_YES_TOKS = {
    "evet","ok","okey","tamam","tamamdir","peki","aynen","olur",
    "uygun","uygundur","oldu","kabul","kabuldur"
}
_NO_TOKS = {
    "hayir","iptal","vazgec","vazgectim","degistir","olmaz",
    "uygundegil","istemiyorum"
}

# normalize SONRASI eşleşecek kalıplar
_NO_PAT  = re.compile(r'\b(iptal\s*et|uygun\s*degil|onaylam[ay]?(?:or|orum)?|istemiyorum)\b')
_YES_PAT = re.compile(r'\b(onayliyorum|onayladim|onaylarim|onay\s*ver(?:iyorum)?|kabul\s*ed(?:iyorum)?)\b')

def _looks_like_question(msg: str) -> bool:
    t = msg or ""
    return ("?" in t) or bool(_QPART_RE.search(_norm_tr(t)))

def _is_no(msg: str) -> bool:
    t = _norm_tr(msg)
    toks = set(t.split())
    if toks & _NO_TOKS: return True
    if _NO_PAT.search(t): return True
    return False

def _is_yes(msg: str) -> bool:
    if _is_no(msg): return False
    if _looks_like_question(msg): return False
    if _BUT_RE.search(_norm_tr(msg)): return False

    t = _norm_tr(msg)
    toks = set(t.split())
    if toks & _YES_TOKS: return True
    if _YES_PAT.search(t): return True
    return False

# intent
_STAFF_Q_RE = re.compile(r'(hangi\s+(hoca|uzman|doktor)|kim\s+(ilgilen(ecek|ir)|bakacak))', re.IGNORECASE)
def _is_staff_intent(q: str) -> bool:
    return bool(_STAFF_Q_RE.search(q or ""))

def set_stage(sid: str, stage: str | None):
    slots = get_slots(sid)
    slots["stage"] = stage  # stage'i None yapabilmek için ayrı setter

def _mask_phone(p: str | None) -> str:
    if not p: return "kayıtlı numaranız"
    return p[:-2] + "**" if len(p) > 2 else p

def _ensure_hm(val):
    """(h, m) bekler; string '12:30' gelirse parse eder; yoksa None döner."""
    if not val:
        return None
    # tuple/list ise
    if isinstance(val, (tuple, list)) and len(val) == 2:
        try:
            return int(val[0]), int(val[1])
        except Exception:
            return None
    # '12:30' veya '12.30' gibi string ise
    import re
    m = re.match(r'^\s*(\d{1,2})[:.](\d{2})\s*$', str(val))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def _safe2(ret, fallback_reply="Bir şeyler ters gitti. Lütfen tekrar dener misiniz?"):
    """
    run_planner / on_user_* dönüşleri için güvenli açma.
    Tuple(reply, ctx) döner; değilse (fallback, Ctx()) döndürür ve loglar.
    """
    from datetime import datetime
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret
    log(f"[SAFE2] unexpected return={type(ret).__name__} at {datetime.now().isoformat()}")
    return fallback_reply, (Ctx() if 'Ctx' in globals() else None)


def answer(question: str, sid: str, kvkk_ok: bool = False) -> str:
    """
    Randevu FSM + fiyat / bilgi / KVKK akışı + genel LLM akışı.
    """
    import re
    import uuid
    
    rid = f"{uuid.uuid4().hex[:8]}" 

    # ---- Hazırlık ----
    past = history_as_text(sid)
    log("[answer]:[question]", question)
    log("[past]:", past)
    log("kvkk_ok", kvkk_ok)

    # 0) Her mesajdan alanları çıkar → slotlara yaz
    update_slots_from_text(sid, question)
    slots = get_slots(sid)  # {"name","phone","day","time","service","language"?}

    # --- SLOTS -> Ctx köprüsü (kimlik aktar) ---
    ctx = SESS.get(sid) or Ctx()
    try:
        if slots.get("name") and not is_valid_fullname(ctx.goal.customer.get("fullName")):
            set_fullname(ctx, slots["name"], source="slots")
        if slots.get("phone") and not ctx.goal.customer.get("phone"):
            ctx.goal.customer["phone"] = slots["phone"]
        # service slot'unu da Ctx içine taşıyabilirsin (isteğe bağlı)
        if slots.get("service"):
            ctx.goal.customer["service"] = slots["service"]
    except Exception:
        pass
    SESS[sid] = ctx

    # ---- META / FORCE-WAIT KONTROLÜ ----
    meta = getattr(ctx, "meta", {}) or {}
    force_wait_contact = bool(meta.get("force_wait_contact", False))
    
    # Kullanıcı gerçekten iletişim bilgisi veriyor mu?
    contact_like = (";" in question)
    try:
        if not contact_like and "parse_phone_any" in globals():
            contact_like = parse_phone_any(question.lower()) is not None
    except Exception:
        pass
    
    # Eğer daha önce "Ad Soyad ; Telefon ..." cevabı verildiyse
    # ve hâlâ iletişim bilgisi gelmediyse → aynı mesaja yönlendir.
    if force_wait_contact and not contact_like:
        log("[FORCE_WAIT_CONTACT] still waiting for contact info")
        return (
            # "KVKK Aydınlatma Metni’ni okuduğunuzu ve onayladığınızı belirten,\n"
            "KVKK kutusunu işaretleyip, Ad Soyad ; Telefon \n "
            "bilgilerinizi örnekteki gibi paylaşırsanız (Örn: Ad Soyad ; 05XX XXXXXXX), \n" 
            "müşteri temsilcimiz en kısa sürede size dönüş sağlayacaktır."
        )
    
    # Eğer force_wait_contact açıksa ve artık iletişim bilgisi geldiyse:
    if force_wait_contact and contact_like:
        # 1) KVKK YOKSA: BİLGİYİ ALMA, UYAR
        if not kvkk_ok:
            log("[FORCE_WAIT_CONTACT] contact-like but no KVKK")
            return (
                "Kişisel bilgilerinizi almadan önce KVKK Aydınlatma Metni’ni onaylamanız gerekiyor.\n "
                "Lütfen sohbet penceresinin altındaki KVKK kutusunu işaretleyip, \n"
                "Ad Soyad ; Telefon ; (ve varsa hizmet) formatında tekrar yazar mısınız?"
            )
    
        # 2) KVKK VARSA: bayrağı kapat, normal akışa devam et
        meta["force_wait_contact"] = False
        ctx.meta = meta
        SESS[sid] = ctx
        log("[FORCE_WAIT_CONTACT] contact info received WITH KVKK, resuming normal flow")


    # Küçük yardımcı
    def _mask_phone(p: str | None) -> str:
        if not p:
            return "kayıtlı numaranız"
        return p[:-2] + "**" if len(p) > 2 else p

    # --- Randevu / KVKK Akışı ---
    ctx = SESS.get(sid)
    tnorm = question.lower().strip()

    # Daha önce davet edildi mi?
    invited = False
    try:
        meta = getattr(ctx, "meta", {}) or {}
        invited = bool(meta.get("invited_to_schedule"))
    except Exception:
        invited = False

    # Kimlik (ad+telefon) tamam mı?
    needs_identity = not identity_complete(ctx)

    # --- PENDING FLAG GET (KVKK sonrası randevu) ---
    pending = meta.get("pending_rdv_after_kvkk", False)

    # ADIM 1: Randevu isteği + KVKK yok → KVKK iste
    if is_rdv_intent(tnorm) and not kvkk_ok:
        meta["pending_rdv_after_kvkk"] = True
        ctx.meta = meta
        SESS[sid] = ctx
        return (
            "Randevu oluşturabilmemiz için KVKK Aydınlatma Metni’ni okuduğunuzu ve \n"
            "kişisel verilerinizin işlenmesini onayladığınızı belirtmeniz gerekiyor. \n"
            "Lütfen kutucuğu işaretleyerek onay verin ve ardından **“Onayladım”** diye yazın."
        )

    # Kimlik tekrar kontrol
    needs_identity = not identity_complete(ctx)

    # ADIM 2: KVKK onaylandı + pending + kimlik eksik → Ad Soyad ; Tel ; Hizmet iste
    if kvkk_ok and pending and needs_identity:
        meta["pending_rdv_after_kvkk"] = False  # flag reset
        ctx.meta = meta
        SESS[sid] = ctx

        return (
            "Teşekkürler, KVKK onayınız alındı. Lütfen Adınızı Soyadınızı ve Telefon Numaranızı örnekteki gibi yazar mısınız?\n"
            "(Örn Ad Soyad ; 5xx xxxxxxx ; İlgilendiğiniz Hizmet)  (örneğin: İşitme Testi,İşitme Cihazı Denemesi,Genel Değerlendirme vb.)"
        )

    # ADIM 3: Kullanıcı bilgi yazdı → slotları güncelle (zaten başta yaptık, sadece yeniden çekelim)
    slots = get_slots(sid)

    name = slots.get("name")
    phone = slots.get("phone")
    service = slots.get("service")
    language = slots.get("language")
    log("[SLOTS_IN_ANSWER] name=%r phone=%r service=%r language=%r" % (name, phone, service, language))

    # --- Fiyat / genel bilgi lead tipi (service yoksa) ---
    generic_service = None
    if not service:
        # Fiyatla ilgili soru mu?
        if any(w in tnorm for w in ["fiyat", "ücret", "ne kadar", "kaç tl", "kaça", "ücreti"]):
            generic_service = "Fiyat Bilgisi Talebi"
        else:
            generic_service = "Genel Bilgi Talebi"

    # ----------------------------------------------------
    # 1) HİZMET BİLGİSİ VARSA → HİZMET BAZLI LEAD
    # ----------------------------------------------------
    if kvkk_ok and name and phone and service:
        # from common_log import log

        res = ensure_crm_lead_from_chat(
            full_name=name,
            phone=phone,
            service=service,
            language=language,
        )
        log("[answer][ensure_crm_lead_from_chat][with_service] res =", res)

        if res.get("lead") and res.get("reason") == "ok":
            return (
                f"Teşekkürler Sayın {name} 🙏\n\n"
                f"'{service}' ile ilgili talebinizi başarıyla oluşturduk. "
                f"En kısa sürede {phone} numarasından sizinle iletişime geçeceğiz. "
                "Görüşmek üzere."
            )
        else:
            return (
                f"Teşekkürler Sayın {name} 🙏\n\n"
                f"'{service}' ile ilgili talebinizi aldım ancak sistemimizde kayıt oluştururken teknik bir sorun yaşandı. "
                f"Endişe etmeyin, ekip arkadaşlarımız yine de {phone} numarasından en kısa sürede size dönüş yapacaktır."
            )

    # ----------------------------------------------------
    # 2) SADECE AD + TELEFON VAR (service yok) → 
    #    'Üzgünüm...' ve fiyat senaryoları için LEAD
    # ----------------------------------------------------
    log("kvkk ok:" , kvkk_ok, "name:", name, " phone:", phone, "service:",service )
    if kvkk_ok and name and phone and not service:
        # from common_log import log

        # Burada service yerine "Fiyat Bilgisi Talebi" veya "Genel Bilgi Talebi"
        res = ensure_crm_lead_from_chat(
            full_name=name,
            phone=phone,
            service=service,
            language=language,
        )
        log("[answer][ensure_crm_lead_from_chat][no_service] res =", res)

        # Fiyat / genel bilgi için mesaj
        if res.get("lead") and res.get("reason") == "ok":
            return (
                f"Teşekkürler Sayın {name} 🙏\n\n"
                f"Talebinizi başarıyla kaydettim"
                f" ({generic_service}). "
                f"En kısa sürede {phone} numarasından sizinle iletişime geçeceğiz."
            )
        else:
            return (
                f"Teşekkürler Sayın {name} 🙏\n\n"
                f"Talebinizi aldım"
                f" ({generic_service}), ancak sistemimizde kayıt oluştururken teknik bir sorun yaşandı. "
                f"Endişe etmeyin, ekip arkadaşlarımız yine de {phone} numarasından en kısa sürede size dönüş yapacaktır."
            )


    # --- Normal LLM / RAG akışı ---
    log(f"[REQ] rid={rid} q='{question[:80]}'")
    ctx_text = build_context(question, rid=rid)

    filled = (
        "Önceki konuşma (kısa):\n" + (past or "—") + "\n\n" +
        PROMPT_TMPL.format(context=ctx_text, question=question)
    )
    cnt = RETRIEVE_COUNT.get(rid, 0)
    log(f"[REQ] rid={rid} retrieve_calls={cnt}")
    RETRIEVE_COUNT.pop(rid, None)

    reply = _safe_llm(filled) or "Mesajınızı aldım. Daha iyi yardımcı olabilmem için biraz daha detay verebilir misiniz?"

    # --- FORCE_WAIT_CONTACT BAYRAĞINI LLM CEVABINA GÖRE AYARLA ---
    # Eğer LLM, fiyat/bağlam dışı durumda "Ad Soyad ; Telefon paylaşırsanız..." içeren bir cevap verdiyse
    # bir sonraki mesajda yalnızca iletişim bilgisi bekle.
    if "Ad Soyad ; Telefon paylaşırsanız" in reply:
        ctx = SESS.get(sid) or Ctx()
        meta = getattr(ctx, "meta", {}) or {}
        meta["force_wait_contact"] = True
        ctx.meta = meta
        SESS[sid] = ctx
        log("[FORCE_WAIT_CONTACT] set to True due to LLM reply")

    # --- Konuşma geçmişini güncelle ---
    h = get_history(sid)
    h.append(("user", question))
    h.append(("assistant", reply))
    log(f"[HIST] sid={sid} len={len(h)} reply_len={len(reply)}")
    log("[Reply] : ", reply)

    return reply




VDB = None  # ⬅️ GLOBAL



def build_or_load_vectorstore(force_rebuild: bool = False):
    """Her çağrıda GLOBAL VDB'yi günceller ve döner."""
    import re
    from collections import Counter
    global VDB

    # --- Embedding objesi ---
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # 1) Cache varsa yükle
    if not force_rebuild and _cache_files_exist():
        try:
            log("[EMBED] try cache load")
            VDB = FAISS.load_local(VSTORE_DIR, emb, allow_dangerous_deserialization=True)
            log("[EMBED] index loaded from cache")
            return VDB
        except Exception as e:
            log("[EMBED] cache load failed, will rebuild:", repr(e))

    # --------- Yardımcılar (YOL B için) ----------
    def _extract_tags_to_meta(s: str):
        """
        Metin başındaki [TAG:deger] etiketlerini metadata'ya alır, gövdeyi temiz döndürür.
        Örn: [URL:https://x]\nLorem...
        """
        meta = {}
        pos = 0
        # birden çok etiketi ardışık okur (örn [URL:..][TITLE:..] ...)
        while True:
            m = re.match(r'\[(\w+):(.*?)\]\s*', s[pos:], flags=re.S)
            if not m:
                break
            tag = m.group(1).upper()
            val = m.group(2).strip()
            pos += m.end()
            if tag == "URL":
                meta["source_url"] = val
                meta.setdefault("source_type", "url")
            elif tag == "FILE":
                meta["source_path"] = val
                meta.setdefault("source_type", "file")
            elif tag == "TITLE":
                meta["title"] = val
            elif tag == "ID":
                meta["id"] = val
            else:
                meta[tag.lower()] = val
        body = s[pos:].strip()
        return body, meta

    # metin listesi + meta listesi üret
   # dosyanın başlarında:
   

    def _to_texts_and_metas(named_list):
        clean_texts, metas = [], []
        for i, item in enumerate(named_list or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name, text = item[0], item[1]
            else:
                name, text = f"doc::{i}", str(item) if item is not None else ""
    
            if not text or not str(text).strip():
                continue
    
            body, meta = _extract_tags_to_meta(str(text))
    
            # [URL:...] etiketi yoksa, name'den türet
            if "source_url" not in meta and "source_path" not in meta:
                if isinstance(name, str) and name.startswith(("http://", "https://")):
                    meta["source_url"] = name
                    meta.setdefault("source_type", "url")
                else:
                    # name bir dosya yolu/adı ise
                    meta["source_path"] = name
                    meta["filename"] = os.path.basename(name)
                    meta.setdefault("source_type", "file")
    
            clean_texts.append(body)
            metas.append(meta)
        return clean_texts, metas


    # Basit string tabanlı chunk'lama (istersen sonra ayarlarsın)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # eski sürüm uyumu
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _split_texts_with_meta(texts, metas):
        chunked_texts, chunked_metas = [], []
        for t, m in zip(texts, metas):
            chunks = splitter.split_text(t) if t else []
            if not chunks:
                chunks = [t]  # en az bir parça olsun
            for ch in chunks:
                chunked_texts.append(ch)
                chunked_metas.append(m)
        return chunked_texts, chunked_metas

    def _dbg_types(name, seq):
        c = Counter(type(x).__name__ for x in (seq or []))
        log(f"[EMBED][DBG] {name}: n={len(seq or [])} types={dict(c)}")

    # 2) Rebuild: önce lokal dosyalar
    named_texts = load_all_docs()  # beklenen: [(name, text)] ama string listesi de gelebilir
    if not isinstance(named_texts, list) or (named_texts and not isinstance(named_texts[0], (list, tuple, str))):
        # Güvence: tamamen farklı bir tip dönerse güvenli hale getir
        named_texts = [(f"doc::{i}", str(t)) for i, t in enumerate(named_texts or [])]

    # 2b) URL içeriklerini ekle
    try:
        url_named_texts = _ingest_urls_in_dirs(max_urls=200)  # sende bazen [str] bazen [(name,text)] olabilir
        if url_named_texts:
            # Eğer saf str listesiyse name ekle
            if isinstance(url_named_texts[0], str):
                url_named_texts = [(f"url::{i}", s) for i, s in enumerate(url_named_texts)]
            named_texts.extend(url_named_texts)
    except Exception as e:
        log("[INGEST] urls skipped due to error:", repr(e))

    # 3) (YOL B) String -> (clean_text, meta) ayır
    clean_texts, metas = _to_texts_and_metas(named_texts)
    _dbg_types("clean_texts", clean_texts)

    # 3b) Chunk'la
    chunked_texts, chunked_metas = _split_texts_with_meta(clean_texts, metas)
    log(f"[EMBED] from_texts: n_texts={len(chunked_texts)}")
    assert len(chunked_texts) == len(chunked_metas), "texts/metas boyutları eşleşmiyor"

    # 4) FAISS index oluştur/kaydet (YOL B: from_texts)
    os.makedirs(VSTORE_DIR, exist_ok=True)
    # Not: FAISS.from_texts signature: (texts, embedding, metadatas=None, ...)
    VDB = FAISS.from_texts(chunked_texts if chunked_texts else [""], emb,
                           metadatas=chunked_metas if chunked_metas else [{}])
    VDB.save_local(VSTORE_DIR)
    _log_cache_sizes()
    log("[EMBED] index built & saved")
    return VDB



def get_vdb():
    """Her yerden güvenle çağır; yoksa yükler."""
    global VDB
    if VDB is None:
        return build_or_load_vectorstore(False)
    return VDB




vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k":4}) 

# NEW: RAG yardımcılarını hazırla
set_vectordb(vectordb)
init_prompt_and_llm()



@app.route("/chatbot_embed.html")
def chatbot_embed():
    return render_template("chatbot_embed.html")








@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()

        # ✅ SID’i burada normalize et / üret
        sid = get_session_id(data.get("session_id"))
        kvkk_ok = bool(data.get("kvkk_ok", False))

        if not user_message:
            return jsonify({"ok": False, "reply": "", "error": "Boş mesaj"}), 400

        # ✅ Aynı SID’i answer'a geçir
        bot_reply = (answer(user_message, sid, kvkk_ok=kvkk_ok) or "").strip()
        if not bot_reply:
            bot_reply = "Merhaba! Mesajınızı aldım. Nasıl yardımcı olabilirim?"

        log(f"[/chat] reply_len={len(bot_reply)} sid={sid}")
        return jsonify({"ok": True, "reply": bot_reply, "session_id": sid}), 200

    except Exception as e:
        import traceback
        log("[/chat] EXC:", repr(e))
        log(traceback.format_exc())
        return jsonify({"ok": False, "reply": "", "error": repr(e)}), 500


    

# app_embedding_crm_prompt.py (veya backend dosyan)
from web_ingest import fetch_from_urls_file

@app.route("/reindex", methods=["POST"])
def reindex():
    global VDB
    try:
        log(f"[REINDEX] start EMBED_DIR={EMBED_DIR}")
        urls_file = os.path.join(EMBED_DIR, "urls.txt")  # örn: dist\desktop_app\embeds\urls.txt
        log("URLS_FILE", urls_file)
        saved = fetch_from_urls_file(urls_file=urls_file, out_dir=EMBED_DIR)
        log(f"[REINDEX] ingest saved={len(saved)}")
    except Exception as e:
        log("[REINDEX] ingest error:", repr(e))

    VDB = build_or_load_vectorstore(force_rebuild=True)
    return jsonify({"status": "ok", "ingested": len(saved) if 'saved' in locals() else 0})


# Sağlık kontrolü (webview açmadan önce ping edeceğiz)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

# Ana sayfa (HTML’yi kökten servis edelim)
@app.route("/", methods=["GET"])
def index():
    return render_template("chatbot_embed.html")

@app.route("/debug/parse_contact", methods=["POST"])
def debug_parse_contact():
    data = request.get_json(force=True) or {}
    txt = data.get("text", "")
    parsed = parse_contact_answer_llm(txt)
    return jsonify(parsed)



def bootstrap_diag():
    log("[DIAG] bootstrap start")
    try:
        # 1) Vektör veritabanını hazırla
        vs = get_vectorstore()
        set_vectordb(vs)
        log("[DIAG] vectorstore ready")

        # 2) prompt.txt + LLM’i hazırla
        init_prompt_and_llm()

        # 3) prompt dosyası gerçekten var mı?
        p = get_resource_path("prompt.txt")
        log(f"[DIAG] prompt path={p} exists={p.exists()}")

        # 4) Kısa bir self-test
        try:
            resp = answer("çalışma saatleriniz nedir?")
            log("[DIAG] selftest answer:", (resp or "")[:200])
        except Exception as e:
            import traceback
            log("[DIAG] selftest failed:", repr(e))
            log(traceback.format_exc())

    except Exception as e:
        import traceback
        log("[DIAG] fatal:", repr(e))
        log(traceback.format_exc())


def _start_backend():
    # RAG başlangıcı (tek yerde toplayalım)
    vs = get_vectorstore()
    set_vectordb(vs)
    init_prompt_and_llm()
    # (opsiyonel) bootstrap_diag()
    # bootstrap_diag()

def run_server():
    ensure_dirs_and_seed()   # <<< ÖNEMLİ
    # ... sonra vectorstore build_or_load ve app.run ...
    try:
        _log_paths()  # ← EMBED/URLS dizinleri nereden okunuyor, burada bir kez log'a düşer
    except Exception as e:
        log("[PATHS] log failed:", repr(e))
    # Flask'ı tek proses, tek thread: reloader kapalı
    _start_backend()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    # Modülü direkt çalıştırırsan da aynı yolu izlesin
      run_server()



