# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 13:38:31 2025

@author: Serkan
"""

# mvp_agentic_appointments.py
# MVP randevu akışı — tek hekim, generic CRM uçları.
# Plan→Act→Observe→Replan döngüsü ile semi-agentic davranış.
# Python 3.10+, sadece standart kütüphaneler.


from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time, date
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo
import os, re, uuid
from common_log import log

from datetime import timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError




def get_istanbul_tz():
    try:
        return ZoneInfo("Europe/Istanbul")
    except ZoneInfoNotFoundError:
        try:
            import tzdata  # kurulu olduğundan emin ol
            return ZoneInfo("Europe/Istanbul")
        except Exception:
            # Son çare: kalıcı UTC+3 (yaz/kış saati yok)
            return timezone(timedelta(hours=3))

IST = get_istanbul_tz()

# =========================
# Konfig (MVP)
# =========================
SLOT_SIZE_MIN   = int(os.getenv("SLOT_SIZE_MIN", "15"))
LEAD_TIME_MIN   = int(os.getenv("LEAD_TIME_MIN", "120"))
WEEKEND_ALLOWED = os.getenv("WEEKEND_ALLOWED", "false").lower() == "true"
MAX_REPLANS     = int(os.getenv("MAX_REPLANS", "2"))
HOLD_TTL_SEC    = int(os.getenv("HOLD_TTL_SEC", "300"))

# Tek hekim
RESOURCE_ID     = int(os.getenv("RESOURCE_ID", "1"))

# Çalışma saatleri (hafta içi 09:00-17:00), öğle arası 12:30-13:30
WORKING_HOURS: Dict[int, List[Tuple[time, time]]] = {
    0: [(time(9,0), time(17,0))],
    1: [(time(9,0), time(17,0))],
    2: [(time(9,0), time(17,0))],
    3: [(time(9,0), time(17,0))],
    4: [(time(9,0), time(17,0))],
}
BREAKS: Dict[int, List[Tuple[time, time]]] = {
    0: [(time(12,30), time(13,30))],
    1: [(time(12,30), time(13,30))],
    2: [(time(12,30), time(13,30))],
    3: [(time(12,30), time(13,30))],
    4: [(time(12,30), time(13,30))],
}
HOLIDAYS: set[str] = set()  # "2025-08-30" gibi tarihler ekleyebilirsin



# Şube isimleri → resourceId eşlemesi (örnek)
BRANCHES = {
    "kozyatağı": 1,
    "bakırköy": 2,
    "göztepe": 3,
    "şişli torun center": 4,
    "mecidiyeköy":5
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

import unicodedata
import re


_TURKISH_TRANS = str.maketrans({
    "ı":"i","İ":"i","ş":"s","Ş":"s","ğ":"g","Ğ":"g",
    "ç":"c","Ç":"c","ö":"o","Ö":"o","ü":"u","Ü":"u",
})
def _norm_tr(s: str) -> str:
#    print("_norm_tr", s)
    s = (s or "").casefold().translate(_TURKISH_TRANS)
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def _tokens_tr(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm_tr(s)) if t]

# --- yaklaşık saat haritası (normalize edilmiş anahtarlarla) ---
_PART_OF_DAY = {
    "sabah":   "10:00",
    "ogle":    "12:00",   # öğle/öğlen
    "oglen":   "12:00",
    "ikindi":  "16:00",
    "aksam":   "18:00",   # akşam
    "gece":    "21:00",
}
# Hafta günü eşlemeleri (0=Mon..6=Sun)
_DOW = {
    "pazartesi": 0, "pzt": 0, "sali": 1, "salı": 1, "carsamba": 2, "çarşamba": 2,
    "persembe": 3, "perşembe": 3, "cuma": 4, "cumartesi": 5,"cmt": 5,"pazar": 6,
}
_DOW_NORM = { _norm_tr(k): v for k, v in _DOW.items() }


def find_weekday_idx(text: str):
    n = _norm_tr(text)
    log("find_weekday_idx")
    log("n:",n)
    for token, idx in sorted(_DOW_NORM.items(), key=lambda kv: -len(kv[0])):
        if token in n:   # "cumartesi" önce denenir, "cuma"ya sıra gelmez
            return idx
    return None

def _next_weekday_from(today, target_idx: int) -> str:
    log("_next_weekday_from içinde")
    log("today",today)
    log("target_idx",target_idx)
    log(WEEKDAYS_TR[today.weekday()])
    today_idx = find_weekday_idx(WEEKDAYS_TR[today.weekday()]) 
    log("today_idx", today_idx)
    delta = (target_idx - today_idx) % 7
    log("delta",delta)
    if delta == 0:  # aynı gün yazıldıysa bir SONRAKİNİ al
        delta = 7
    return (today + timedelta(days=delta)).isoformat()

def _title_words_tr(s: str) -> str:
    # her kelimenin ilk harfini büyüt; geri kalanı olduğu gibi bırak (diakritikler korunur)
    import re
    parts = re.split(r'(\s+)', (s or '').strip())
    return ''.join(p if p.isspace() else (p[:1].upper() + p[1:]) for p in parts if p != '')

def resolve_rid_from_display_any(mapping: dict, display: str):
    """Görünen ad (örn. 'Kozyatağı') → RID. mapping:
       - {'kozyatagi': ('kozyatağı', 1), ...}   # BRANCH_INDEX
       - {'kozyatagi': 1, ...}                  # BRANCHES
       - {'kozyatagi': {'display':'kozyatağı','rid':1}, ...}
    """
    if not display or not mapping:
        return None
    disp_norm = normalize_tr(display)
    for k, v in mapping.items():
        # tuple/list -> (display, rid)
        if isinstance(v, (tuple, list)) and v:
            disp = v[0]; rid = v[1] if len(v) > 1 else None
            if normalize_tr(disp) == disp_norm or normalize_tr(k) == disp_norm or k == disp_norm:
                return rid
        # dict -> {'display':..., 'rid'/'id'/'resourceId':...}
        elif isinstance(v, dict):
            disp = v.get('display') or v.get('name') or k
            rid  = v.get('rid') or v.get('id') or v.get('resourceId')
            if normalize_tr(disp) == disp_norm or normalize_tr(k) == disp_norm or k == disp_norm:
                return rid
        else:
            if normalize_tr(k) == disp_norm or k == disp_norm:
                return v
    return None

def _iter_branches(branch_index: Dict) -> list[tuple[str,str,Optional[str]]]:
    """BRANCH_INDEX'i tek biçime getir: (key, display, rid)."""
    items = []
    for k, v in branch_index.items():
        display, rid = None, None
        if isinstance(v, (tuple, list)) and v:
            display = v[0]
            rid = v[1] if len(v) > 1 else None
        elif isinstance(v, dict):
            display = v.get("display") or v.get("name") or k
            rid = v.get("id") or v.get("rid") or v.get("resourceId")
        else:
            display = k
            rid = v
        # Hepsi lower ise ilk harfi büyüt (ör: 'kozyatağı' -> 'Kozyatağı')
        if isinstance(display, str) and display and display == display.lower():
            display = display[:1].upper() + display[1:]
        items.append((k, display, rid))
    return items

def _resolve_branch_display(text: str, branch_index: Dict[str, object]) -> Optional[str]:
    """Metinden bulunursa BRANCH_INDEX'teki **görünen adı** döndür."""
    txt_norm = _norm_tr(text)
    txt_tokens = set(_tokens_tr(text))
    candidates = _iter_branches(branch_index)

    # 1) tam kelime (token) eşleşmesi
    for k, disp, _ in candidates:
        if _norm_tr(disp) in txt_tokens or _norm_tr(k) in txt_tokens:
            return disp
    # 2) kelime-sınırı benzeri
    for k, disp, _ in candidates:
        for probe in (_norm_tr(disp), _norm_tr(k)):
            if re.search(rf"(?:^|[^a-z0-9_]){re.escape(probe)}(?:[^a-z0-9_]|$)", txt_norm):
                return disp
    # 3) esnek alt dize
    for k, disp, _ in candidates:
        if _norm_tr(disp) in txt_norm or _norm_tr(k) in txt_norm:
            return disp
    return None

def _merge_held(state, **kwargs):
    """state.held sözlüğünü, eski değerleri (appointment_id vs.) koruyarak günceller."""
    prev = state.held if isinstance(state.held, dict) else {}
    state.held = {**prev, **kwargs}
    
# def _extract_appt_id(held) -> str | None:
#     """state.held içinden appointment id'yi güvenle çıkar (dict/obj/tuple destekli)."""
#     if not held:
#         return None
#     # tuple/list -> ilk ögeyi deneyelim
#     if isinstance(held, (tuple, list)) and held:
#         held = held[0]
#     # dict
#     if isinstance(held, dict):
#         return held.get("appointment_id") or held.get("id") or held.get("appt_id")
#     # obje gibi
#     if hasattr(held, "appointment_id"):
#         return getattr(held, "appointment_id")
#     if hasattr(held, "id"):
#         return getattr(held, "id")
#     return None

def _extract_appt_id(held) -> str | None:
    log("_extract_appt_id içinde", held)
    if not held:
        return None
    if isinstance(held, (tuple, list)) and held:
        held = held[0]
        log("held=", held)
    if isinstance(held, dict):
        for k in ("appointment_id", "appointmentId", "appt_id", "apptId", "id"):
            v = held.get(k)
            log("v=",v)
            if v:
                return v
        return None
    # obje
    for k in ("appointment_id", "appointmentId", "appt_id", "apptId", "id"):
        if hasattr(held, k):
            v = getattr(held, k)
            log("v=",v)
            if v:
                return v
    d = getattr(held, "__dict__", None)
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() in ("appointment_id", "appointmentid", "appt_id", "apptid", "id"):
                log("v=",v)
                return v
    return None

def _extract_expires_sec(held) -> int | None:
    """
    Çeşitli alan adlarını destekle: expires_in, expiresIn, expiresInSec, ttl, ttlSec, timeout, expires_at/At (ISO).
    Bulunamazsa None döner.
    """
    import datetime as _dt

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return None

    if not held:
        return None

    # dict
    if isinstance(held, dict):
        for k in ("expires_in", "expiresIn", "expiresInSec", "ttl", "ttlSec", "timeout", "hold_ttl"):
            if k in held and held[k] is not None:
                v = _as_int(held[k])
                if v is not None:
                    return v
        # expires_at / expiresAt → ISO’dan saniye hesapla
        for k in ("expires_at", "expiresAt", "holdExpiresAt"):
            if k in held and held[k]:
                try:
                    # ISO string'i aware/naive olabilir; fromisoformat çoğu vakayı çözer
                    target = _dt.datetime.fromisoformat(str(held[k]).replace("Z", "+00:00"))
                    now = _dt.datetime.now(target.tzinfo) if target.tzinfo else _dt.datetime.now()
                    sec = int(max(0, (target - now).total_seconds()))
                    return sec
                except Exception:
                    pass
        return None

    # obje benzeri
    for k in ("expires_in", "expiresIn", "expiresInSec", "ttl", "ttlSec", "timeout", "hold_ttl"):
        if hasattr(held, k):
            v = _as_int(getattr(held, k))
            if v is not None:
                return v
    for k in ("expires_at", "expiresAt", "holdExpiresAt"):
        if hasattr(held, k):
            try:
                import datetime as _dt
                target = _dt.datetime.fromisoformat(str(getattr(held, k)).replace("Z", "+00:00"))
                now = _dt.datetime.now(target.tzinfo) if target.tzinfo else _dt.datetime.now()
                return int(max(0, (target - now).total_seconds()))
            except Exception:
                pass
    return None

import re

DATE_WORDS = {
    "bugün","bugun","yarın","yarin","haftaya","ertesi","öbür",
    "pazartesi","salı","sali","çarşamba","carsamba","perşembe","persembe","cuma","cumartesi","pazar",
    "bugün için","yarın için"
}
MONTH_WORDS = {"ocak","şubat","subat","mart","nisan","mayıs","mayis","haziran","temmuz","ağustos","agustos","eylül","eylul","ekim","kasım","kasim","aralık","aralik"}
DATE_PAT = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\s*(ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik))\b",
    re.I
)

def has_explicit_date_tr(txt: str) -> bool:
    if not txt: return False
    t = txt.casefold()
    if any(w in t for w in DATE_WORDS): return True
    if any(m in t for m in MONTH_WORDS): return True
    if DATE_PAT.search(txt): return True
    return False

def _date_only(val) -> str:
    """'YYYY-MM-DD' döndürür; ISO datetime gelirse günü kırpar."""
    if val is None:
        return ""
    from datetime import datetime as _dt
    if isinstance(val, _dt):
        return val.date().isoformat()
    s = str(val).strip()
    return s.split("T", 1)[0]

def _hhmm_only(val) -> str:
    """'HH:MM' döndürür; 'HH:MM:SS' veya boşluklu varyantlar gelirse düzeltir."""
    import re
    if val is None:
        return ""
    s = str(val).strip()
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", s)
    if not m:
        return ""
    h = int(m.group(1)); mnt = int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mnt <= 59):
        return ""
    return f"{h:02d}:{mnt:02d}"
def is_new_appt_intent(text: str) -> bool:
    """
    'yeni randevu', 'randevu al(abilir miyim)', 'rezervasyon oluştur',
    'randevu ayarla', 'randevu lazım' vb. geniş niyet detektörü.
    Tek kelimelik 'randevu?' da True döner.
    """
    t = normalize_tr(text or "")
    # Noun/keyword varyasyonları (typo’lar dahil çok hafif)
    nouns = [
        "randevu", "randavu", "randevo", "rezervasyon", "rezerv", "rezervasyon"
    ]
    # Fiil/istek kalıpları
    verbs = [
        "al", "almak", "alalım", "alabilir", "alabilir miy", "almak ist",
        "oluştur", "olustur", "ayarla", "ayarlayabilir", "ayarlamak ist",
        "talep", "rica", "istiyor", "istiyorum", "isterim",
        "lazım", "gerek", "gerekli", "uygun saat", "müsait saat"
    ]

    has_noun = any(n in t for n in nouns)
    has_verb = any(v in t for v in verbs)

    # Başlangıç/tek kelime/kalıp yakalama
    starts_like = bool(re.match(r"^(yeni|başka)?\s*(bir\s+)?(randevu|rezervasyon)\b", t))
    single_word = t.strip() in {"randevu", "rezervasyon", "randevu?", "rezervasyon?"}
    polite_q = bool(re.search(r"(randevu|rezervasyon).*(al|ayarla|oluştur|olustur).*(mi|mı|miyim|miyim\?)", t))

    return has_noun and (has_verb or starts_like or single_word or polite_q)

def is_explicit_reset(text: str) -> bool:
    """Kullanıcı gerçekten sıfırdan başlamak istiyorsa (emredici)"""
    t = normalize_tr(text or "")
    hard = [
        "baştan başla", "bastan basla", "sıfırla", "sifirla", "yeni baştan",
        "yeniden başla", "yeni randevu başlat", "başka randevu", "başa dön",
        "yeni randevu"
    ]
    return any(h in t for h in hard)


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
    e164 = "+90 " + cand                      # "+905301234567"
    return disp, e164

def phone_to_e164_any(text: str) -> str | None:
    """Her türlü girdiden E.164 üret; olmazsa None."""
    out = parse_phone_any(text)
    return out[1] if out else None

def _ensure_resource_id(c) -> str | None:
    if getattr(c, "resourceId", None):
        return c.resourceId
    rid = None
    if 'BRANCH_INDEX' in globals() and BRANCH_INDEX:
        rid = resolve_rid_from_display_any(BRANCH_INDEX, c.branch)
    if rid is None and 'BRANCHES' in globals() and BRANCHES:
        rid = resolve_rid_from_display_any(BRANCHES, c.branch)
    if rid:
        c.resourceId = rid
    return rid

# Türkçe ad-soyad çıkarımı (cümle içinden)
import re

def strip_markup(text: str) -> str:
    # markdown bold/italic/inline-code vb. işaretleri temizle
    return re.sub(r'[*_`~]+', '', text or '')

NAME_HINT_RE = re.compile(
    r"\b(ad[ıi]m|ismim|ben)\s*[:\-]?\s+([A-Za-zÇĞİÖŞÜçğıöşü’'\-]+(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü’'\-]+){1,3})\b",
    re.I
)
PURE_NAME_RE = re.compile(
    r"^[A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,}(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü’'\-]{2,})+$"
)

HONORIFIC_RE = re.compile(r'^(say(?:ı|i)n|sn\.?|dr\.?|mr\.?|mrs\.?|prof\.?|doç\.?|doc\.?)\s+', re.I)

def clean_name(name: str | None) -> str:
    if not name: return ""
    n = strip_markup(name)
    n = HONORIFIC_RE.sub("", n).strip()
    n = re.sub(r'["“”\'’]+', '', n)       # tırnakları at
    n = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü’'\-\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def extract_fullname(text: str) -> str | None:
    t = strip_markup((text or "").strip())
    m = NAME_HINT_RE.search(t)
    if m:
        return " ".join(m.group(2).split())
    if PURE_NAME_RE.fullmatch(t):
        return " ".join(t.split())
    return None

def is_valid_fullname(name: str | None) -> bool:
    if not name: return False
    parts = [p for p in clean_name(name).split() if p]
    return len(parts) >= 2 and all(len(p) >= 2 for p in parts)

def set_fullname(state: Ctx, name: str, source: str = "") -> bool:
    clean = clean_name(name)  # senin clean_name fonksiyonun (tırnak/unvan temizler)
    if not is_valid_fullname(clean):
        return False
    prev = state.goal.customer.get("fullName")
    state.goal.customer["fullName"] = clean
    log(f"[NAME_SET] src={source} prev={prev!r} -> new={clean!r}")
    return True

# --- Saat yakalama (metin içinde) ---
TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[:\.]\s*([0-5]\d)(?!\d)")

# --- TR hafta içi adı → bir sonraki tarih ---
WEEKDAY_MAP = {
    "pazartesi": 0, "salı": 1, "çarşamba": 2, "carsamba": 2,
    "perşembe": 3, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6
}

# Ay adları (TR)


MONTHS_TR_MAP = {
    # normalize (ascii) anahtarlar
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5,
    "haziran": 6, "temmuz": 7, "agustos": 8, "eylul": 9,
    "ekim": 10, "kasim": 11, "aralik": 12,
    # diakritikli yazımlar (gerekli değil ama güvence)
    "şubat": 2, "mayıs": 5, "ağustos": 8, "eylül": 9,
    "kasım": 11, "aralık": 12,
}

def month_num_tr(mon: str) -> int | None:
    return MONTHS_TR_MAP.get(normalize_tr(mon))

def guess_next_weekday_iso(weekday_idx: int) -> str:
    today = datetime.now(IST).date()
    today_idx = find_weekday_idx(WEEKDAYS_TR[today.weekday()])  
    log("today:",today)
    diff = (weekday_idx - today_idx) % 7 or 7
    return (today + timedelta(days=diff)).isoformat()

# --- Göreli TR tarihler ---
def _parse_relative_date(text: str) -> str | None:
    t = text.lower()
    today = datetime.now(IST).date()
    if "bugün" in t or "bugun" in t: return today.isoformat()
    if "yarın" in t or "yarin" in t: return (today + timedelta(days=1)).isoformat()
    if "öbür gün" in t or "obur gun" in t or "yarından sonra" in t or "yarindan sonra" in t:
        return (today + timedelta(days=2)).isoformat()
    m = re.search(r"(haftaya|gelecek hafta)\s+(pazartesi|sal[ıi]|çarşamba|carsamba|perşembe|persembe|cuma|cumartesi|pazar)", t)
    if m:
        idx = WEEKDAY_MAP.get(m.group(2), None)
        if idx is not None:
            base = today + timedelta(days=7)
            diff = (idx - base.weekday()) % 7
            return (base + timedelta(days=diff)).isoformat()
    if "haftaya" in t or "gelecek hafta" in t:
        return (today + timedelta(days=7)).isoformat()
    return None


def _parse_any_calendar_date(text: str) -> str | None:
    t = text.lower()
    # 21 Ağustos [2025]
    m = re.search(r"(\d{1,2})\s+([a-zçğıöşü]+)(?:\s+(\d{4}))?", t, re.I)
    if m:
        d = int(m.group(1)); mon = m.group(2); y = int(m.group(3) or datetime.now(IST).year)
        mo = month_num_tr(mon)   # <<< BURASI
        if mo:
            try:
                return datetime(y, mo, d, tzinfo=IST).date().isoformat()
            except ValueError:
                pass
    # 21.08[.2025] / 21/08[/2025]
    m = re.search(r"(\d{1,2})[\.\/](\d{1,2})(?:[\.\/](\d{4}))?", t)
    if m:
        d = int(m.group(1)); mo = int(m.group(2)); y = int(m.group(3) or datetime.now(IST).year)
        try:
            return datetime(y, mo, d, tzinfo=IST).date().isoformat()
        except ValueError:
            pass
    return None

# --- PLACE: mvp_agentic_appointments.py (eski parse_identity_semicolon'ı bununla değiştir) ---

# --- replace: parse_identity_semicolon ---
import re, unicodedata

_HONORIFICS = {"sn","sayın","sayin","dr","dr.","doktor","prof","prof.","doç","doc","doç.","öğr","ogr","öğr.","mr","mrs","ms"}

def _clean_honorifics(name: str) -> str:
    if not name: return ""
    t = re.sub(r"[^\w\sÇĞİÖŞÜçğıöşü'-]+", " ", name, flags=re.UNICODE)
    toks = [w for w in t.strip().split() if w.casefold().strip(".") not in _HONORIFICS]
    return " ".join(toks)

_MOBILE = re.compile(r"(?:\+?90)?\s*0?\s*(?P<a>5\d{2})\D*(?P<b>\d{3})\D*(?P<c>\d{2})\D*(?P<d>\d{2})")
_FIXED  = re.compile(r"(?:\+?90)?\s*0?\s*(?P<a>\d{3})\D*(?P<b>\d{3})\D*(?P<c>\d{2})\D*(?P<d>\d{2})")

def _fmt_phone(m: re.Match) -> tuple[str, str]:
    a,b,c,d = m.group("a","b","c","d")
    disp = f"0{a} {b} {c} {d}"
    e164 = f"+90{a}{b}{c}{d}"
    return disp, e164

def parse_identity_semicolon(msg: str):
    if not msg:
        return None
    s = unicodedata.normalize("NFC", msg).strip()
    if ";" not in s:
        return None

    left, right = s.split(";", 1)
    name_raw = _clean_honorifics(left.strip())

    # 1–5 parça (Türkçe harf destekli)
    tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü’'-]*", name_raw)
    if not (1 <= len(tokens) <= 5):
        return None
    name = " ".join(tokens)

    # <<< BURAYI DEĞİŞTİR >>>
    # Eski (başına 0 ekleyen yol):
    # m = _PHONE_ANY.search(right)
    # if not m: return None
    # disp, e164 = _format_phone(m)

    # Yeni (senin ortak normalizer’ın):
    ph = parse_phone_any(right)
    if not ph:
        return None
    disp, e164 = ph

    return (name, disp, e164)



# --- Detay mesajı: "Şube; Gün/Tarih; Saat" ---
def parse_details_semicolon(text: str, branches: dict[str,int] | None = None):
    t = (text or "").replace("\u00A0"," ")
    if t.count(";") < 2: return None, None, None
    b_raw, d_raw, h_raw, *_ = [p.strip() for p in t.split(";", 3)]
    # şube
    branch = None
    if branches:
        br_l = b_raw.lower()
        for k in branches.keys():
            if k in br_l:
                branch = k.title()
                break
    else:
        branch = b_raw.title() if b_raw else None
    # gün/tarih
    date_iso = _parse_relative_date(d_raw) or \
               next((guess_next_weekday_iso(idx) for k,idx in WEEKDAY_MAP.items() if k in d_raw.lower()), None) or \
               _parse_any_calendar_date(d_raw)
    # saat
    m = TIME_RE.search(h_raw)
    hhmm = f"{int(m.group(1)):02d}:{m.group(2)}" if m else None
    return branch, date_iso, hhmm



# =========================
# Tipler / State
# =========================
@dataclass
class Constraints:
    date: Optional[str] = None           # "2025-08-21"
    approxTime: Optional[str] = None     # "öğleden sonra" veya "14:00"
    branch: Optional[str] = None         # "Kozyatağı" (MVP’de sadece metinsel)
    resourceId: Optional[int] = None     # hekim/oda (MVP: tek hekim = RESOURCE_ID)
    durationMin: Optional[int] = None    # 15, 30 (MVP: SLOT_SIZE_MIN)

@dataclass
class Preferences:
    preferredHourStart: str = "10:00"
    preferredHourEnd: str = "16:00"
    preferredBranch: Optional[str] = None
    preferredResourceId: Optional[int] = None

@dataclass
class Goal:
    channel: str = "whatsapp"  # whatsapp/web/phone
    customer: Dict[str, Optional[str|int]] = field(default_factory=lambda: {"fullName": None, "phone": None, "id": None})
    constraints: Constraints = field(default_factory=Constraints)
    notes: Optional[str] = None

@dataclass
class Held:
    apptId: str
    slotISO: str
    expiresInSec: int

@dataclass
class Ctx:
    status: str = "idle"  # idle|collecting|offering|holding|waiting_confirm|confirmed|failed
    goal: Goal = field(default_factory=Goal)
    memory: Preferences = field(default_factory=Preferences)
    policy: Dict[str, int|bool] = field(default_factory=lambda: {
        "slotSizeMin": SLOT_SIZE_MIN,
        "leadTimeMin":  LEAD_TIME_MIN,
        "weekendAllowed": WEEKEND_ALLOWED,
        "maxReplans": MAX_REPLANS
    })
    slots: List[str] = field(default_factory=list)       # ISO slot listesi
    offeredSlots: List[str] = field(default_factory=list)
    held: Optional[Held] = None
    replans: int = 0

# =========================
# Basit tercih belleği (stub)
# =========================
_PREF_STORE: Dict[str, Preferences] = {}
def get_prefs(user_id: str) -> Preferences:
    return _PREF_STORE.get(user_id, Preferences())
def update_prefs(user_id: str, patch: Dict):
    cur = get_prefs(user_id)
    data = asdict(cur); data.update(patch)
    _PREF_STORE[user_id] = Preferences(**data)

# =========================
# Yardımcılar
# =========================
WEEKDAY_MAP = {
    "pazartesi": 0, "salı": 1, "çarşamba": 2, "carsamba": 2,
    "perşembe": 3, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6
}

WEEKDAYS_TR = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"]
MONTHS_TR   = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"]



def _parse_approx_time(text: str) -> str | None:
    t = text.lower()
    # kesin saat (09:30, 9.30)
    m = re.search(r'(\d{1,2})[:\.](\d{2})', t)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2))
        return f"{hh:02d}:{mm:02d}"
    # esnek ifadeler
    if "sabah" in t: return "10:00"
    if "öğlen" in t or "oglen" in t: return "12:30"
    if "öğleden sonra" in t or "ogleden sonra" in t: return "14:00"
    if "akşam" in t or "aksam" in t: return "18:00"
    return None

def choose_nearest_date_if_only_time(hhmm: str, lead_min: int, weekend_allowed: bool) -> str:
    now = datetime.now(IST)
    target = now.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:5]), second=0, microsecond=0)
    if target < now + timedelta(minutes=lead_min):
        target += timedelta(days=1)
    while target.weekday() >= 5 and not weekend_allowed:
        target += timedelta(days=1)
    return target.date().isoformat()

# def extract_details_freeform(text: str, branch_index: Dict[str, object]) -> Dict[str, str]:
#     """
#     Ör: 'yarın Kozyatağı 12:00' -> {'branch':'Kozyatağı','time':'12:00','date':'YYYY-MM-DD'}
#     - Şube: BRANCH_INDEX anahtarlarını (normalize) metinde arar, en uzun eşleşmeyi seçer,
#             değerden görünen adı alır (tuple/dict/düz değer destekli).
#     - Saat: doğrudan HH:MM; yoksa 'akşam/öğle/sabah...' ≈ HH:MM fallback
#     - Tarih: 'bugün/yarın' veya 'perşembe/cuma...' ≈ bir sonraki o gün; yoksa bugün
#     """
#     t = text or ""
#     out: Dict[str, str] = {}

#     # (1) Saat (HH:MM)
# #    m = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", t)
#     m = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", t)
#     time_hm = f"{m.group(1)}:{m.group(2)}" if m else None

#     # (2) Tarih
#     now = datetime.now()
#     low = t.casefold()
#     date_iso: Optional[str] = None
#     if ("yarın" in low) or ("yarin" in low):
#         date_iso = (now + timedelta(days=1)).date().isoformat()
#     elif ("bugün" in low) or ("bugun" in low):
#         date_iso = now.date().isoformat()
#     else:
#         n = _norm_tr(t)
#         for token, idx in _DOW_NORM.items():
#             if token in n:
#                 date_iso = _next_weekday_from(now.date(), idx)
#                 break
#         if not date_iso:
#             date_iso = now.date().isoformat()

#     # (3) Yaklaşık saat yoksa part-of-day fallback
#     if not time_hm:
#         n = _norm_tr(t)
#         for key, hhmm in _PART_OF_DAY.items():
#             if key in n:
#                 time_hm = hhmm
#                 break

#     # (4) Şube: normalize anahtarları en uzun eşleşmeye göre tara
#     if isinstance(branch_index, dict) and branch_index:
#         norm_text = _norm_tr(t)
#         # En uzun anahtar önce (örn. 'sisli torun center' > 'sisli')
#         for key in sorted(branch_index.keys(), key=len, reverse=True):
#             k_norm = key if key == _norm_tr(key) else _norm_tr(key)
#             if k_norm and k_norm in norm_text:
#                 v = branch_index[key]
#                 if isinstance(v, (tuple, list)) and v:
#                     disp = v[0]
#                 elif isinstance(v, dict):
#                     disp = v.get("display") or v.get("name") or key
#                 else:
#                     disp = key
#                 if isinstance(disp, str) and disp:
#                    out["branch"] = _title_words_tr(disp) if disp == disp.lower() else disp
#                 break

#     # (5) Sadece dolu alanları koy
#     if time_hm:
#         out["time"] = time_hm
#     if date_iso:
#         out["date"] = date_iso
#     return out

def extract_details_freeform(text: str, branch_index: Dict[str, object]) -> Dict[str, str]:
    """
    Serbest metinden {branch?, date?, time?} çıkarır.

    🔎 Kurallar
    - Saat: 'HH:MM' / 'HH.MM' → 'HH:MM' normalize. Yoksa 'sabah/öğle/akşam/gece' benzeri _PART_OF_DAY ile saat tahmini.
    - Tarih: sadece AÇIKÇA yazıldıysa set edilir:
        * 'bugün/yarın', hafta günü adı ('perşembe' vb. → bir SONRAKİ o gün)
        * sayısal tarih: 21.08 / 21-08 / 21/08 (/yyyy opsiyonel) / yyyy-mm-dd
        * metinli ay: '21 ağustos' (bu yıl)
      ➕ Eğer sadece saat verildiyse (tarih yok), bugüne ânkorlanır.
      ➖ Aksi halde (tarih/saat yoksa) date eklenmez.
    - Şube: branch_index içindeki anahtarlar _norm_tr ile normalize edilerek en uzun eşleşme seçilir.
    """
    import re
    from datetime import datetime, timedelta
    log("extract_details_freeform başladı")
    log("text:",text)
    log("branch_index:", branch_index)

    t = text or ""
    out: Dict[str, str] = {}

    # ---------------- (1) SAAT ----------------
    m = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", t)
    time_hm = f"{int(m.group(1)):02d}:{m.group(2)}" if m else None  # '10.30' -> '10:30'

    # Part-of-day fallback (saat hiç yoksa)
    if not time_hm:
        n = _norm_tr(t)
        for key, hhmm in _PART_OF_DAY.items():  # örn. {'sabah':'09:00','öğle':'12:30','akşam':'18:00',...}
            if key in n:
                time_hm = hhmm
                break

    # ---------------- (2) TARİH (yalnızca açık ifade varsa) ----------------
    now = datetime.now()
    low = t.casefold()
    date_iso: Optional[str] = None

    # a) 'bugün/yarın'
    if ("yarın" in low) or ("yarin" in low):
        date_iso = (now + timedelta(days=1)).date().isoformat()
    elif ("bugün" in low) or ("bugun" in low):
        date_iso = datetime.now().isoformat()

    # b) hafta günü (bir SONRAKİ o gün)
    if not date_iso:
        log("date_iso1", date_iso)
        log("t:",t)
        n = _norm_tr(t)
        log("n=",n)
        log("_DOW_NORM:",_DOW_NORM.items())
        for token, idx in sorted(_DOW_NORM.items(), key=lambda kv: -len(kv[0])):
             if token in n:   # "cumartesi" önce denenir, "cuma"ya sıra gelmez
                 date_iso = _next_weekday_from(datetime.now(IST).date(), idx) 
                 break
           
#        idx = find_weekday_idx(n)  
#        date_iso = _next_weekday_from(datetime.now(IST).date(), idx)           
        log("date_iso2",date_iso)
    # c) sayısal tarih (21.08[/2025], 21-08, 21/08, yyyy-mm-dd)
    if not date_iso:
        mnum = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)  # ISO
        if mnum:
            yyyy, mm, dd = mnum.groups()
            date_iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        else:
            mnum2 = re.search(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b", t)
            if mnum2:
                dd, mm, yy = mnum2.group(1), mnum2.group(2), mnum2.group(3)
                DD, MM = int(dd), int(mm)
                if yy:
                    YY = int(yy)
                    YY = YY + 2000 if YY < 100 else YY
                else:
                    YY = now.year
                try:
                    date_iso = f"{YY:04d}-{MM:02d}-{DD:02d}"
                except Exception:
                    date_iso = None  # geçersiz tarihse set etme

    # d) metinli ay (21 ağustos)
    if not date_iso:
        # ay isimleri normalize edilmiş olmalı (agustos vs.)
        MONTHS = {
            "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4,
            "mayıs": 5, "mayis": 5, "haziran": 6, "temmuz": 7,
            "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
            "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12
        }
        mtxt = re.search(
            r"\b(\d{1,2})\s*(ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik)\b",
            t, re.IGNORECASE
        )
        if mtxt:
            dd = int(mtxt.group(1))
            mon = mtxt.group(2).casefold()
            mm = MONTHS.get(mon)
            if mm:
                try:
                    date_iso = f"{now.year:04d}-{mm:02d}-{dd:02d}"
                except Exception:
                    date_iso = None

    # e) hiçbir açık tarih bulunmadıysa ve SAAT varsa → bugüne ânkorla
    if (not date_iso) and time_hm:
        date_iso = datetime.now().isoformat()

    # ⚠️ eğer ne tarih ne saat varsa → date set etme (sürpriz 'bugün' yok)
    # ---------------- (3) ŞUBE ----------------
   # --- Şube (display adı) ---
    disp = _resolve_branch_display(t, branch_index)
    if disp:
        out["branch"] = _title_words_tr(disp)

    # ---------------- (4) Yalnız dolu alanları döndür ----------------
    if time_hm:
        out["time"] = time_hm
    if date_iso:
        out["date"] = date_iso
    return out


def normalize_tr(s: str) -> str:
    if not s: return ""
    s = s.replace("İ","I").replace("ı","i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def build_branch_index(branches: dict[str,int]) -> dict[str, tuple[str,int]]:
    """
    Arama için normalize edilmiş anahtarlar üretir.
    'şişli torun center' için ekstra anahtarlar: 'sisli', 'torun center'
    """
    idx: dict[str, tuple[str,int]] = {}
    for disp, rid in branches.items():
        norm = normalize_tr(disp)
        idx[norm] = (disp, rid)

        # Çok kelimeli şube için pratik kısayollar
        tokens = norm.split()
        if len(tokens) >= 1:
            idx[tokens[0]] = (disp, rid)  # 'sisli' → 'şişli torun center'
        if len(tokens) >= 2:
            tail = " ".join(tokens[1:])
            idx[tail] = (disp, rid)       # 'torun center' → 'şişli torun center'
    return idx

BRANCH_INDEX = build_branch_index(BRANCHES)

# def detect_branch_freeform(text: str, branch_index: dict[str, tuple[str,int]]):
#     """
#     Metin içinde normalize anahtarları arar; en uzun eşleşmeyi döner.
#     DÖNÜŞ: (display_name, resourceId) veya (None, None)
#     """
#     t = normalize_tr(text)
#     for key in sorted(branch_index.keys(), key=len, reverse=True):
#         if key and key in t:
# #            disp, rid = branch_index[key]
#              disp, rid = _resolve_rid_by_display(branch_index, branch)   # yeni
# #             if rid is None:
# #                 return ("Şube tanımı bulunamadı. Lütfen şubeyi tekrar belirtir misiniz?", ctx)
#             # >>> Title-Case ile döndür (TR diakritikler korunur)
#         return (disp.title(), rid)
#     return (None, None)




def ensure_e164(phone: Optional[str]) -> Optional[str]:
    if not phone: return None
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("90"): return f"+{digits}"
    if digits.startswith("0"):  return f"+9{digits}"
    if digits.startswith("5"):  return f"+90{digits}"
    if phone.startswith("+"):   return phone
    return f"+{digits}"

# def to_iso_on_date(date_yyyy_mm_dd: str, hhmm: str) -> str:
#     h, m = [int(x) for x in hhmm.split(":")]
#     dt = datetime.fromisoformat(f"{date_yyyy_mm_dd}T{h:02d}:{m:02d}:00").replace(tzinfo=IST)
#     return dt.isoformat()
def to_iso_on_date(date_yyyy_mm_dd: str, hhmm: str) -> str:
    s = str(date_yyyy_mm_dd).strip()
    # date_str içinde saat/mikrosaniye varsa sadece YYYY-MM-DD parçasını al
    if "T" in s:
        s = s.split("T", 1)[0]
    # beklenen format: YYYY-MM-DD ve HH:MM
    h, m = [int(x) for x in hhmm.split(":")]
    dt = datetime.strptime(f"{s} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
    return dt.isoformat()

def to_iso_end(start_iso: str, dur_min: int) -> str:
    dt = datetime.fromisoformat(start_iso)
    return (dt + timedelta(minutes=dur_min)).isoformat()

def human(iso: str) -> str:
    dt = datetime.fromisoformat(iso).astimezone(IST)
    gun = WEEKDAYS_TR[dt.weekday()]
    ay  = MONTHS_TR[dt.month-1]
    return f"{dt.day:02d} {ay} {gun} {dt:%H:%M}"

def fmt(list_: List[str]) -> str:
    return ", ".join(human(s) for s in list_)

def shift_date(date_yyyy_mm_dd: str, days: int) -> str:
    d = datetime.fromisoformat(date_yyyy_mm_dd).date()
    return (d + timedelta(days=days)).isoformat()

def pick_top_slots(slots_iso: List[str], pref: Preferences, top_n=3) -> List[str]:
    ps_h, ps_m = [int(x) for x in pref.preferredHourStart.split(":")]
    pe_h, pe_m = [int(x) for x in pref.preferredHourEnd.split(":")]
    ps = ps_h * 60 + ps_m; pe = pe_h * 60 + pe_m
    def score(iso: str) -> int:
        dt = datetime.fromisoformat(iso).astimezone(IST)
        m = dt.hour * 60 + dt.minute
        if m < ps: return ps - m
        if m > pe: return m - pe
        return 0
    return sorted(slots_iso, key=score)[:top_n]

def normalize_slots(date_str: str, slots_from_adapter: List[str]) -> List[str]:
    iso_slots = []
    date_only = str(date_str).split("T", 1)[0]
    for s in slots_from_adapter:
        if "T" in s: iso_slots.append(s)
        else:        iso_slots.append(to_iso_on_date(date_only, s))
    return iso_slots

# =========================
# Extractor (MVP — regex)
# Üretimde LLM + tool-calling kullanman önerilir.
# =========================
def extract_constraints(user_text: str) -> Dict:
    text = user_text.lower()

    # şube basit çıkarımı (eski mantık korunuyor)
    branch = None
    if "kozyatağ" in text or "kozyatag" in text: branch = "Kozyatağı"
    if "mecidiyeköy" in text or "mecidiyekoy" in text: branch = branch or "Mecidiyeköy"

    # 1) göreli tarih (bugün/yarın/öbür/haftaya ...)
    date_str = _parse_relative_date(text)

    # 2) hafta içi adı (pazartesi/…): yalnızca hâlâ tarih bulunamadıysa
    if not date_str:
        for k, idx in WEEKDAY_MAP.items():
            if k in text:
                date_str = guess_next_weekday_iso(idx)
                break

    # 3) yaklaşık saat
    approx = _parse_approx_time(text)

    return {"branch": branch, "date": date_str, "approxTime": approx}

# =========================
# Generic CRM Adapter — InMemory (çalışır)
# =========================
class GenericCRMAdapterInMemory:
    """
    MVP için çalışan in-memory CRM.
    availability/hold/confirm/cancel davranışlarını simüle eder.
    Daha sonra HTTP adaptöre geçip gerçek CRM uçlarına bağlanabilirsin.
    """
    def __init__(self):
        # appts: {id, res_id, start:datetime, end:datetime, status, hold_expires_at:datetime|None}
        self.appts: List[Dict] = []

    def _busy_intervals(self, resource_id: int, day: date) -> List[Tuple[time, time]]:
        busy: List[Tuple[time, time]] = []
        for a in self.appts:
            if a["res_id"] != resource_id: continue
            if a["status"] in ("hold", "confirmed") and a["start"].date() == day:
                # süresi dolmuş hold'ları temizle
                if a["status"] == "hold" and a["hold_expires_at"] and datetime.now(IST) > a["hold_expires_at"]:
                    a["status"] = "cancelled"
                    continue
                busy.append((a["start"].time(), a["end"].time()))
        return busy

    def availability(self, resource_id: int, date_str: str, slot_size: int = SLOT_SIZE_MIN) -> List[str]:
        if date_str in HOLIDAYS: return []
        d = datetime.fromisoformat(date_str).date()
        wd = d.weekday()
        if wd >= 5 and not WEEKEND_ALLOWED:
            return []
        blocks = WORKING_HOURS.get(wd, [])
        if not blocks: return []
        busy = self._busy_intervals(resource_id, d) + BREAKS.get(wd, [])
        slots: List[str] = []
        for start_t, end_t in blocks:
            cur = datetime.combine(d, start_t, IST)
            end = datetime.combine(d, end_t, IST)
            while cur + timedelta(minutes=slot_size) <= end:
                s_t = cur.time(); e_t = (cur + timedelta(minutes=slot_size)).time()
                if not any(not (e_t <= b_s or s_t >= b_e) for (b_s, b_e) in busy):
                    slots.append(cur.strftime("%H:%M"))
                cur += timedelta(minutes=slot_size)
        return slots

    def hold(self, resource_id: int, start_iso: str, end_iso: str, customer: Dict[str, str],
              notes: Optional[str], channel: Optional[str]) -> Tuple[Optional[Dict], bool]:
        start = datetime.fromisoformat(start_iso); end = datetime.fromisoformat(end_iso)
        # çakışma?
        for a in self.appts:
            if a["res_id"] != resource_id: continue
            if a["status"] in ("hold", "confirmed"):
                # hold süresi bitti mi?
                if a["status"] == "hold" and a["hold_expires_at"] and datetime.now(IST) > a["hold_expires_at"]:
                    a["status"] = "cancelled"; continue
                if not (end <= a["start"] or start >= a["end"]):
                    return None, True  # conflict
        appt_id = uuid.uuid4().hex[:12]
        self.appts.append({
            "id": appt_id, "res_id": resource_id,
            "start": start, "end": end, "status": "hold",
            "hold_expires_at": datetime.now(IST) + timedelta(seconds=HOLD_TTL_SEC)
        })
        return {"appointment_id": appt_id, "status": "hold", "expires_in": HOLD_TTL_SEC}, False

    def confirm(self, appt_id: str) -> Tuple[Optional[Dict], bool]:
        for a in self.appts:
            if a["id"] == appt_id:
                if a["status"] != "hold":
                    return {"status": a["status"]}, False
                if datetime.now(IST) > a["hold_expires_at"]:
                    a["status"] = "cancelled"
                    return None, True  # expired
                a["status"] = "confirmed"
                a["hold_expires_at"] = None
                return {"status": "confirmed"}, False
        return None, True  # bulunamadı → expired gibi davran

    def cancel(self, appt_id: str):
        for a in self.appts:
            if a["id"] == appt_id and a["status"] in ("hold", "confirmed"):
                a["status"] = "cancelled"
                return

# ---- Basit fabrika: env'e göre memory vs http seçimi
# class GenericCRMAdapterInMemory:
#     """Placeholder: Sizin mevcut sınıfınız. Buraya sadece imza için minimal gövde eklendi."""
#     def __init__(self):
#         self.appts: List[Dict] = []
#     def availability(self, resource_id: int, date_str: str, slot_size: int = 30) -> List[str]:
#         return ["10:00", "10:30"]
#     def hold(self, resource_id: int, start_iso: str, end_iso: str, customer: Dict[str, str], notes: Optional[str], channel: Optional[str]) -> Tuple[Optional[Dict], bool]:
#         return {"appointment_id": "demo", "status": "hold", "expires_in": 120}, False
#     def confirm(self, appt_id: str) -> Tuple[Optional[Dict], bool]:
#         return {"status": "confirmed"}, False
#     def cancel(self, appt_id: str) -> None:
#         return


def make_crm_adapter() -> Any:
    print("CRM Adapter İçinde3")
    # adapter = GenericCRMAdapterHTTP(
    #     base_url="http://hospital.creqit.com:8000",  # sondaki / yok
    #     auth_mode="frappe_token",
    #     api_key="992390ede230b76",
    #     api_secret="640bc0c0f1edc81",
    #     timeout=10,
    # )
    
    # === config ===
    APPT_BASE_URL = os.getenv("APPT_BASE_URL", "http://odyoduyu.sailead.com.tr")  # 443
    APPT_AUTH_MODE = os.getenv("APPT_AUTH_MODE", "frappe_token")
    APPT_API_KEY   = os.getenv("APPT_API_KEY", "") # c2eaf5cedf87e65
    APPT_API_SECRET= os.getenv("APPT_API_SECRET", "") # ea279e4183f904e
    APPT_TIMEOUT   = float(os.getenv("APPT_TIMEOUT", "8"))
#    CHECK_AVAIL    = os.getenv("CHECK_AVAIL", "1") == "1"  # 0 yaparsan hiç çağırmaz
    print("APPT_BASE_URL:", APPT_BASE_URL)
    adapter = GenericCRMAdapterHTTP(
        base_url=APPT_BASE_URL.rstrip("/"),
        auth_mode=APPT_AUTH_MODE,
        api_key=APPT_API_KEY,
        api_secret=APPT_API_SECRET,
        timeout=APPT_TIMEOUT,
    )
   
    # Frappe yollarını doğrudan objede override et
    adapter.path_prefix  = "api/resource"                 # artık v1 değil
    adapter.path_appts   = "api/resource/Appointment"     # liste/create için doğru yol
    # (Lead entegrasyonu kullanacaksanız)
    adapter.path_leads   = "api/resource/Lead"
    adapter.default_lead_fields = [
        "name","full_name","email","phone",
        "lead_validity","lead_statu",
        "source_group","main_source","sub_source",
        "lead_channel","language","country",
        "provinces","district",
        "main_services","sub_services","treatments",
    ]
    # (Kaynağa göre çakışma filtresi kullanacaksanız; yoksa None bırakın)
    adapter.resource_field = None  # örn. "service_provider" / "examination_room" / "hospital"

    # Appointment default fields sizde zaten tanımlı; gerekirse burada da güncelleyebilirsiniz:
    # adapter.default_fields = ["name", "start_appointment_time", "end_appointment_time", ...]
    return adapter

"""
GenericCRMAdapterHTTP — gerçek CRM API'sine bağlanmak için adaptör iskeleti
-------------------------------------------------------------------------
Bu dosya, in‑memory adaptörünüzle aynı imzaları sağlayan (availability, hold, confirm, cancel)
HTTP tabanlı bir adaptör ve küçük bir arayüz/altyapı sunar. 

Kullanım özeti
-------------
from crm_adapter import GenericCRMAdapterHTTP
crm = GenericCRMAdapterHTTP(
    base_url=os.environ["CRM_BASE_URL"],
    api_key=os.getenv("CRM_API_KEY"),           # veya bearer_token
    bearer_token=os.getenv("CRM_BEARER_TOKEN"),
    tenant_id=os.getenv("CRM_TENANT_ID"),       # varsa
    timeout=10,
)
slots = crm.availability(resource_id=123, date_str="2025-10-02", slot_size=30)
hold_info, conflict = crm.hold(
    resource_id=123,
    start_iso="2025-10-02T10:00:00+03:00",
    end_iso="2025-10-02T10:30:00+03:00",
    customer={"fullName": "Burcu Uman", "phone": "+90...", "email": "..."},
    notes="WhatsApp kanalı",
    channel="whatsapp",
)

Notlar
------
- HTTP istekleri için requests + urllib3 Retry kullanılmaktadır (429/5xx için otomatik retry)
- İdempotency-Key başlığı her hold çağrısında otomatik üretilir (env'den önek verebilirsiniz)
- 409 (conflict) → (None, True) döner, 410/404 (expired/not found) confirm için (None, True) döner
- Zamanlar ISO8601 (tz-aware) beklenir/gönderilir; boş/"HH:MM" slot yanıtları da tolere edilir
- Hata tipleri tipik durumlar için özel exception sınıflarıyla ayrıştırılır
"""


import os
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import json as _json


# ---- Hatalar

class CRMError(Exception):
    pass

class CRMAuthError(CRMError):
    pass

class CRMConflictError(CRMError):
    """409: Slot çakışması vb."""
    pass

class CRMNotFound(CRMError):
    pass

class CRMExpiredHoldError(CRMError):
    """410: Hold süresi geçmiş"""
    pass


# ---- Yardımcılar
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=list(_ALLOWED_METHODS),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _norm_slots(payload: Any) -> List[str]:
    """CRM'in farklı slot formatlarını tek tipe çevir: ["HH:MM", ...]."""
    if payload is None:
        return []
    if isinstance(payload, dict) and "slots" in payload:
        slots_raw = payload["slots"]
    else:
        slots_raw = payload
    out: List[str] = []
    for x in (slots_raw or []):
        if isinstance(x, str):
            if "T" in x:
                # ISO datetime string -> HH:MM
                try:
                    dt = datetime.fromisoformat(x)
                    out.append(dt.strftime("%H:%M"))
                except Exception:
                    # varsayılan olarak olduğu gibi bırak
                    out.append(x)
            else:
                # "HH:MM" gibi bırak
                out.append(x)
        elif isinstance(x, dict) and "start" in x:
            # {start: iso, end: iso}
            try:
                dt = datetime.fromisoformat(x["start"])  # type: ignore
                out.append(dt.strftime("%H:%M"))
            except Exception:
                pass
    return sorted(set(out))



def _parse_any(s: Optional[str]) -> Optional[datetime]:
    log("_parse_any içinde", "[s]",s)
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    try:
        # "YYYY-MM-DD HH:MM:SS"
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _to_hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def _overlap(a_s, a_e, b_s, b_e):
    # not (a_e <= b_s or a_s >= b_e)
    return not (a_e <= b_s or a_s >= b_e)

# booking için hangi metodlar gerekli?
_REQUIRED_BOOKING = (
    "free_slots",        # gün için boş slotları üret
    "is_range_free",     # verilen aralık boş mu?
    "find_lead",         # lead bulma
    "book_if_free",      # aralık boşsa insert
    "create_appointment" # doğrudan insert (fallback)
    "create_lead"
)

def ensure_booking_adapter(crm, crm_booking=None):
    """
    crm/fixture FakeCRM olsa bile, booking için gereken yüzey yoksa
    gerçek HTTP adaptörüne düş. Varsa aynen geri döndür.
    """
    cand = crm_booking or crm
    if all(hasattr(cand, m) for m in _REQUIRED_BOOKING):
        return cand
    if crm_booking and all(hasattr(crm_booking, m) for m in _REQUIRED_BOOKING):
        return crm_booking
    # Son çare: gerçek adaptör
    return make_crm_adapter()

# =========================
# (Şablon) Generic CRM Adapter — HTTP (gerçeğe geçiş)
# =========================
class GenericCRMAdapterHTTP:
    """
    Frappe/ERPNext stilindeki Appointment endpoint'ine bağlanan adaptör.

    auth_mode:
      - frappe_token (önerilen): Authorization: token <api_key>:<api_secret>
      - bearer/basic/api_key/header_pair/hmac (gerekirse eklenebilir)
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 10,
        auth_mode: str = "frappe_token",
        # Frappe ayarları:
        path_appts: str = "api/resource/Appointment",
        resource_field: Optional[str] = None,  # "service_provider" / "examination_room" / "hospital" vb.
        default_fields: Optional[List[str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url zorunludur")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = _build_session()

        # Kimlik doğrulama
        self.api_key = api_key
        self.api_secret = api_secret
        self.auth_mode = (auth_mode or "frappe_token").lower()

        # Headerlar
        self.default_headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Frappe yolları/alanlar
        self.path_appts = path_appts  # "api/resource/Appointment"
        self.resource_field = resource_field  # örn: "service_provider"
        self.default_fields = default_fields or [
            "name",
            "data_pool_reference",
            "hospital",
            "examination_room",
            "service_provider",
            "start_appointment_time",
            "end_appointment_time",
            "appointment_type",
            "appointment_notes",
        ]

        # Idempotency (şimdilik sadece isim yer tutucu; Frappe tarafında pek gerekmez)
        self.idempotency_prefix = "appt-"
        
        # __init__ içinde, Appointment ayarlarının altına ekleyin:
        self.path_leads = "api/resource/Lead"
        self.default_lead_fields = [
            "name", "full_name", "email", "phone",
            "lead_validity", "lead_statu",
            "source_group", "main_source", "sub_source",
            "lead_channel", "language", "country",
            "provinces", "district",
            "main_services", "sub_services", "treatments",
        ]
        # appts: {id, res_id, start:datetime, end:datetime, status, hold_expires_at:datetime|None}
        self.appts: List[Dict] = []

#    import json as _json  # dosyada varsa tekrar import etmeyin

    def get_lead_by_name(self, name: str, fields: Optional[List[str]] = None) -> Dict:
        """
        GET /api/resource/Lead/{name} → tek lead döndürür.
        """
        log("ge_lead_by_name içinde", name)
        payload = self._request("GET", f"{self.path_leads}/{name}", expected=(200,))
        data = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(data, dict):
            raise CRMNotFound(f"Lead {name} bulunamadı veya beklenmeyen yanıt.")
        # İstenirse alan daraltma (opsiyonel)
        if fields:
            return {k: data.get(k) for k in fields}
        return data
    
    
    def find_lead(
        self,
        *,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        filters: Optional[List | Dict[str, Any]] = None,  # Frappe filters: list-of-lists veya dict
        fields: Optional[List[str]] = None,
        limit: int = 1,
        order_by: Optional[str] = None,
    ) -> List[Dict]:
        """
        GET /api/resource/Lead?fields=...&filters=... → lead listesi.
        Basit aramalar için phone/email/full_name paramlarını kullanın.
        """
        log("self path leads", self.path_leads)
        log("phone",phone)
        doctype = self.path_leads.split("/")[-1] or "Lead"
        log("find_lead içinde", doctype)
        flt: List = []
        if phone:
            flt.append([doctype, "phone", "=", phone])
        if email:
            flt.append([doctype, "email", "=", email])
        if full_name:
            flt.append([doctype, "full_name", "=", full_name])
    
        if filters:
            if isinstance(filters, dict):
                flt.extend([[doctype, k, "=", v] for k, v in filters.items()])
            elif isinstance(filters, list):
                flt.extend(filters)
            else:
                raise ValueError("filters list veya dict olmalı")
        log("doctype", doctype) 
        params = {
            "fields": _json.dumps(fields or self.default_lead_fields),
            "filters": _json.dumps(flt),
            "limit_page_length": max(1, int(limit)),
        }
       

        log("params", params)
        if order_by:
            params["order_by"] = order_by
    
        payload = self._request("GET", self.path_leads, params=params, expected=(200,))
        log("[payload]", payload)
        if isinstance(payload, dict):
            log("payload 1")
            items = payload.get("data") or payload.get("message") or []
        else:
            log("payload 2")
            items = payload
        log("items", items)    
        return items or []

    # --- kimlik başlıkları
    def _auth_headers(self) -> Dict[str, str]:
        if self.auth_mode == "frappe_token":
            if not (self.api_key and self.api_secret):
                raise CRMAuthError("API key/secret gerekli (frappe_token)")
            return {"Authorization": f"token {self.api_key}:{self.api_secret}"}
        elif self.auth_mode == "bearer":
            if not self.api_key:
                raise CRMAuthError("Bearer token gerekli (api_key alanında bekleniyor)")
            return {"Authorization": f"Bearer {self.api_key}"}
        # Diğer modlar gerekirse eklenir
        return {}

    # --- düşük seviye istek
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expected: Tuple[int, ...] = (200, 201, 204),
    ) -> Any:
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"Desteklenmeyen method: {method}")
        url = f"{self.base_url}/{path.lstrip('/')}"
        hdrs = dict(self.default_headers)
        hdrs.update(self._auth_headers())
        if headers:
            hdrs.update(headers)
        resp = self.session.request(method, url, params=params, json=json, headers=hdrs, timeout=self.timeout)

        if resp.status_code in expected:
            return None if resp.status_code == 204 else (resp.json() if resp.content else None)
        if resp.status_code in (401, 403):
            raise CRMAuthError(resp.text)
        if resp.status_code == 404:
            raise CRMNotFound(resp.text)
        if resp.status_code == 409:
            raise CRMConflictError(resp.text)
        if resp.status_code == 410:
            raise CRMExpiredHoldError(resp.text)
        raise CRMError(f"{resp.status_code}: {resp.text}")
        
    # GenericCRMAdapterHTTP sınıfının içine EKLEYİN (ör. check_between(...)'ün altına)

    def create_lead(
        self,
        *,
        lead_validity: Optional[str] = None,
        lead_statu: Optional[str] = None,
        source_group: Optional[str] = None,
        main_source: Optional[str] = None,
        sub_source : Optional[str] = None,
        language : Optional[str] = None,
        country : Optional[str] = None,
        main_services : Optional[str] = None,
        sub_services : Optional[str] = None,
        full_name : str,
        phone : str,
        squad : Optional[str] = None,
        advertisement_message : Optional[str] = None,
    ) -> Dict:
        """
        Frappe: POST /api/resource/Lead ile yeni lead kaydı oluşturur.
        Dönen payload'tan 'data' veya 'message' içindeki doc'u döndürür.
        """
        doc: Dict[str, Any] = {
            "phone": phone,
            "full_name": full_name,
        }
        
        if lead_validity is not None:
            doc["lead_validity"] = lead_validity
        if lead_statu is not None:
            doc["lead_statu"] = lead_statu
        if source_group is not None:
            doc["source_group"] = source_group
        if main_source is not None:
            doc["main_source"] = main_source
        if sub_source is not None:
            doc["sub_source"] = sub_source  
        if language is not None:
            doc["language"] = language   
        if country is not None:
            doc["country"] = country
        if main_services is not None:
            doc["main_services"] = main_services
        if sub_services is not None:
            doc["sub_services"] = sub_services
        if squad is not None:
            doc["squad"] = squad
        if advertisement_message is not None:
            doc["advertisement_message"] = advertisement_message  
    
        payload = self._request("POST", self.path_leads, json=doc, expected=(200, 201))
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], dict):
                return payload["data"]
            if "message" in payload and isinstance(payload["message"], dict):
                return payload["message"]
        return payload  # fallback    
    def create_appointment(
        self,
        *,
        start_dt: str,
        end_dt: str,
        data_pool_reference: Optional[str] = None,
        hospital: Optional[str] = None,
        examination_room: Optional[str] = None,
        service_provider: Optional[str] = None,
        appointment_type: Optional[str] = None,
        appointment_notes: Optional[str] = None,
    ) -> Dict:
        """
        Frappe: POST /api/resource/Appointment ile yeni randevu kaydı oluşturur.
        Dönen payload'tan 'data' veya 'message' içindeki doc'u döndürür.
        """
        doc: Dict[str, Any] = {
            "start_appointment_time": start_dt,
            "end_appointment_time": end_dt,
        }
        if data_pool_reference is not None:
            doc["data_pool_reference"] = data_pool_reference
        if hospital is not None:
            doc["hospital"] = hospital
        if examination_room is not None:
            doc["examination_room"] = examination_room
        if service_provider is not None:
            doc["service_provider"] = service_provider
        if appointment_type is not None:
            doc["appointment_type"] = appointment_type
        if appointment_notes is not None:
            doc["appointment_notes"] = appointment_notes
    
        payload = self._request("POST", self.path_appts, json=doc, expected=(200, 201))
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], dict):
                return payload["data"]
            if "message" in payload and isinstance(payload["message"], dict):
                return payload["message"]
        return payload  # fallback
    
    
    def book_if_free(
        self,
        *,
        start_dt: str,
        end_dt: str,
        resource_id: Optional[str | int] = None,
        confirm: bool = False,
        data_pool_reference: Optional[str] = None,
        hospital: Optional[str] = None,
        examination_room: Optional[str] = None,
        service_provider: Optional[str] = None,
        appointment_type: Optional[str] = None,
        appointment_notes: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        1) Verilen aralıkta çakışma var mı kontrol eder.
        2) Çakışma yoksa ve confirm=True ise yeni Appointment oluşturur.
        Dönüş: {created: bool, reason?: str, conflicts?: [...], doc?: {...}}
        """
        log("book_if_free/data_pool_referance", data_pool_reference)
        exists, conflicts = self.check_between(
            start_dt, end_dt, resource_id,
#            data_pool_reference,
            extra_filters=extra_filters,
            fetch_fields=[
                "name","start_appointment_time","end_appointment_time",
                "service_provider","examination_room","hospital"
            ],
        )
        if exists:            
            return {"created": False, "reason": "conflict", "conflicts": conflicts}
    
        if not confirm:
            return {"created": False, "reason": "free", "hint": "confirm=True ile oluşturabilirsiniz"}
    
        doc = self.create_appointment(
            start_dt=start_dt,
            end_dt=end_dt,
            data_pool_reference=data_pool_reference,
            hospital=hospital,
            examination_room=examination_room,
            service_provider=service_provider,
            appointment_type=appointment_type,
            appointment_notes=appointment_notes,
        )
        return {"created": True, "doc": doc}


    # --- İKİ TARİH ARASI ÇAKIŞMA KONTROLÜ (Önerilen metod)
    def check_between(
        self,
        start_dt: str,
        end_dt: str,
        resource_id: Optional[str | int] = None,
        *,
#        data_pool_reference: str,
        extra_filters: Optional[Dict[str, Any]] = None,
        fetch_fields: Optional[List[str]] = None,
    ) -> Tuple[bool, List[Dict]]:
        """
        Verilen aralıkta randevu var mı?
        Dönüş: (exists, matches)
        """
#        import json as _json
        log("check_between içinde")
        log("[start_dt]",start_dt,"[end_dt]",end_dt)
        # Girdi doğrulama
        s = _parse_any(start_dt); e = _parse_any(end_dt)
        log("[s]",s,"[e]",e)
        if not s or not e:
            raise ValueError("start_dt/end_dt 'YYYY-MM-DD HH:MM:SS' veya ISO8601 olmalı")

        doctype = (self.path_appts.split("/")[-1] or "Appointment")
        filters = [
            [doctype, "start_appointment_time", "<", end_dt],
            [doctype, "end_appointment_time",   ">", start_dt],
        ]
        log("start_dt",start_dt,"end_dt",end_dt)
        
        # Kaynağa göre filtre (doktor/oda/şube vs.)
        if resource_id is not None and self.resource_field:
            filters.append([doctype, self.resource_field, "=", str(resource_id)])
        # if data_pool_reference is not None:
        #     filters.append([doctype, "name", "=", str(data_pool_reference)])
            
        if extra_filters:
            for k, v in extra_filters.items():
                filters.append([doctype, k, "=", v])
        log("extra_filters", extra_filters)        
        fields = fetch_fields or self.default_fields
        params = {
            "fields": _json.dumps(fields),
            "filters": _json.dumps(filters),
            "limit_page_length": 500,
            "order_by": "start_appointment_time asc",
        }

        payload = self._request("GET", self.path_appts, params=params)
        log("[payload]",payload)
        items: List[Dict] = []
        if isinstance(payload, dict) and "data" in payload:
            items = payload["data"]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and "message" in payload:
            # bazı Frappe sürümleri data yerine message döndürebilir
            items = payload["message"] or []

        # Yerel çakışma teyidi
        matches: List[Dict] = []
        log("[items]",items)
        for it in items:
            a_s = _parse_any(it.get("start_appointment_time"))
            a_e = _parse_any(it.get("end_appointment_time"))
            if a_s and a_e and not (e <= a_s or s >= a_e):
                matches.append(it)
        log("[matches]",matches)
        return (len(matches) > 0, matches)
    
#     def book_if_free_with_lead(
#         self,
#         *,
#         start_dt: str,
#         end_dt: str,
#         confirm: bool = True,
#         # Lead için bakılacak anahtarlar:
#         lead_name: Optional[str] = None,
#         phone: Optional[str] = None,
#         email: Optional[str] = None,
#         full_name: Optional[str] = None,
#         lead_filters: Optional[List | Dict[str, Any]] = None,
#         # Appointment alanları:
#         hospital: Optional[str] = None,
#         examination_room: Optional[str] = None,
#         service_provider: Optional[str] = None,
#         appointment_type: Optional[str] = None,
#         appointment_notes: Optional[str] = None,
#         # book_if_free passthrough:
#         resource_id: Optional[str | int] = None,
#         extra_filters: Optional[Dict[str, Any]] = None,
#      ) -> Dict[str, Any]:
#     # """
#     # 1) Lead'i (name/phone/email) ile bulur.
#     # 2) Çakışma yoksa randevu yaratır (confirm=True).
#     # 3) data_pool_reference = lead.name olarak set edilir.
#     # """
#     # 1) Lead'i getir
#         log("book_if_free_with_lead içinde")
#         lead_doc = None
#         if lead_name:
#             lead_doc = self.get_lead_by_name(lead_name, fields=["name","full_name","phone","email"])
#         else:
#             results = self.find_lead(
#                 phone=phone, email=email, full_name=full_name,
#                 filters=lead_filters, fields=["name","full_name","phone","email"],
#                 limit=1
#             )
#             if results:
#                 lead_doc = results[0]
#                 log("lead_doc", lead_doc)
#                 log("lead_doc.name", lead_doc.get("name"))
                
                
#         if not lead_doc:
# # lead bulunmadıysa lead yaratacak    
#             if hasattr(self, "create_lead"):
#                         # çakışma kontrolü yoksa doğrudan yarat
#                doc = self.create_lead(
#                lead_validity = "Geçerli",
#                lead_statu = "ATAMA BEKLİYOR",
#                source_group = "Dijital Pazarlama",
#                main_source = "WhatsApp İletişim",
#                sub_source = "WhatsApp",
#                language = "Türkçe",
#                country = "Türkiye",
#                full_name = lead_doc["full_name"],
#                phone = phone,
#                squad = "Yurt İçi"
#                )
#                log("create_lead", doc)
#                create_res = {"created": True, "doc": doc}                
# #                    create_res = {"created": False, "reason": "lead_not_found"}
            
#             # return {"created": False, "reason": "lead_not_found"}
  
#         # 2) Çakışma kontrolü + oluşturma
#         # book_if_free metodunuz varsa onu kullanın:
# #         log("extra_filtres",extra_filters)  
        
# #         extra_filters = {"name": lead_doc["name"]}
# #         log("extra_filtres2", extra_filters)
#         lead_doc = None
#         if lead_name:
#             lead_doc = self.get_lead_by_name(lead_name, fields=["name","full_name","phone","email"])
#         else:
#             results = self.find_lead(
#                 phone=phone, email=email, full_name=full_name,
#                 filters=lead_filters, fields=["name","full_name","phone","email"],
#                 limit=1
#             )
#             if results:
#                 lead_doc = results[0]
#                 log("lead_doc", lead_doc)
#                 log("lead_doc.name", lead_doc.get("name"))
#         if not lead_doc:
#             return {"created": False, "reason": "lead_not_found"}
            
#         else:
            
#             if hasattr(self, "book_if_free"):
#                 return self.book_if_free(
#                     start_dt=start_dt,
#                     end_dt=end_dt,
#                     resource_id=resource_id,
#                     confirm=confirm,
#                     data_pool_reference=lead_doc["name"],
#                     hospital=hospital,
#                     examination_room=examination_room,
#                     service_provider=service_provider,
#                     appointment_type=appointment_type,
#                     appointment_notes=appointment_notes,
#                     extra_filters=extra_filters,
#                 )
        
#             # Eğer sizde book_if_free yoksa doğrudan create_appointment yapın (önce manuel çakışma bakılabilir):
#             if hasattr(self, "check_between"):
#                 exists, conflicts = self.check_between(start_dt, end_dt, resource_id, extra_filters=extra_filters)
#                 if exists:
#                     return {"created": False, "reason": "conflict", "conflicts": conflicts}
        
#             if not hasattr(self, "create_appointment"):
#                 raise NotImplementedError("create_appointment/book_if_free metodu bulunamadı")
        
#             doc = self.create_appointment(
#                 start_dt=start_dt,
#                 end_dt=end_dt,
#                 data_pool_reference=lead_doc["name"],
#                 hospital=hospital,
#                 examination_room=examination_room,
#                 service_provider=service_provider,
#                 appointment_type=appointment_type,
#                 appointment_notes=appointment_notes,
#             )
#             return {"created": True, "doc": doc}

    def book_if_free_with_lead(
        self,
        *,
        start_dt: str,
        end_dt: str,
        confirm: bool = True,
            # Lead için bakılacak anahtarlar:
        lead_name: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        lead_filters: Optional[List | Dict[str, Any]] = None,
            # Appointment alanları:
        hospital: Optional[str] = None,
        examination_room: Optional[str] = None,
        service_provider: Optional[str] = None,
        appointment_type: Optional[str] = None,
        appointment_notes: Optional[str] = None,
            # book_if_free passthrough:
        resource_id: Optional[str | int] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # """
        # 1) Lead'i (name/phone/email) ile bulur.
        # 2) Çakışma yoksa randevu yaratır (confirm=True).
        # 3) data_pool_reference = lead.name olarak set edilir.
        # """
   
        log("book_if_free_with_lead içinde")

        # --- 1) LEAD'i getir
        lead_doc: dict | None = None
    
        try:
            if lead_name:
                lead_doc = self.get_lead_by_name(
                    lead_name, fields=["name", "full_name", "phone", "email"]
                )
            else:
                results = self.find_lead(
                    phone=phone,
                    email=email,
                    full_name=full_name,
                    filters=lead_filters,
                    fields=["name", "full_name", "phone", "email"],
                    limit=1,
                ) or []
                if results:
                    lead_doc = results[0]
                    log("lead_doc", lead_doc)
                    log("lead_doc.name", (lead_doc or {}).get("name"))
        except Exception as e:
            log("[lead lookup] exception:", repr(e))
    
        # --- 2) Yoksa oluştur (gerekli alanları parametreden kullan!)
        if not lead_doc:
            if hasattr(self, "create_lead"):
                try:
                    doc = self.create_lead(
                        lead_validity="Geçerli",
                        lead_statu="RANDEVU VERİLDİ",
                        source_group="Dijital Pazarlama",
                        main_source="Website",   # alan adı sizin API’nize göre
                        sub_source="Website",
                        language="Türkçe",
                        country="Türkiye",
                        main_services="Belirsiz",
                        sub_services="",
                        full_name=full_name ,
                        phone=phone,
                        squad="Yurt İçi",
                        advertisement_message="Sanal Asistan tarafından üretildi",
                    )
                    log("create_lead", doc)
                    # create_lead dönüşü tipik olarak {"data": {...}} veya direkt dict olabilir
                    # normalize edelim:
                    if isinstance(doc, dict) and "data" in doc and isinstance(doc["data"], dict):
                        lead_doc = doc["data"]
                    else:
                        lead_doc = doc if isinstance(doc, dict) else None
                except Exception as e:
                    log("[create_lead] exception:", repr(e))
                    return {"created": False, "reason": f"create_lead_failed: {e!r}"}
            else:
                return {"created": False, "reason": "lead_not_found"}
    
        # Bu noktada lead_doc olmalı
        if not lead_doc or not isinstance(lead_doc, dict) or not lead_doc.get("name"):
            return {"created": False, "reason": "lead_missing_after_create"}
    
        lead_name_final = lead_doc["name"]
    
        # --- 3) Çakışma kontrolü + oluşturma
        # Mümkünse hazır fonksiyonunuzu kullanın
        if hasattr(self, "book_if_free"):
            try:
                return self.book_if_free(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    resource_id=resource_id,
                    confirm=confirm,
                    data_pool_reference=lead_name_final,  # Appointment↔Lead bağlama
                    hospital=hospital,
                    examination_room=examination_room,
                    service_provider=service_provider,
                    appointment_type=appointment_type,
                    appointment_notes=appointment_notes,
                    extra_filters=extra_filters,
                ) or {"created": False, "reason": "book_if_free_returned_none"}
            except Exception as e:
                log("[book_if_free] exception:", repr(e))
                # alttaki fallback’a devam
    
        # book_if_free yoksa/fail ise manuel kontrol + create_appointment
        try:
            if hasattr(self, "check_between"):
                exists, conflicts = self.check_between(
                    start_dt, end_dt, resource_id, extra_filters=extra_filters
                )
                if exists:
                    return {"created": False, "reason": "conflict", "conflicts": conflicts}
    
            if not hasattr(self, "create_appointment"):
                return {"created": False, "reason": "no_create_method"}
    
            doc = self.create_appointment(
                start_dt=start_dt,
                end_dt=end_dt,
                data_pool_reference=lead_name_final,
                hospital=hospital,
                examination_room=examination_room,
                service_provider=service_provider,
                appointment_type=appointment_type,
                appointment_notes=appointment_notes,
            )
            return {"created": True, "doc": doc}
        except Exception as e:
            log("[create_appointment] exception:", repr(e))
            return {"created": False, "reason": f"create_appointment_failed: {e!r}"}

    def is_range_free(
        self,
        start_dt: str,
        end_dt: str,
        resource_id: Optional[str | int] = None,
        *,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[Dict]]:
        """
        True => aralık boş (uygun), False => çakışma var.
        İkinci dönen değer çakışan kayıtların ham satırlarıdır.
        """
        log("is_range_free içinde")
        exists, rows = self.check_between(
            start_dt, end_dt, resource_id,
            extra_filters=extra_filters,
            fetch_fields=["name","start_appointment_time","end_appointment_time"]
        )
        return (not exists, rows)

    def _busy_intervals_for_day(
        self,
        date_str: str,
        resource_id: Optional[str | int] = None,
        *,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[time, time]]:
        """Gün boyu randevuları (start/end) toplayıp saat aralıklarına çevirir."""
        log("_busy_intervals_for_day içinde")
        log("[date_str]",date_str)
        
        if isinstance(date_str, datetime):
            date_str = date_str.strftime("%Y-%m-%d")
        elif isinstance(date_str, str) and "T" in date_str:
            date_str = date_str.split("T", 1)[0]
        
        day_start = f"{date_str} 00:00:00"
        day_end   = f"{date_str} 23:59:59"
        log("[day_start]", day_start,"[day_end]",day_end)
        _, rows = self.check_between(
            day_start, day_end, resource_id,
            extra_filters=extra_filters,
            fetch_fields=["start_appointment_time","end_appointment_time"]
        )
        out: List[Tuple[time,time]] = []
        for r in rows:
            s = _parse_any(r.get("start_appointment_time"))
            e = _parse_any(r.get("end_appointment_time"))
            if s and e and e > s:
                out.append((s.time(), e.time()))
        return out

    def free_slots(
        self,
        date_str: str,
        slot_size: int,
        resource_id: Optional[str | int] = None,
        *,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Eski availability'nin yaptığı gibi: gün için boş HH:MM slot listesi döner."""
        d = datetime.fromisoformat(date_str).date()
        wd = d.weekday()

        # Tatil/hafta sonu kuralları varsa koru
        if 'HOLIDAYS' in globals() and date_str in HOLIDAYS:
            return []
        if 'WEEKEND_ALLOWED' in globals() and not WEEKEND_ALLOWED and wd >= 5:
            return []

        # Çalışma bloklarını al (yoksa varsayılan 09:00-18:00)
        if 'WORKING_HOURS' in globals() and WORKING_HOURS.get(wd):
            blocks = WORKING_HOURS[wd]
        else:
            blocks = [(time(9,0), time(18,0))]

        busy = self._busy_intervals_for_day(date_str, resource_id, extra_filters=extra_filters)

        # Opsiyonel: mola bloklarını da “busy”e ekle
        if 'BREAKS' in globals():
            busy += BREAKS.get(wd, [])

        slots: List[str] = []
        for start_t, end_t in blocks:
            cur = datetime.combine(d, start_t)
            end = datetime.combine(d, end_t)
            while cur + timedelta(minutes=slot_size) <= end:
                s_t = cur.time()
                e_t = (cur + timedelta(minutes=slot_size)).time()
                if not any(_overlap(s_t, e_t, b_s, b_e) for (b_s, b_e) in busy):
                    slots.append(cur.strftime("%H:%M"))
                cur += timedelta(minutes=slot_size)
        # benzersiz & sıralı
        return sorted(set(slots))
    # --- Basit "availability": bu aralıkta kayıtlı randevuların BAŞLANGIÇ saatlerini döndürür
    # def availability(
    #     self,
    #     start_dt: str,
    #     end_dt: str,
    #     resource_id: Optional[str | int] = None,
    #     *,
    #     extra_filters: Optional[Dict[str, Any]] = None,
    # ) -> List[str]:
    #     exists, rows = self.check_between(start_dt, end_dt, resource_id, extra_filters=extra_filters,
    #                                       fetch_fields=["start_appointment_time", "end_appointment_time"])
    #     if not exists:
    #         return []
    #     hhmm = []
    #     for r in rows:
    #         s = _parse_any(r.get("start_appointment_time"))
    #         if s:
    #             hhmm.append(_to_hhmm(s))
    #     # benzersiz ve sıralı
    #     return sorted(set(hhmm))

   
    # def _busy_intervals(self, resource_id: int, day: date) -> List[Tuple[time, time]]:
    #     busy: List[Tuple[time, time]] = []
    #     for a in self.appts:
    #         if a["res_id"] != resource_id: continue
    #         if a["status"] in ("hold", "confirmed") and a["start"].date() == day:
    #             # süresi dolmuş hold'ları temizle
    #             if a["status"] == "hold" and a["hold_expires_at"] and datetime.now(IST) > a["hold_expires_at"]:
    #                 a["status"] = "cancelled"
    #                 continue
    #             busy.append((a["start"].time(), a["end"].time()))
    #     return busy

    # def availability(self, resource_id: int, date_str: str, slot_size: int = SLOT_SIZE_MIN) -> List[str]:
    #     if date_str in HOLIDAYS: return []
    #     d = datetime.fromisoformat(date_str).date()
    #     wd = d.weekday()
    #     if wd >= 5 and not WEEKEND_ALLOWED:
    #         return []
    #     blocks = WORKING_HOURS.get(wd, [])
    #     if not blocks: return []
    #     busy = self._busy_intervals(resource_id, d) + BREAKS.get(wd, [])
    #     slots: List[str] = []
    #     for start_t, end_t in blocks:
    #         cur = datetime.combine(d, start_t, IST)
    #         end = datetime.combine(d, end_t, IST)
    #         while cur + timedelta(minutes=slot_size) <= end:
    #             s_t = cur.time(); e_t = (cur + timedelta(minutes=slot_size)).time()
    #             if not any(not (e_t <= b_s or s_t >= b_e) for (b_s, b_e) in busy):
    #                 slots.append(cur.strftime("%H:%M"))
    #             cur += timedelta(minutes=slot_size)
    #     return slots

    def hold(self, resource_id: int, start_iso: str, end_iso: str, customer: Dict[str, str],
              notes: Optional[str], channel: Optional[str]) -> Tuple[Optional[Dict], bool]:
        start = datetime.fromisoformat(start_iso); end = datetime.fromisoformat(end_iso)
        # çakışma?
        for a in self.appts:
            if a["res_id"] != resource_id: continue
            if a["status"] in ("hold", "confirmed"):
                # hold süresi bitti mi?
                if a["status"] == "hold" and a["hold_expires_at"] and datetime.now(IST) > a["hold_expires_at"]:
                    a["status"] = "cancelled"; continue
                if not (end <= a["start"] or start >= a["end"]):
                    return None, True  # conflict
        appt_id = uuid.uuid4().hex[:12]
        self.appts.append({
            "id": appt_id, "res_id": resource_id,
            "start": start, "end": end, "status": "hold",
            "hold_expires_at": datetime.now(IST) + timedelta(seconds=HOLD_TTL_SEC)
        })
        return {"appointment_id": appt_id, "status": "hold", "expires_in": HOLD_TTL_SEC}, False

    def confirm(self, appt_id: str) -> Tuple[Optional[Dict], bool]:
        for a in self.appts:
            if a["id"] == appt_id:
                if a["status"] != "hold":
                    return {"status": a["status"]}, False
                if datetime.now(IST) > a["hold_expires_at"]:
                    a["status"] = "cancelled"
                    return None, True  # expired
                a["status"] = "confirmed"
                a["hold_expires_at"] = None
                return {"status": "confirmed"}, False
        return None, True  # bulunamadı → expired gibi davran

    def cancel(self, appt_id: str):
        for a in self.appts:
            if a["id"] == appt_id and a["status"] in ("hold", "confirmed"):
                a["status"] = "cancelled"
                return



# class GenericCRMAdapterHTTP:
#     """
#     GERÇEK CRM uçlarına geçince bunu kullan.
#     Aşağıdaki TODO kısımlarını doldurman yeterli.
#     """
#     def __init__(self, base_url: str, token: str):
#         self.base_url = base_url.rstrip("/")
#         self.token = token

#     # -- örnek imzalar; gerçek uçlarını burada eşleştir --
#     def availability(self, resource_id: int, date_str: str, slot_size: int = SLOT_SIZE_MIN) -> List[str]:
#         # TODO: HTTP POST {resource_id, date, slot_size} → {"slots": ["HH:MM" veya ISO]}
#         raise NotImplementedError

#     def hold(self, resource_id: int, start_iso: str, end_iso: str, customer: Dict[str, str],
#              notes: Optional[str], channel: Optional[str]) -> Tuple[Optional[Dict], bool]:
#         # TODO: HTTP POST /hold ... → 409 ise (None, True) döndür
#         raise NotImplementedError

#     def confirm(self, appt_id: str) -> Tuple[Optional[Dict], bool]:
#         # TODO: HTTP POST /confirm ... → 410 ise (None, True) döndür
#         raise NotImplementedError

#     def cancel(self, appt_id: str):
#         # TODO: HTTP POST /cancel ...
#         raise NotImplementedError

# =========================
# Planner (MVP)
# =========================
def run_planner(user_id: str, user_text: str, crm, ctx: Optional[Ctx] = None) -> Tuple[str, Ctx]:
    """
    Randevu planlayıcı.
    Kimlik: 'Ad( …) ; 5xx xxx xx xx' biçiminde istenir (isim 1–5 kelime olabilir).
    Detaylar: şube/gün/saat serbest metinden çıkarılır.
    """
    import re
    from datetime import datetime, timedelta, date as _date
    log ("Run_Planner Başladı")
    # ---------- yardımcılar ----------
    WD_WORDS = {"pazartesi","salı","sali","çarşamba","carsamba","perşembe","persembe","cuma","cumartesi","pazar"}
    MONTHS  = {"ocak","şubat","subat","mart","nisan","mayıs","mayis","haziran","temmuz","ağustos","agustos","eylül","eylul","ekim","kasım","kasim","aralık","aralik"}

    def _has_explicit_date(txt: str) -> bool:
        log("_has_explicit_date başladı")
        if not txt: return False
        t = txt.casefold()
        log(t)
        if any(w in t for w in {"bugün","bugun","yarın","yarin"}): return True
        if any(w in t for w in WD_WORDS): return True
        if any(m in t for m in MONTHS): return True
        pat = re.compile(r"\b(\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?|\d{4}-\d{2}-\d{2})\b")
        return bool(pat.search(txt))

    def _has_weekday_word(txt: str) -> bool:
        t = (txt or "").casefold()
        return any(w in t for w in WD_WORDS)

    def _name_for_prompt(st) -> str:
        nm = (clean_name(st.goal.customer.get("fullName")) or "").strip()
        return nm if nm else "Değerli misafirimiz"

    def _contains_any(txt, words): return any(w in (txt or "") for w in words)

    def _today_ist():
        try:
            now = datetime.now(IST)
        except Exception:
            now = datetime.now()
        return now

    def _iso_to_date(s: str) -> _date:
        return datetime.strptime(s, "%Y-%m-%d").date()

    # ---------- hazırlık ----------
    state = ctx or Ctx()
    state.memory = get_prefs(user_id)
    if not getattr(state, "status", None):
        state.status = "collecting"
    c = state.goal.constraints
    low  = (user_text or "")
    norm = normalize_tr(user_text or "")

    # ---------- yeni randevu / reset ----------
    if is_new_appt_intent(low) or is_explicit_reset(low):
        log ("Run_Planner 1")
        has_id  = bool(state.goal.customer.get("fullName") or state.goal.customer.get("phone"))
        has_con = any([getattr(c,"branch",None), getattr(c,"resourceId",None), getattr(c,"date",None), getattr(c,"approxTime",None)])
        if not (has_id or has_con) or is_explicit_reset(low):
            log ("Run_Planner 2")
            state.goal.customer["fullName"] = ""
            state.goal.customer["phone"]    = ""
            for attr in ("branch","resourceId","date","approxTime"):
                try: setattr(c, attr, None)
                except Exception: pass
            state.slots = []; state.offeredSlots = []; state.held = None; state.replans = 0
            state.status = "collecting"

    # ---------- confirming kısa devre ----------
    if getattr(state,"status",None) == "confirming" and getattr(state,"held",None):
        log ("Run_Planner 3")
        appt_id = state.held.get("appointment_id") if isinstance(state.held, dict) else None
        yes = ["evet","onay","onayla","onaylıyorum","kabul","uygundur","uygun","tamam","olur"]
        no  = ["hayır","hayir","iptal","iptal edelim","iptal et","vazgeç","vazgec","istemiyorum"]
        if _contains_any(norm, yes):
            try: crm.confirm(appt_id)
            except Exception: pass
            state.status = "idle"
            return ("Randevunuz onaylandı. Görüşmek üzere.", state)
        if _contains_any(norm, no):
            try: crm.cancel(appt_id)
            except Exception: pass
            state.status = "idle"; state.held = None
            return ("Randevunuz iptal edildi.", state)
        return ("Onaylıyor musunuz? Evet / İptal", state)

    # ---------- tek hekim fallback ----------
    if not getattr(c,"resourceId",None):
        log ("Run_Planner 4")
        if not ('BRANCHES' in globals() and BRANCHES):
            try: c.resourceId = RESOURCE_ID
            except NameError: pass

    # ========== A) KİMLİK — yalnız ';' formatı (isim 1–5 kelime) ==========
    full_name  = state.goal.customer.get("fullName")
    phone_disp = state.goal.customer.get("phone")
    log ("Run_Planner 5")
    log (full_name, phone_disp)
    def _need_identity():
        has_name  = bool(full_name)  # isim doğrulamasını parse fonksiyonu yaptı
        has_phone = parse_phone_any(phone_disp or "") is not None
        return (not has_name) or (not has_phone)

    if _need_identity():
        log ("Run_Planner 6")
        ident = None
        try: ident = parse_identity_semicolon(user_text)  # (name, disp, e164)
       
        except Exception: ident = None
        if ident:
            name, disp, _ = ident
            set_fullname(state, name, source="identity_semicolon")
            state.goal.customer["phone"] = disp
        else:
            state.meta = getattr(state,"meta",{}); state.meta["expecting_identity"] = True
            return (
                "Randevu için lütfen Adınızı Soyadınızı ve Telefon Numaranızı örnekteki gibi yazın.\n"
                "Örn: Ad Soyad ; 5xx xxxxxxx",
                state
            )
            # return ("Randevu talebinizi oluşturmak için ; KVKK Aydınlatma Metni: https://siteniz.com/kvkk.\n"
            #         "Okudum, onaylıyorum. (Evet,Hayır)",
            #         state
            # )

    # ========== B) DETAY — şube/gün/saat ==========
    branch_index = globals().get("BRANCH_INDEX", None)
    det = extract_details_freeform(user_text, branch_index)  # extractor 'bugün' varsayımı yapsa bile aşağıda maskeleyiyoruz

    provided_branch = bool(det.get("branch"))
    provided_date   = bool(det.get("date")) and _has_explicit_date(user_text)
    provided_time   = bool(det.get("time"))

    # ŞUBE: her zaman güncelle
    if det.get("branch"):
        c.branch = det["branch"]

    # TARİH: yalnızca açık ifade varsa set et
    if det.get("date") and _has_explicit_date(user_text):
        c.date = det["date"]

    # SAAT: bu turda geldiyse set et
    if det.get("time") and not getattr(c,"approxTime",None):
        c.approxTime = det["time"]

    # SAAT var & TARİH yok → yalnız bu turda saat verildiyse yakın tarihi seç
    if not c.date and c.approxTime and provided_time:
        c.date = choose_nearest_date_if_only_time(
            c.approxTime, state.policy["leadTimeMin"], state.policy["weekendAllowed"]
        )

    # --- sadece şube verildiyse, bu turda eski date/time'ı maskeleriz ---
    only_branch_this_turn = provided_branch and not (provided_date or provided_time)

    # --- Geçmiş tarih düzeltmesi & "perşembe" --> gelecek hafta semantiği ---
    # (1) Mesajda GÜN adı var ve c.date geçmişte ise: gelecek haftadaki aynı güne atla.
    if provided_date and _has_weekday_word(user_text) and c.date:
        try:
            d = _iso_to_date(c.date); today = _today_ist().date()
            while d <= today:
                d = _iso_to_date(shift_date(d.isoformat(), +7))
            c.date = d.isoformat()
        except Exception:
            pass

    # (2) Mesajda AY/GÜN (örn. '21 ağustos') olup tarih geçmişteyse: kullanıcıdan ileri tarih iste.
    past_date_note = False
    if provided_date and c.date:
        try:
            d = _iso_to_date(c.date); today = _today_ist().date()
            if d < today and not _has_weekday_word(user_text):
                past_date_note = True
        except Exception:
            pass

    # --- eksik/maskeli alanlar ---
    has_branch_name = bool(c.branch)
    has_date        = bool(c.date) and not only_branch_this_turn
    has_time        = bool(c.approxTime) and not only_branch_this_turn

    log(f"[DET] {det} | expl_date={_has_explicit_date(user_text)} "
        f"| prov(b,d,t)=({provided_branch},{provided_date},{provided_time}) mask={only_branch_this_turn} "
        f"| ctx_date={c.date} ctx_time={c.approxTime} status={state.status}")

    # ---------- PAST DATE uyarısı (döngüyü kes) ----------
    if past_date_note and has_branch_name and not has_time:
        state.status = "collecting"
        nm = _name_for_prompt(state)
        return (f"Sayın {nm}, seçtiğiniz **{c.date}** tarihi geçmişte. "
                f"Lütfen ileri bir **gün** belirtir misiniz? (örn. *Yarın* veya *Perşembe*)", state)

    # ---------- 1) Şube + Gün var, Saat yok → uygun saatleri öner ----------
    if has_branch_name and has_date and not has_time:
        if not getattr(c,"resourceId",None):
            rid = None
            if 'BRANCH_INDEX' in globals() and BRANCH_INDEX:
                rid = resolve_rid_from_display_any(BRANCH_INDEX, c.branch)
            if rid is None and 'BRANCHES' in globals() and BRANCHES:
                rid = resolve_rid_from_display_any(BRANCHES, c.branch)
            if rid is not None:
                c.resourceId = rid
            else:
                return ("Şube tanımı bulunamadı. Lütfen şubeyi tekrar belirtir misiniz?", state)

        state.status = "offering"
        # raw_slots = crm.availability(c.resourceId, c.date, state.policy["slotSizeMin"])
        try:
            slot_min = int(state.policy.get("slotSizeMin", 30))
        except Exception:
            slot_min = 30
        
        booking = ensure_booking_adapter(crm)  # <— burada garantili adaptör
        # --- TEŞHİS: bağlanan adaptör/methodlar ---
        log("ADAPTER:", type(booking).__name__,
            "has_free_slots:", hasattr(booking, "free_slots"),
            "has_check_between:", hasattr(booking, "check_between"))
        
        # --- 1) free_slots dene ---
#        date_str = datetime.now().date().isoformat()  # "YYYY-MM-DD"
        date_str = c.date
        raw_slots = booking.free_slots(date_str, slot_min, resource_id=c.resourceId)
        log("DEBUG raw_slots:", raw_slots)
        
        # Eğer hâlâ boşsa, nedenini anlamak için busy ve çalışma bloklarını yazdır
        wd = datetime.fromisoformat(c.date).date().weekday()
        blocks = WORKING_HOURS.get(wd, []) if 'WORKING_HOURS' in globals() else []
        log("DEBUG date:", c.date, "wd:", wd, "blocks:", blocks,
            "WEEKEND_ALLOWED:", globals().get("WEEKEND_ALLOWED", None),
            "is_holiday:", c.date in globals().get("HOLIDAYS", set()))
        
        # Gün boyu randevular (busy) – check_between ile
        if isinstance(date_str, datetime):
            date_str = date_str.strftime("%Y-%m-%d")
        elif isinstance(date_str, str) and "T" in date_str:
            date_str = date_str.split("T", 1)[0]
        
        day_start = f"{date_str} 00:00:00"
        day_end   = f"{date_str} 23:59:59"
        log("[day_start]", day_start,"[day_end]",day_end)
      
        if hasattr(booking, "check_between"):
            _, rows = booking.check_between(
                day_start, day_end, c.resourceId,
                fetch_fields=["start_appointment_time","end_appointment_time"]
            )
            log("DEBUG busy_rows:", len(rows), rows[:2])  # fazla uzamasın diye birkaçını göster
        else:
            rows = []
        
        # --- 2) normalize + lead-time (SADECE bugün için) ---
        slots = normalize_slots(date_str, raw_slots or [])
        log("SLOTS1:", slots, date_str)
        
        lead_min = int(state.policy.get("leadTimeMin", 120))
        now_ist = datetime.now(ZoneInfo("Europe/Istanbul"))
        log("[c.date]", c.date, "now_ist", now_ist.strftime("%Y-%m-%d"))
        is_today = (c.date == now_ist.strftime("%Y-%m-%d"))
        if is_today:
            threshold = now_ist + timedelta(minutes=lead_min)
            slots = [s for s in slots if datetime.fromisoformat(s) >= threshold]
        log("SLOTS2:", slots)
        
        if not slots:
            state.replans += 1
            if state.replans > state.policy["maxReplans"]:
                state.status = "failed"
                return (f"Üzgünüm, **{c.date}** için uygun saat bulamadım. Farklı bir gün ya da saat deneyelim mi?", state)
            # bir sonraki güne teklif et (metinde 'Bugün' demeyelim; gerçek tarihi yazalım)
            c.date = shift_date(c.date, +1)
            return (f"**{shift_date(c.date, -1)}** için uygun saat yok. **{c.date}** tarihi için deneyeyim mi?", state)

        state.slots = slots
        top = pick_top_slots(slots, state.memory, 3)
        state.offeredSlots = top
        nm = _name_for_prompt(state)
        return (f"Sayın {nm}, şu saatler uygun görünüyor: {fmt(top)}. Tercihiniz hangisi? (örn. 12:00)", state)

    # ---------- 2) Her şey tamam → HOLD ----------
   # ---------- 2) Her şey tamam → (HOLD YOK) UYGUNLUK KONTROL + ONAY ----------
   
    log("[has_branch_name]", has_branch_name, "[has_date]", has_date, "[has_time]", has_time)
    if has_branch_name and has_date and has_time:
        if not getattr(c, "resourceId", None):
            rid = None
            if 'BRANCH_INDEX' in globals() and BRANCH_INDEX:
                rid = resolve_rid_from_display_any(BRANCH_INDEX, c.branch)
            if rid is None and 'BRANCHES' in globals() and BRANCHES:
                rid = resolve_rid_from_display_any(BRANCHES, c.branch)
            if rid is None:
                state.status = "collecting"
                return ("Şube tanımı bulunamadı. Lütfen şubeyi tekrar belirtir misiniz?", state)
            c.resourceId = rid
    
        # Slot uzunluğu
        try:
            slot_min = int(state.policy.get("slotSizeMin", 30))
        except Exception:
            slot_min = 30
    
        # Başlangıç/bitişi üret
        tz_off = "+03:00"
        start_dt_str = f"{c.date} {c.approxTime}:00"  # 'YYYY-MM-DD HH:MM:SS'
        start_iso = f"{c.date}T{c.approxTime}:00{tz_off}"
    
        hh, mm = map(int, c.approxTime.split(":"))
        date_only = _date_only(c.date)
        hhmm      = _hhmm_only(c.approxTime)
        log("[c.date]", c.date, "[date_only]", date_only)
        start_dt_str = f"{date_only} {hh:02d}:{mm:02d}"  # 'YYYY-MM-DD HH:MM'
        end_dt = datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M") + timedelta(minutes=slot_min)
        end_dt_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        tz_off = "+03:00"
        start_iso = f"{date_only}T{hh:02d}:{mm:02d}:00{tz_off}"
        end_iso   = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_off
#        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_off
    
        # lead-time (yakın saatleri reddet)
        lead_min = int(state.policy.get("leadTimeMin", 120))
        now_naive = datetime.now()
        
        start_naive = datetime.strptime(f"{date_only} {hhmm}", "%Y-%m-%d %H:%M")
        log("[start_naive]", start_naive)
#        start_naive = datetime.strptime(f"{c.date} {c.approxTime}", "%Y-%m-%d %H:%M")
        if start_naive < now_naive:
            state.status = "collecting"
            return ("Seçtiğiniz saat geçmiş görünüyor. Biraz daha ileri bir saat/dakika yazabilir misiniz?", state)
        
        elif start_naive < now_naive + timedelta(minutes=lead_min):
            state.status = "collecting"
            return ("Seçtiğiniz saat çok yakın görünüyor. Biraz daha ileri bir saat/dakika yazabilir misiniz?", state)
    
        # 1) Verilen aralık uygun mu? (check_between kullanan sarmalayıcı)
        booking = ensure_booking_adapter(crm)
        try:
            if hasattr(booking, "is_range_free"):
                is_free, conflicts = booking.is_range_free(
                    start_dt_str, end_dt_str, resource_id=c.resourceId
                )
            elif hasattr(booking, "check_between"):
                exists, rows = booking.check_between(
                    start_dt_str, end_dt_str, c.resourceId
                )
                is_free, conflicts = (not exists, rows)
            else:
                # booking adaptöründe beklenen yüzey yoksa: açık hata
                raise AttributeError("Adapter has neither is_range_free nor check_between")
        except Exception as e:
            # Teşhis için bir log bırak (kullanıcıya nazik mesaj)
            try:
                log(f"[availability-check] error: {e}")  # varsa log()
            except Exception:
                pass
            state.status = "collecting"
            return ("Şu an uygunluk kontrolünde bir sorun oluştu. Biraz sonra tekrar dener miyiz?", state)
    
        if not is_free:
            state.status = "collecting"
            return ("Üzgünüm, bu saat **dolu** görünüyor. Farklı bir saat önereyim mi?", state)
    
        rid = _ensure_resource_id(c)
        if not rid:
            state.status = "collecting"
            return ("Şube tanımı bulunamadı. Lütfen şubeyi tekrar belirtir misiniz?", state)
    # 2) Uygun → 'soft-hold' bilgilerini state'e koy, onay iste
        _merge_held(
            state,
            resource_id=getattr(c, "resourceId", None),
            start_iso=start_iso,
            end_iso=end_iso,
            start_dt=start_dt_str,
            end_dt=end_dt_str,
            branch=c.branch,
        )
        state.date_iso = c.date
        state.time_hm = c.approxTime
        state.branch = c.branch
        state.status = "confirming"
        nm = _name_for_prompt(state)
        log("Onay öncesi [state]",state)
        return (f"{c.date} {c.approxTime} için {c.branch} şubemizde randevu **uygun** görünüyor. Onaylıyor musunuz?", state)


    # ---------- 3) Eksik — yalnız eksikleri sor ----------
    nm = _name_for_prompt(state)
    if (not has_branch_name) and (not has_date) and (not has_time):
        state.status = "collecting"
        return (f"Sayın {nm},  hangi şubemizde, hangi **gün ve saat** uygundur? Örn: Kozyatağı, Perşembe 14:30", state)
    if has_branch_name and (not has_date) and (not has_time):
        state.status = "collecting"
        return (f"Sayın {nm}, **{c.branch}** için hangi **gün ve saat** uygundur? Örn: Perşembe 14:30", state)
    if has_branch_name and has_date and (not has_time):
        state.status = "collecting"
        return (f"Sayın {nm}, **{c.branch} – {c.date}** için hangi **saat** uygundur? Örn: 14:30", state)
    if has_branch_name and (not has_date) and has_time:
        state.status = "collecting"
        return (f"Sayın {nm}, **{c.branch} – {c.approxTime}** için hangi **gün/tarih** uygundur? Örn: Yarın veya 21 Ağustos",state)
    if (not has_branch_name) and has_date and (not has_time):
       state.status = "collecting"
       return (f"Sayın {nm}, hangi **şubemiz** uygundur? Örn: Bakırköy", state)

    if (not has_branch_name) and (not has_date) and has_time:
       state.status = "collecting"
       return (f"Sayın {nm}, hangi **gün/tarih** ve hangi **şube**? "
              f"Örn: Bakırköy yarın", state)

    nm = _name_for_prompt(state)
    return (f"Sayın {nm}, hangi şube, hangi gün ve saat uygundur? "
            f"Örn: Bakırköy, Perşembe 14:30", state)

#    print("DEBUG_HOLD_CHECK:", c.branch, c.resourceId, c.date, c.approxTime) 

def on_user_select_slot(state: Ctx, chosen_hhmm: str, crm) -> Tuple[str, Ctx]:
    c = state.goal.constraints
    log ("on_user_select_slot başladı")
    log ("c=", c)
    # ---- Zorunlu alan kontrolleri ----
    if not c.date:
        state.status = "collecting"
        return ("Randevu için gün bilgisini paylaşır mısınız? (örn. 21 Ağustos veya Perşembe)", state)

    if not (c.branch and c.resourceId):
        state.status = "collecting"
        try:
            branch_opts = " / ".join(n.title() for n in BRANCHES.keys())
        except NameError:
            branch_opts = None
        reply = "Hangi şubede randevu istersiniz?"
        if branch_opts:
            reply += f" ({branch_opts})"
        return (reply, state)

    full_name = (state.goal.customer.get("fullName") or "").strip()
    # parts = [p for p in full_name.split() if p]
    # if len(parts) < 2:
    #     state.status = "collecting"
    #     return ("Randevuyu ayırmadan önce ad soyadınızı yazar mısınız? ", state)

    # Telefonu her türlü yazımdan normalize et (parse_phone_any fonksiyonu sende mevcut olmalı)
    norm = parse_phone_any(state.goal.customer.get("phone") or "")
    if not norm:
        state.status = "collecting"
        return ("Telefon numaranızı yazar mısınız", state)
    disp_phone, e164_phone = norm
    state.goal.customer["phone"] = disp_phone

    # ---- Slotu ayır (HOLD) ----
    date_only = c.date.split("T", 1)[0] if isinstance(c.date, str) else c.date
    start_iso = to_iso_on_date(date_only, chosen_hhmm)
    end_iso   = to_iso_end(start_iso, state.policy["slotSizeMin"])

    state.status = "holding"
    customer = {
        "full_name": full_name,
        "phone": e164_phone,  # CRM'e E.164 gönder
    }

    # hold_resp, conflict = crm.hold(
    #     c.resourceId, start_iso, end_iso, customer, state.goal.notes, state.goal.channel
    # )
    ret = crm.hold(c.resourceId, start_iso, end_iso, customer, state.goal.notes, state.goal.channel)
    log(f"[hold] type={type(ret).__name__} value={ret!r}")
    log(ret)
    hold_resp, conflict = None, None
    if isinstance(ret, (tuple, list)):
        # ilk iki öğeyi al, fazlasını yok say
        hold_resp = ret[0] if len(ret) > 0 else None
        conflict  = ret[1] if len(ret) > 1 else None
    elif isinstance(ret, dict):
        # olası anahtarlar
        hold_resp = ret.get("hold") or ret.get("data") or ret.get("reservation") or ret
        conflict  = ret.get("conflict") or ret.get("is_conflict") or ret.get("busy")
    else:
        # tek nesne dönmüşse
        hold_resp = ret
    log(f"HOLD_RESP : {hold_resp}")    
    if conflict:
        state.replans += 1
        if state.replans > state.policy["maxReplans"]:
            state.status = "failed"
            return ("Seçtiğiniz saat yeni doldu. Farklı bir saat deneyelim mi?", state)

        alternatives = [s for s in state.slots if s != start_iso]
        top = pick_top_slots(alternatives, state.memory, 3)
        state.offeredSlots = top
        state.status = "offering"
        return (f"Az önce seçilen saat doldu. Alternatifler: {fmt(top)}", state)

    # ---- Başarılı HOLD → isim/şube/gün/saat ile onay metni ----
    appt_id = _extract_appt_id(hold_resp)
    log("APPT_ID =", appt_id)
    expires_sec = _extract_expires_sec(hold_resp) or 300  # bulunamazsa varsayılan 300 sn
    
    if not appt_id:
        state.status = "collecting"
        state.held = None
        return ("Randevu detayı bulunamadı. Yeniden saat seçebilir miyiz?", state)
    
    state.held = Held(
        apptId=appt_id,
        slotISO=start_iso,
        expiresInSec=expires_sec
    )
    state.status = "waiting_confirm"
    log(f"[hold-norm] appt_id={appt_id} expires_sec={expires_sec} raw_keys={list(hold_resp.keys()) if isinstance(hold_resp, dict) else type(hold_resp).__name__}")

    dt_local = datetime.fromisoformat(start_iso).astimezone(IST)
    gun = WEEKDAYS_TR[dt_local.weekday()]          # örn. "perşembe"
    saat = dt_local.strftime("%H:%M")              # örn. "12:00"
    sube = (c.branch or "").title()                # "Kozyatağı"

    reply = (
        f"Sayın {full_name}, {sube} şubemizde {gun} günü saat {saat} için yerinizi geçici olarak ayırdım. "
        f"Onaylıyor musunuz? (Onay/İptal)"
    )
    return (reply, state)




# def on_user_confirm(state: Ctx, crm) -> Tuple[str, Ctx]:
#     log("on_user_confirm başladı")
#     log(state.held)
#     if not state.held:
#         return ("Onaylanacak aktif bir randevu yok görünüyor.", state)

#     res, expired = crm.confirm(state.held.apptId)
#     if expired:
#         state.status = "offering"
#         alt = pick_top_slots([s for s in state.slots if s != state.held.slotISO], state.memory, 3)
#         state.offeredSlots = alt
#         state.held = None
#         return ("Onay süresi doldu. Şu alternatiflerden seçebilirsiniz: " + fmt(alt), state)

#     # Başarı
#     state.status = "confirmed"

#     # (İstersen tercih belleğini güncel tutmaya devam edelim)
#     picked_dt = datetime.fromisoformat(state.held.slotISO).astimezone(IST)
#     hour_bucket = f"{picked_dt.hour:02d}:00"
#     update_prefs("user:"+str(state.goal.customer.get("id") or "guest"), {
#         "preferredHourStart": hour_bucket,
#         "preferredBranch": state.goal.constraints.branch
#     })

#     # Mesajı istediğin şekilde kur
#     gun   = WEEKDAYS_TR[picked_dt.weekday()]      # "Perşembe"
#     saat  = picked_dt.strftime("%H:%M")               # "12:00"
#     sube  = (state.goal.constraints.branch or "Merkez").title()

#     reply = (
#         f"Harika! Randevunuz onaylandı, {sube} şubesinde {gun} günü saat {saat}'de görüşmek üzere!"
#     )
#     return (reply, state)

def on_user_confirm(state, crm, crm_booking=None):
    """Hold'u confirm et; başarılıysa telefonu lead'e bağlayıp appointment yarat."""
    log("On_user_confirm içinde","[state]",state)
    # --- ufak yardımcılar ---
    # def _extract_appt_id(held):
    #     if isinstance(held, dict):
    #         return held.get("appointment_id") or held.get("id") or held.get("name")
    #     return None

    def _fmt_yyyy_mm_dd_hh_mm_ss(x):
        from datetime import datetime as _dt
        if not x:
            return None
        try:
            return _dt.fromisoformat(x).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(x)

    def _get_phone_from_state(st):
        # kendi state şemana göre burayı genişletebilirsin
        for path in [
            ("phone",),
            ("Goal","customer","phone"),
            ("goal","customer","phone"),
            ("customer","phone"),
            ("user","phone"),
        ]:
            cur = st
            log("path:", path)
            log("cur:", cur)
            try:
                for p in path:
                    cur = getattr(cur, p) if not isinstance(cur, dict) else cur.get(p)
                log("cur:",cur)  
                disp,e164 = parse_phone_any(cur)
                if e164:
                    return e164
                
            except Exception:
                pass
        return None
    def _get(field_source, *keys):
        for k in keys:
            if isinstance(field_source, dict):
                v = field_source.get(k)
            else:
                v = getattr(field_source, k, None)
            if v:
                return v
        return None

    def _to_std_str(dt_like):
        """ISO (tz’li) veya 'YYYY-MM-DD HH:MM[:SS]' → 'YYYY-MM-DD HH:MM:SS'."""
        if dt_like is None:
            return None
        s = str(dt_like).strip()
        # ISO (timezone'lu) dene
        try:
            return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        # T->space, Z/+... kırp
        s2 = s.replace("T", " ").split("Z")[0].split("+")[0].strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s2, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        return None
    # --- 1) confirm ---
    if not getattr(state, "held", None):
        return ("Onaylanacak aktif bir randevu yok görünüyor.", state)

    appt_id = _extract_appt_id(state.held)
    log("APPT_ID =", appt_id)
    if not appt_id:
        state.status = "collecting"
        state.held = None
        return ("Randevu detayı bulunamadı. Yeniden saat seçebilir miyiz?", state)

    ok = False
    try:
        res = crm.confirm(appt_id)
        if isinstance(res, tuple):
            ok = bool(res[0])
        else:
            ok = bool(res)
    except Exception as e:
        # kullanıcı deneyimini bozma: iyimser davran
        ok = True
    log("RES=",res)    
    # --- 2) confirm başarılıysa: lead bul -> appointment yarat ---
    create_res = {"created": False, "reason": "unknown"}
    if ok:
        # a) booking için gerçek adapter'ı hazırla
        try:
            booking = crm_booking
            if booking is None:
                # crm FakeCRM ise booking yetenekleri yok; gerçek HTTP adaptörü oluştur
                has_booking = all(hasattr(crm, m) for m in ("book_if_free_with_lead","find_lead","book_if_free","create_appointment"))
                booking = crm if has_booking else make_crm_adapter()
        except Exception:
            booking = make_crm_adapter()

        # b) telefon/saat bilgileri
        log("state",state)
        log("_get_phone_from_state(state)",_get_phone_from_state(state))
        phone = _get_phone_from_state(state) or "+90 5351001010"  # fallback'ı kendine göre değiştir
        # held içinden saatleri almaya çalış
        log("state.held:",state.held)
        log("state.memory:",state.memory)
        log("getattr:", getattr(state.held, "slotISO", None))
        log("full_name:", state.goal.customer.get("fullName"))
        full_name = state.goal.customer.get("fullName")
#        start_dt = _fmt_yyyy_mm_dd_hh_mm_ss(getattr(state.held, "slotISO", None))
#        end_dt = (datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15))\
#         .strftime("%Y-%m-%d %H:%M:%S")
        # yoksa sabit saat kullan
        raw_start = _get(state.held, "slotISO", "start_iso", "start", "slotStart", "slot")
        start_dt = _to_std_str(raw_start)

        if not start_dt:
    # burada istersen fallback ver ya da kullanıcıdan yeniden iste
            raise ValueError(f"Start time yok/parse edilemedi. held={repr(state.held)}")

        end_dt = (datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15)
                  ).strftime("%Y-%m-%d %H:%M:%S")

        if not start_dt or not end_dt: 
            start_dt = "2025-10-11 10:00:00"
            end_dt   = "2025-10-11 10:30:00"
        log("start_dt =", start_dt, "end_dt =",end_dt)
        log("phone önce",phone)
        # c) mümkünse tek adımda: book_if_free_with_lead
        try:
            if hasattr(booking, "book_if_free_with_lead"):
                create_res = booking.book_if_free_with_lead(
                    start_dt=start_dt, end_dt=end_dt, confirm=True,
                    phone=phone,
                    full_name=full_name,
                    hospital="Kozyatağı Vertigocenter",
                    examination_room="KV-TEST ve RANDEVU ODASI",
                    appointment_type="Ön Görüşme",
                    service_provider="İrem Kevser Uygun",
                    appointment_notes="Chatbot tarafından otomatik oluşturuldu",
                )
                if "reason" not in create_res:
                    create_res["reason"] = "ok" if create_res.get("created") else "failed"
                log("book_if_free_with_lead", create_res)
            else:
                # d) parça parça: find_lead -> book_if_free (veya create_appointment)
                lead_doc = None
                if hasattr(booking, "find_lead"):
                    try:
                        log("[CALL] before find_lead phone=", phone)
                        found = booking.find_lead(phone=phone, fields=["name","full_name","phone","email"], limit=1) or []
                        log("[found-type]", type(found), "len", (len(found) if hasattr(found, "__len__") else "NA"))
                        log("[found]", found)
                    
                        if found:
                            log("find_lead", found)
                            lead_doc = found[0]
                            create_res = {"created": False, "reason": "unknown"}
                        else:
                            # Lead yokken ilerlemeyin; burada erken dönün veya kullanıcıyı bilgilendirin
                            create_res = {"created": False, "reason": "lead_not_found"}
                            created = create_res["created"]
                            reason  = create_res["reason"]
                            log("[created]", created, "[reason]", reason)
                        if not lead_doc:
        # lead bulunmadıysa lead yaratacak    
                            if hasattr(booking, "create_lead"):
                                # çakışma kontrolü yoksa doğrudan yarat
                                doc = booking.create_lead(
                                    lead_validity="Geçerli",
                                    lead_statu="RANDEVU VERİLDİ",
                                    source_group="Dijital Pazarlama",
                                    main_source="Website",   # alan adı sizin API’nize göre
                                    sub_source="Website",
                                    language="Türkçe",
                                    country="Türkiye",
                                    main_services="Belirsiz",
                                    sub_services="",
                                    full_name=full_name ,
                                    phone=phone,
                                    squad="Yurt İçi",
                                    advertisement_message="Sanal Asistan tarafından üretildi",
                                )
                                log("create_lead", doc)
                                create_res = {"created": True, "doc": doc}
                                if "reason" not in create_res:
                                    create_res["reason"] = "ok" if create_res.get("created") else "failed"
        #                    create_res = {"created": False, "reason": "lead_not_found"}
                    except Exception as e:
                        import traceback as _tb
                        log("[CALL] EXC:", repr(e), "\n", _tb.format_exc())
                        create_res = {"created": False, "reason": f"exception: {e}"}
               
                lead_doc = None
                if hasattr(booking, "find_lead"):
                    log("find lead 2")
                    found = booking.find_lead(phone=phone, fields=["name","full_name","phone"], limit=1)
                    log("[found2]", found)
                    if found:
                        log("find_lead3", found)
                        lead_doc = found[0]
                        create_res = {"created": False, "reason": "unknown"}
                    else:
                        # Lead yokken ilerlemeyin; burada erken dönün veya kullanıcıyı bilgilendirin
                        create_res = {"created": False, "reason": "lead_not_found"}
                        created = create_res["created"]
                        reason  = create_res["reason"]
                        log("[created2]", created, "[reason2]", reason)
                      
                    log("lead_doc =", lead_doc)
                if not lead_doc:
                    create_res = {"created": False, "reason": "lead_not_found"}
                else:
                    if hasattr(booking, "book_if_free"):
                        create_res = booking.book_if_free(
                            start_dt=start_dt, end_dt=end_dt, confirm=True,
                            data_pool_reference=lead_doc["name"],
                            hospital="Kozyatağı Vertigocenter",
                            examination_room="KV-TEST ve RANDEVU ODASI",
                            appointment_type="Ön Görüşme",
                            service_provider="İrem Kevser Uygun",
                            appointment_notes=f"Chatbot tarafından oluşturuldu: {lead_doc.get('full_name','?')} / {lead_doc.get('phone','')}",
                        )
                        log("book_if_free", create_res)
                        if "reason" not in create_res:
                            create_res["reason"] = "ok" if create_res.get("created") else "failed"
                    elif hasattr(booking, "create_appointment"):
                        # çakışma kontrolü yoksa doğrudan yarat
                        doc = booking.create_appointment(
                            start_dt=start_dt, end_dt=end_dt,
                            data_pool_reference=lead_doc["name"],
                            hospital="Kozyatağı Vertigocenter",
                            examination_room="KV-TEST ve RANDEVU ODASI",
                            appointment_type="Ön Görüşme",
                            service_provider="İrem Kevser Uygun",
                            appointment_notes=f"Chatbot tarafından oluşturuldu: {lead_doc.get('full_name','?')} / {lead_doc.get('phone','')}",
                        )
                        log("create_appointment", doc)
                        create_res = {"created": True, "doc": doc, "reason": "ok"}
                    else:
                        create_res = {"created": False, "reason": "no_booking_methods"}
        except Exception as e:
            # oluşturma hatasını kullanıcıya yansıtma
            log("exception", e)
            create_res = {"created": False, "reason": f"exception: {e}"}
        # (istersen burada create_res'i logla)

    log("[created]",create_res["created"],"[reason]", create_res["reason"])      # False lead_not_found
    log(create_res.get("created"), create_res.get("reason"))
    if create_res.get("created") : 
        state.status = "idle"
        msg = "Randevunuz onaylandı. Görüşmek üzere." 
    elif create_res.get("reason") == "conflict":        
        msg = "Seçtiğiniz saat dolu görünüyor, başka bir saat denemek ister misiniz ?"
    elif create_res.get("reason") == "lead_not_found":
        msg = "Müşteri kaydınız bulunamadı. Daha sonra tekrar deneyebilirsiniz."
    else:
        msg = "Onaylama sırasında bir sorun oluştu. Biraz sonra tekrar deneyebiliriz."
    return (msg, state)



def on_user_reject(state: Ctx, crm) -> Tuple[str, Ctx]:
    if state.held:
        try: crm.cancel(state.held.apptId)
        except Exception: pass
        state.held = None
    state.status = "offering"
    more = pick_top_slots(state.slots[3:], state.memory, 3) if len(state.slots) > 3 else []
    state.offeredSlots = more
    if not more:
        return ("Peki, başka bir gün veya saat aralığı önerebilirim. Hangi gün/saat aralığı uygundur?", state)
    return ("Tamam, şu saatleri deneyebiliriz: " + fmt(more), state)


# def on_user_cancel(state: Ctx, crm) -> Tuple[str, Ctx]:
#     """
#     Kullanıcı randevuyu tamamen iptal etmek istiyor. Alternatif saat önermeden iptal et.
#     """
#     # Aktif bir hold/confirmed var mı?
#     if state.held and state.status in {"holding","waiting_confirm","confirmed"}:
#         # CRM'de de iptal et (hold/confirmed fark etmez)
#         try:
#             crm.cancel(state.held.apptId)
#         except Exception:
#             pass

#         # Bilgilendirici mesaj
#         dt_local = datetime.fromisoformat(state.held.slotISO).astimezone(IST)
#         gun  = WEEKDAYS_TR[dt_local.weekday()]
#         saat = dt_local.strftime("%H:%M")
#         sube = (state.goal.constraints.branch or "Merkez").title()

#         state.status = "cancelled"
#         msg = f"{sube} şubesindeki {gun} günü saat {saat} randevunuz iptal edildi. Yardımcı olabileceğim başka bir konu var mı?"
#         # İptal sonrası elde tutulan veriyi temizlemek istersen:
#         # state.held = None
#         return (msg, state)

#     # Hiç aktif randevu yoksa:
#     return ("İptal edilecek aktif bir randevunuz görünmüyor.", state)

def on_user_cancel(state: Ctx, crm) -> Tuple[str, Ctx]:
    if not state.held:
        return ("İptal edilecek aktif bir randevu görünmüyor.", state)

    appt_id = _extract_appt_id(state.held)
    try:
        if appt_id:
            crm.cancel(appt_id)
    except Exception as e:
        log(f"[CANCEL] CRM error: {e}")

    state.status = "idle"
    state.held = None
    return ("Randevunuz iptal edildi.", state)



# =========================
# (Opsiyonel) Basit CLI demo
# =========================
# if __name__ == "__main__":
#     crm = GenericCRMAdapterInMemory()
#     ctx: Optional[Ctx] = None
#     user_id = "demo-user"

#     print("→ Kullanıcı: 'Perşembe öğleden sonra Kozyatağı, 12:00 uygun mu?'")
#     reply, ctx = run_planner(user_id, "Perşembe öğleden sonra Kozyatağı, 12:00 uygun mu?", crm, ctx)
#     print("← Bot:", reply)

#     # kullanıcı 12:00 dedi
#     print("→ Kullanıcı: '12:00'")
#     reply, ctx = on_user_select_slot(ctx, "12:00", crm)
#     print("← Bot:", reply)

#     # kullanıcı onaylıyor
#     print("→ Kullanıcı: 'Onaylıyorum'")
#     reply, ctx = on_user_confirm(ctx, crm)
#     print("← Bot:", reply)
