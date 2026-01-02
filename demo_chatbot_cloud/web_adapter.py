# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:27:12 2025

@author: Serkan
"""

# web_adapter.py — FastAPI katmanı (API)
# -------------------------------------------------
# Gereksinimler (aktif .venv içinde kur):
#   pip install fastapi uvicorn pydantic
# -------------------------------------------------

from fastapi import HTTPException, Depends, Header
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import asyncio, time, logging

# --- Import path emniyeti: bu dosyanın klasörünü sys.path'e ekle ---
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- Senin ana fonksiyonun (app_embedding_crm_prompt.py ile aynı klasörde olmalı) ---
from app_embedding_crm_prompt import answer as core_answer  # def answer(question: str, sid: str) -> str

# === FastAPI app ===
app = FastAPI(title="Chatbot Web Adapter", version="1.0.0")

# 🔒 TOKEN AYARI
API_TOKEN = "abcd1234securetoken"  # Örn: "abcd1234securetoken"

# CORS: Frontend origin(ler)ini buraya yaz
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8001",
    "http://localhost:5500",
    "https://api-saily.com",
    "https://sailyai.com",
    "https://chat.api.sailyai.com",
    "https://wwww.odyoduyu.com",
    "http:://hospitable-love-production.up.railway.app",
    # "https://www.senin-site.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Basit istek loglama (uvicorn.access değil; çakışmasın diye ayrı logger)
logger = logging.getLogger("app.access")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({dt_ms:.1f} ms)")
    return response

# 🔐 Token kontrol fonksiyonu
def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")

    return True


# Sağlık ucu
@app.get("/ping")
def ping():
    return {"pong": True}

# İstek/yanıt şemaları
class ChatIn(BaseModel):
    session_id: str = Field(..., description="Oturum anahtarı (sid)")
    message: str = Field(..., description="Kullanıcı mesajı / soru")
    kvkk_ok: bool = Field(False, description="KVKK onayı (checkbox)")
    whatsapp_ok: bool = Field(False, description="Whatsapp mesajı")

class ChatOut(BaseModel):
    reply: str

# Answer çağrısı için güvenli zaman aşımı (sn)
ANSWER_TIMEOUT = 25

@app.post("/api/chat", response_model=ChatOut)
async def chat(in_: ChatIn) -> ChatOut:
    """
    Frontend'den gelen mesajı app_embedding_crm_prompt.answer(question, sid) ile yanıtlar.
    question = in_.message
    sid      = in_.session_id
    """
    
    t0 = time.perf_counter()
    logger.info(f"CALL answer sid={in_.session_id} len(msg)={len(in_.message)} kvkk_ok={in_.kvkk_ok} whatsapp_ok={in_.whatsapp_ok}")

    try:
        # answer senkron → event loop'u bloklamamak için threadpool'da
        reply: str = await asyncio.wait_for(
            run_in_threadpool(core_answer, in_.message, in_.session_id, in_.kvkk_ok, in_.whatsapp_ok),
            timeout=ANSWER_TIMEOUT
        )
    except asyncio.TimeoutError:
        dt = (time.perf_counter() - t0) * 1000
        logger.error(f"TIMEOUT after {dt:.1f} ms (>{ANSWER_TIMEOUT}s)")
        return ChatOut(reply=f"Üzgünüm, yanıt üretme süresi aşıldı (~{ANSWER_TIMEOUT} sn).")
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.exception(f"ERROR after {dt:.1f} ms: {e}")
        return ChatOut(reply=f"Üzgünüm, beklenmeyen bir hata oluştu: {e}")

    dt = (time.perf_counter() - t0) * 1000
    logger.info(f"RET answer in {dt:.1f} ms len(reply)={len(reply or '')}")
    return ChatOut(reply=reply or "")


    