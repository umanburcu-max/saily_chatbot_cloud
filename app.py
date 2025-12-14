# -*- coding: utf-8 -*-
from __future__ import annotations


"""
Created on Wed Oct 29 14:05:32 2025

@author: Serkan


"""


"""
Flask API example that logs chatbot conversations to a SQL database.
- One-file starter: run directly with `python app.py`
- Uses SQLite by default (file: chatlogs.db). Switch to PostgreSQL by changing DATABASE_URL.
- Endpoints:
    POST /chat {"session_id": "abc123", "message": "hi"}
    GET  /conversations/<conversation_id>/messages
    GET  /sessions/<session_id>/messages
    GET  /conversations?limit=50

Replace `generate_reply()` with your real chatbot call.
"""


import os
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

import requests
from flask import Flask, request, jsonify
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session

from flask import send_from_directory

# -----------------------------
# Config
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chatlogs.db")
APP_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

# Upstream chatbot API (Saily)
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api-saily.com/api/chat")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY")  # optional – if your API requires a bearer or custom key
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "30"))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
# -----------------------------
# DB Setup
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()



class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(64), unique=True, index=True, nullable=False)
    session_id = Column(String(128), index=True, nullable=False)  # your app/session identifier
    user_id = Column(String(128), index=True, nullable=True)      # optional: authenticated user id
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta = Column(JSON, nullable=True)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat() + "Z",
            "last_activity_at": self.last_activity_at.isoformat() + "Z",
            "meta": self.meta or {},
        }

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id_fk = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(32), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional observability fields
    request_id = Column(String(64), index=True, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    token_input = Column(Integer, nullable=True)
    token_output = Column(Integer, nullable=True)
    meta = Column(JSON, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id_fk", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() + "Z",
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "token_input": self.token_input,
            "token_output": self.token_output,
            "meta": self.meta or {},
        }
    
class KvkkConsent(Base):
    __tablename__ = "kvkk_consents"

    id = Column(Integer, primary_key=True, index=True)

    # Conversation ile ilişki (opsiyonel ama faydalı)
    conversation_id_fk = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    conversation = relationship("Conversation", backref="kvkk_consents")

    # Aynı zamanda session_id de tutalım (widget tarafındaki SESSION)
    session_id = Column(String(128), index=True, nullable=True)

    # Kişiyi tanımlamaya yarayacak alanlar (şimdilik None da olabilir)
    phone = Column(String(32), index=True, nullable=True)
    name = Column(String(128), nullable=True)

    # Onay bilgisi
    consent_given = Column(Boolean, nullable=False)        # True = onay verdi
    consent_text_version = Column(String(32), nullable=True)  # Örn: "v1.0"
    consent_text_hash = Column(String(128), nullable=True)    # İmzalanan metnin hash’i

    # Teknik bilgiler
    channel = Column(String(32), nullable=True)       # "web_widget", "whatsapp" vs.
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(Text, nullable=True)

    # Zaman damgası
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


Base.metadata.create_all(engine)

# -----------------------------
# Flask App
# -----------------------------
#app = Flask(__name__)

#app = Flask(__name__, static_folder=".", static_url_path="")

from flask_cors import CORS
CORS(app, resources={r"/chat": {"origins": "*"}})  # üretimde * yerine kendi sitenizi yazın

# Utility: create or fetch a conversation for a given session

def _get_or_create_conversation(db, session_id: str, user_id: Optional[str] = None) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.last_activity_at.desc())
        .first()
    )
    if conv is None:
        conv = Conversation(
            conversation_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            meta={"source": "flask-starter"},
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv

# Replace this with your real chatbot backend call

def generate_reply(
        user_text: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        kvkk_ok: Optional[bool] = None,   # ✅ EKLENDİ
    ) -> str:
    """Proxy the user_text to Saily upstream API and return assistant reply text.
    Falls back gracefully if the response shape differs.
    """
    headers = {
        "Content-Type": "application/json",
    }
    # If your API expects Authorization, add it
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    payload: Dict[str, Any] = {
        "message": user_text,
    }
   
    # ✅ KVKK bayrağını da upstream’e taşı
    if kvkk_ok is not None:
        payload["kvkk_ok"] = kvkk_ok
    # Common optional fields that Saily might accept
    if session_id:
        payload["session_id"] = session_id
    if user_id:
        payload["user_id"] = user_id
    if request_id:
        payload["request_id"] = request_id
    print("[UPSTREAM] sending ->", payload) 
    try:
        resp = requests.post(UPSTREAM_URL, json=payload, headers=headers, timeout=UPSTREAM_TIMEOUT_S)
        print("[UPSTREAM] status:", resp.status_code)
        print("[UPSTREAM] body:", (resp.text[:300] if resp.text else "EMPTY"))
    except requests.RequestException as e:
        # Surface a friendly message; also you can log e to a file/monitoring
        return f"Üzgünüm, servis geçici olarak yanıt vermiyor. (network error)"

    if resp.status_code != 200:
        return f"Üzgünüm, servis hata döndü (HTTP {resp.status_code})."

    # Try multiple common shapes
    try:
        data = resp.json()
    except ValueError:
        # Not JSON – return raw text
        return resp.text.strip() or "(boş cevap)"

    # Most likely keys
    for key in ("reply", "text", "message", "output"):
        if key in data and isinstance(data[key], str):
            return data[key]

    # Nested variants
    if "data" in data and isinstance(data["data"], dict):
        for key in ("reply", "text", "message", "output"):
            if key in data["data"] and isinstance(data["data"][key], str):
                return data["data"][key]

    # If nothing matches, return a compact pretty string for debugging
    return str(data)[:2000]  # avoid logging huge payloads in responses

# Core logging helper

def _log_message(
    db,
    conv: Conversation,
    role: str,
    content: str,
    *,
    request_id: Optional[str] = None,
    latency_ms: Optional[int] = None,
    token_in: Optional[int] = None,
    token_out: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Message:
    m = Message(
        conversation=conv,
        role=role,
        content=content,
        request_id=request_id,
        latency_ms=latency_ms,
        token_input=token_in,
        token_output=token_out,
        meta=meta or {},
    )
    conv.last_activity_at = datetime.utcnow()
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

from sqlalchemy.orm import Session

def save_kvkk_consent(
    db: Session,
    *,
    conversation: Optional[Conversation],
    session_id: Optional[str],
    phone: Optional[str],
    name: Optional[str],
    consent_given: bool,
    channel: str,
    ip_address: Optional[str],
    user_agent: Optional[str],
    consent_text_version: Optional[str] = None,
    consent_text_hash: Optional[str] = None,
):
    """
    KVKK onay kaydı oluşturur.
    """
    consent = KvkkConsent(
        conversation=conversation,
        session_id=session_id,
        phone=phone,
        name=name,
        consent_given=consent_given,
        channel=channel,
        ip_address=ip_address,
        user_agent=user_agent,
        consent_text_version=consent_text_version,
        consent_text_hash=consent_text_hash,
    )
    db.add(consent)
    db.commit()
    db.refresh(consent)
    return consent


INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

@app.before_request
def require_internal_key():
    # CORS preflight serbest
    if request.method == "OPTIONS":
        return

    p = request.path or ""

    # Guard dışı bırakılacak yollar (statikler + sağlık)
    if p in ("/health", "/index.html", "/widget.html", "/"):
        return
    if p.startswith("/assets/"):
        return
    # Eğer tüm klasörü statik veriyorsan (static_url_path=""):
    if p.endswith((".html", ".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".ico")):
        return

    # Sadece API uçlarını koru (ör: /chat ve ileride ekleyeceklerin)
    if p.startswith("/chat"):
        key = request.headers.get("X-Internal-Key")
        if INTERNAL_API_KEY and key != INTERNAL_API_KEY:
            return jsonify({"error": "unauthorized"}), 401
@app.get("/")
def index_info():
    return {"ok": True, "use": "GET /health, GET /index.html, GET /widget.html, POST /chat"}

@app.get("/index.html")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/widget.html")
def serve_widget():
    return send_from_directory(BASE_DIR, "widget.html")

@app.get("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)


@app.post("/chat")
def chat():
    db = SessionLocal()
    try:
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or request.headers.get("X-Session-Id")
        user_id = payload.get("user_id") or request.headers.get("X-User-Id")
        message = payload.get("message")
        request_id = payload.get("request_id") or str(uuid.uuid4())
        # ✅ KVKK bayrağını al
        kvkk_ok = payload.get("kvkk_ok", False)

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not message:
            return jsonify({"error": "message is required"}), 400

        conv = _get_or_create_conversation(db, session_id=session_id, user_id=user_id)
        
        # İstemcinin IP ve User-Agent bilgisi
        ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
        user_agent = request.headers.get("User-Agent")
        
        # KVKK onayı geldiyse ve daha önce bu session için onay yoksa kaydet
        if kvkk_ok:
            existing = (
                db.query(KvkkConsent)
                .filter(
                    KvkkConsent.session_id == session_id,
                    KvkkConsent.consent_given == True,
                )
                .first()
            )
            if existing is None:
                save_kvkk_consent(
                    db,
                    conversation=conv,
                    session_id=session_id,
                    phone=None,          # Şimdilik yok; ileride widget’tan alabiliriz
                    name=None,           # Şimdilik yok; ileride eklenebilir
                    consent_given=True,
                    channel="web_widget",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    consent_text_version="v1.0",
                    consent_text_hash=None,  # İstersen KVKK metninin hash’ini koyarsın
                )


        # Log user message
        _log_message(db, conv, role="user", content=message, request_id=request_id)

        # Generate reply and measure latency
        t0 = time.perf_counter()
        reply_text = generate_reply(message, session_id=session_id, user_id=user_id, request_id=request_id, kvkk_ok=kvkk_ok, )
        latency = int((time.perf_counter() - t0) * 1000)

        # Log assistant reply
        msg_assistant = _log_message(db, conv, role="assistant", content=reply_text, request_id=request_id, latency_ms=latency)

        return jsonify({
            "conversation_id": conv.conversation_id,
            "reply": reply_text,
            "assistant_message_id": msg_assistant.id,
            "latency_ms": latency,
        })
    finally:
        db.close()

@app.get("/conversations")
def list_conversations():
    db = SessionLocal()
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        q = db.query(Conversation).order_by(Conversation.last_activity_at.desc()).limit(limit)
        return jsonify([c.to_dict() for c in q])
    finally:
        db.close()

@app.get("/conversations/<conversation_id>/messages")
def get_conversation_messages(conversation_id: str):
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
        if not conv:
            return jsonify({"error": "conversation not found"}), 404
        msgs = (
            db.query(Message)
            .filter(Message.conversation_id_fk == conv.id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return jsonify({
            "conversation": conv.to_dict(),
            "messages": [m.to_dict() for m in msgs],
        })
    finally:
        db.close()

@app.get("/sessions/<session_id>/messages")
def get_session_messages(session_id: str):
    db = SessionLocal()
    try:
        limit = min(int(request.args.get("limit", 200)), 1000)
        convs = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.started_at.asc())
            .all()
        )
        data = []
        for conv in convs:
            msgs = (
                db.query(Message)
                .filter(Message.conversation_id_fk == conv.id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )
            data.append({
                "conversation": conv.to_dict(),
                "messages": [m.to_dict() for m in msgs],
            })
        if not data:
            return jsonify({"error": "no conversations for session"}), 404
        return jsonify(data)
    finally:
        db.close()

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat() + "Z"})

if __name__ == "__main__":
    # Run: FLASK_DEBUG=1 python app.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5500)), debug=APP_DEBUG)
