# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 19:36:15 2025

@author: Serkan
"""

# web_ingest.py
import os, re, time, json, urllib.parse, requests

from  common_log import log

# ---- URL ingest yardımcıları ----
UA = "SailyAI-RAG/1.0 (+https://sailead.com.tr/iletisim; contact: destek@sailead.com.tr)"

def _safe_name(url: str) -> str:
    u = urllib.parse.urlparse(url)
    host = (u.netloc or "site").replace(":", "_")
    path = (u.path or "/").strip("/").replace("/", "_") or "index"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{host}_{path}")[:120]
    # pdf ise pdf, değilse txt
    return base + (".pdf" if (u.path or "").lower().endswith(".pdf") else ".txt")

def fetch_and_save(url: str, out_dir: str) -> str | None:
    os.makedirs(out_dir, exist_ok=True)
    cache_json = os.path.join(out_dir, "_url_cache.json")

    cache = {}
    if os.path.exists(cache_json):
        try:
            cache = json.load(open(cache_json, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    hdrs = {"User-Agent": UA}
    if url in cache:
        if cache[url].get("etag"):    hdrs["If-None-Match"] = cache[url]["etag"]
        if cache[url].get("lastmod"): hdrs["If-Modified-Since"] = cache[url]["lastmod"]

    r = requests.get(url, headers=hdrs, timeout=25)
    if r.status_code == 304:
        log("[INGEST] not-modified:", url)
        return os.path.join(out_dir, _safe_name(url))
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        path = os.path.join(out_dir, _safe_name(url))
        open(path, "wb").write(r.content)
        log("[INGEST] saved PDF:", url, "->", os.path.basename(path))
        cache[url] = {"etag": r.headers.get("ETag"), "lastmod": r.headers.get("Last-Modified"), "ts": int(time.time())}
        json.dump(cache, open(cache_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        time.sleep(0.6)
        return path

    text = r.text or ""
    if "<html" in text.lower():
        try:
            from trafilatura import extract
            try:
                parsed = extract(text, url=url, output_format="txt", favor_precision=True)
            except TypeError:
                parsed = extract(text, url=url)  # eski sürüm uyumu
            if parsed:
                text = parsed
        except Exception:
            pass

    cleaned = (text or "").strip()
    if not cleaned:
        log("[INGEST] empty after parse:", url)
        return None

    tagged = f"[URL:{url}]\n\n{cleaned}\n"
    path = os.path.join(out_dir, _safe_name(url))
    open(path, "w", encoding="utf-8").write(tagged)

    cache[url] = {"etag": r.headers.get("ETag"), "lastmod": r.headers.get("Last-Modified"), "ts": int(time.time())}
    json.dump(cache, open(cache_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    log("[INGEST] saved:", url, "->", os.path.basename(path))
    time.sleep(0.6)
    return path

def fetch_from_urls_file(urls_file: str, out_dir: str) -> list[str]:
    paths = []
    if not os.path.exists(urls_file):
        log("[INGEST] urls_file not found:", urls_file)
        return paths
    for line in open(urls_file, "r", encoding="utf-8"):
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        try:
            p = fetch_and_save(url, out_dir=out_dir)
            if p:
                paths.append(p)
        except Exception as e:
            log("[INGEST] failed:", url, repr(e))
    log(f"[INGEST] total_saved={len(paths)} to {out_dir}")
    return paths
# ---- /URL ingest yardımcıları ----
