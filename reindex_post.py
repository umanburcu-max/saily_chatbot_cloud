# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 17:42:15 2025

@author: Serkan
"""

#!/usr/bin/env python3
# reindex_post.py

import os
import sys
import time
import json
import argparse
from urllib import request as urlrequest

try:
    import requests
except Exception:
    requests = None

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("APP_PORT", "5000"))

def make_base(host: str, port: int) -> str:
    return f"http://{host}:{port}"

def wait_for_server(health_url: str, timeout: int = 20, ping_timeout: int = 2, interval: float = 0.5) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlrequest.urlopen(health_url, timeout=ping_timeout) as r:
                obj = json.loads(r.read().decode("utf-8"))
                if obj.get("ok") is True:
                    return True
        except Exception:
            time.sleep(interval)
    return False

def post_reindex(reindex_url: str, timeout: int = 600) -> None:
    if requests is None:
        raise RuntimeError("`requests` modülü yüklü değil (pip install requests).")
    r = requests.post(reindex_url, timeout=timeout)
    r.raise_for_status()
    try:
        obj = r.json()
        if isinstance(obj, dict) and obj.get("ok") is False:
            raise RuntimeError(f"/reindex returned ok=false: {obj.get('error')}")
    except ValueError:
        pass

def direct_rebuild() -> None:
    try:
        from backend import build_or_load_vectorstore  # type: ignore
    except Exception as e:
        raise RuntimeError(f"backend import edilemedi: {e!r}")
    try:
        build_or_load_vectorstore(force_rebuild=True)
    except Exception as e:
        raise RuntimeError(f"build_or_load_vectorstore(force_rebuild=True) hata: {e!r}")

def main() -> int:
    ap = argparse.ArgumentParser(description="RAG vektör indeksini yeniden kur (HTTP /reindex ya da direkt).")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Backend host (varsayılan: {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Backend port (varsayılan: {DEFAULT_PORT})")
    ap.add_argument("--health-path", default="/health", help="Health endpoint yolu (varsayılan: /health)")
    ap.add_argument("--reindex-path", default="/reindex", help="Reindex endpoint yolu (varsayılan: /reindex)")
    ap.add_argument("--wait", type=int, default=20, help="Health bekleme süresi sn (varsayılan: 20)")
    ap.add_argument("--post-timeout", type=int, default=600, help="/reindex POST timeout sn (varsayılan: 600)")
    ap.add_argument("--direct", action="store_true", help="HTTP kullanmadan doğrudan rebuild yap")
    ap.add_argument("--no-fallback", dest="no_fallback", action="store_true",
                    help="Sunucu yoksa direkt rebuild DENEME; hata ver")
    args = ap.parse_args()

    base = make_base(args.host, args.port)
    health_url = base + args.health_path
    reindex_url = base + args.reindex_path

    print(f"[INFO] Base: {base}")
    if args.direct:
        print("[INFO] --direct verildi; HTTP atlanıyor, doğrudan rebuild...")
        try:
            direct_rebuild()
            print("[OK] Direkt rebuild tamam.")
            return 0
        except Exception as e:
            print(f"[ERROR] Direkt rebuild başarısız: {e}", file=sys.stderr)
            return 3

    print(f"[INFO] Health kontrol: {health_url}")
    if wait_for_server(health_url, timeout=args.wait):
        print("[OK] Sunucu ayakta; /reindex gönderiliyor...")
        try:
            post_reindex(reindex_url, timeout=args.post_timeout)
            print("[OK] /reindex başarılı.")
            return 0
        except Exception as e:
            print(f"[ERROR] /reindex çağrısı başarısız: {e}", file=sys.stderr)
            return 1

    print("[WARN] Sunucu ayakta değil veya /health başarısız.")
    if args.no_fallback:  # <-- DÜZELTİLDİ (no-fallback -> no_fallback)
        print("[ERROR] Fallback kapalı olduğu için işlem sonlandırılıyor.", file=sys.stderr)
        return 2

    print("[INFO] Fallback: HTTP olmadan doğrudan rebuild denenecek...")
    try:
        direct_rebuild()
        print("[OK] Fallback rebuild tamam.")
        return 0
    except Exception as e:
        print(f"[ERROR] Fallback rebuild başarısız: {e}", file=sys.stderr)
        return 3

if __name__ == "__main__":
    sys.exit(main())
