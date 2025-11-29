# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 16:47:54 2025

@author: Serkan
"""

import logging, os, platform
import inspect, traceback
# --- Safe logging (UTF-8) ---

def _user_appdata():
    home = os.path.expanduser("~")
    return os.path.join(home, "AppData", "Local", "OdyoduyuChatbot") if platform.system() == "Windows" \
           else os.path.join(home, ".odyoduyu_chatbot")

# Kökler
APPDATA_DIR = _user_appdata()
LOGFILE       = os.path.join(APPDATA_DIR, "app.log")

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
        needs_trace = msg.startswith("[RANK]") and " rid=" not in msg

        # normal yazma + gerekirse iz bırak
        with open(LOGFILE, "a", encoding="utf-8", errors="replace") as f:
            if needs_trace:
                fr = inspect.stack()[1]  # log'u çağıran frame
                where = f"{os.path.basename(fr.filename)}:{fr.lineno} in {fr.function}"
                f.write(msg + f"  <-- NO RID @ {where}\n")
                f.write("[TRACE]\n" + "".join(traceback.format_stack(limit=6)) + "\n")
            else:
                f.write(msg + "\n")
    except Exception:
        pass