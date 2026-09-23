"""Local Laya server: keeps LayaFast loaded and turns spoken commands into Mac actions.

Endpoints (127.0.0.1 only):
  GET  /health                     -> {"loaded": bool, ...}
  POST /voice    text=<transcript> -> Laya picks an action from the voice_commands.json
                 (form-encoded or JSON {"text": ...}; add dry=1 to decide without running)
  POST /v1/systemone  {"state": ..., "questions": {...}}  -> raw Laya answers (Jev shape)

The model loads at start (LAYA_PRELOAD=1) and unloads after LAYA_IDLE_UNLOAD_S seconds
idle (default 1200) so a resident daemon doesn't hold ~1.5 GB forever; the next command
reloads it (~2 s).

Run:  .venv/bin/python examples/voice/laya_server.py   (port 8765, env LAYA_PORT)
"""
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Repo root holds the runtime modules (laya_fast, laya_api) and the default model.
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PORT = int(os.environ.get("LAYA_PORT", "8765"))
MODEL_DIR = os.environ.get("LAYA_MODEL", str(ROOT / "converted-fp16"))
COMMANDS = Path(os.environ.get("LAYA_COMMANDS", str(HERE / "voice_commands.json")))
IDLE_UNLOAD_S = float(os.environ.get("LAYA_IDLE_UNLOAD_S", "1200"))
# Short on purpose: prompt + options must stay <=128 tokens to run on the Neural
# Engine (~13 ms); a long phrasing measured ~23 ms and was no more accurate.
INSTRUCTIONS = "Which computer action does this spoken command ask for?"
NONE = {"kind": "none"}

_lock = threading.Lock()
_agent = None
_last_used = 0.0


def log(msg):
    print("%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def get_agent():
    global _agent, _last_used
    _last_used = time.time()
    if _agent is None:
        t0 = time.perf_counter()
        from laya_fast import LayaFast
        _agent = LayaFast(MODEL_DIR)
        log("model loaded in %.2f s" % (time.perf_counter() - t0))
    return _agent


def _idle_unloader():
    global _agent
    while True:
        time.sleep(30)
        with _lock:
            if _agent is not None and time.time() - _last_used > IDLE_UNLOAD_S:
                _agent = None
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:  # noqa: BLE001
                    pass
                log("model unloaded after %.0f s idle" % IDLE_UNLOAD_S)


def load_commands():
    """Re-read every request so edits to voice_commands.json apply without a restart."""
    cfg = json.loads(COMMANDS.read_text())
    actions = {k: v for k, v in cfg["actions"].items() if k != "none"}
    return cfg, actions


def notify(text, title="Laya"):
    subprocess.Popen(["/usr/bin/osascript", "-e",
                      "on run argv\n display notification (item 1 of argv) with title (item 2 of argv)\nend run",
                      text, title])


def clean(text):
    return re.sub(r"[\s.!?,;:]+$", "", text.strip())

_POLITE = [r"^(hey|ok|okay|alright)[,\s]+", r"^(can|could|would|will) you (please )?", r"^please ",
           r",? please$", r",? (for me|now|thanks|thank you)$"]


def argument(action, text):
    """Action argument = transcript minus polite wrappers and the action's own
    trigger phrases. 'strip' may be one regex or a list (prefixes and suffixes)."""
    arg = clean(text)
    pats = action.get("strip") or []
    pats = [pats] if isinstance(pats, str) else pats
    for pat in _POLITE + pats + _POLITE:
        arg = clean(re.sub(pat, "", arg, count=1, flags=re.I))
    return arg or clean(text)


def run_action(name, action, arg):
    """Run one catalog action. User text is only ever passed as an argv element."""
    kind = action["kind"]
    if kind == "none":
        return "No command matched"
    if kind == "open_app":
        r = subprocess.run(["/usr/bin/open", "-a", arg], capture_output=True, text=True, timeout=10)
        return "Opening %s" % arg if r.returncode == 0 else "No app named %r" % arg
    if kind == "open_url":
        url = action["url"].replace("{arg}", urllib.parse.quote_plus(arg))
        subprocess.run(["/usr/bin/open", url], timeout=10)
        return "Searching: %s" % arg
    if kind == "shell":
        stamp = time.strftime("%Y-%m-%d at %H.%M.%S")
        argv = [os.path.expanduser(a.replace("{arg}", arg).replace("{time}", stamp))
                for a in action["argv"]]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return (name.replace("_", " ").capitalize() if r.returncode == 0
                else "Failed: %s" % r.stderr.strip())
    if kind == "applescript":
        r = subprocess.run(["/usr/bin/osascript", "-e", action["script"], arg],
                           capture_output=True, text=True, timeout=20)
        return (r.stdout.strip() or name) if r.returncode == 0 else "Failed: %s" % r.stderr.strip()
    return "Unknown action kind %r" % kind


def decide(text):
    cfg, actions = load_commands()
    q = {"action": {"type": "choice", "instructions": INSTRUCTIONS,
                    "criteria": {k: v["description"] for k, v in actions.items()}}}
    t0 = time.perf_counter()
    ans = get_agent().system_one(clean(text), q)["answers"]["action"]
    ms = (time.perf_counter() - t0) * 1e3
    name, p = ans["choice"], ans["probabilities"][ans["choice"]]
    if p < float(cfg.get("min_probability", 0.35)):
        name = "none"
    return name, p, ans["probabilities"], ms, dict(actions, none=NONE)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
        if "json" in (self.headers.get("Content-Type") or ""):
            return json.loads(raw or "{}")
        return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

    def log_message(self, fmt, *args):  # silence default access log
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "loaded": _agent is not None,
                                    "idle_unload_s": IDLE_UNLOAD_S})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
            with _lock:
                if self.path == "/voice":
                    text = (body.get("text") or "").strip()
                    if not text:
                        return self._send(400, {"error": "empty text"})
                    name, p, probs, ms, actions = decide(text)
                    arg = argument(actions[name], text)
                    dry = str(body.get("dry", "")) in ("1", "true")
                    result = "(dry run)" if dry else run_action(name, actions[name], arg)
                    log("%r -> %s p=%.2f arg=%r %.1f ms | %s" % (text, name, p, arg, ms, result))
                    if not dry:
                        notify(result if name != "none"
                               else 'Didn\'t catch a command: "%s"' % clean(text))
                    return self._send(200, {"action": name, "probability": p, "argument": arg,
                                            "result": result, "decide_ms": round(ms, 2),
                                            "probabilities": probs})
                if self.path == "/v1/systemone":
                    return self._send(200, get_agent().system_one(body["state"], body["questions"]))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            log("error: %r" % e)
            self._send(500, {"error": str(e)})


def main():
    threading.Thread(target=_idle_unloader, daemon=True).start()
    if os.environ.get("LAYA_PRELOAD", "1") == "1":
        with _lock:
            get_agent()
    log("listening on http://127.0.0.1:%d" % PORT)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
