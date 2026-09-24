"""Dry-run accuracy check of the voice router against a running examples/voice/laya_server.py.
Nothing is executed (dry=1). Utterances are written the way Superwhisper
transcribes speech (capitalised, trailing period).

Run: .venv/bin/python examples/voice/voice_eval.py
"""
import json
import urllib.parse
import urllib.request
from typing import Any, cast
CASES: list[tuple[str, str, str | None]] = [
    ("Open Safari.", "open_app", "Safari"),
    ("Launch Slack.", "open_app", "Slack"),
    ("Switch to Visual Studio Code.", "open_app", "Visual Studio Code"),
    ("Bring up the calculator.", "open_app", "calculator"),
    ("Open Spotify please.", "open_app", "Spotify"),
    ("Search for the best ramen in Toronto.", "web_search", "the best ramen in Toronto"),
    ("Google how tall is the CN Tower.", "web_search", "how tall is the CN Tower"),
    ("Look up the weather tomorrow.", "web_search", "the weather tomorrow"),
    ("What's the capital of Australia?", "web_search", None),
    ("Remind me to call mom at five.", "new_reminder", "call mom at five"),
    ("New reminder buy milk.", "new_reminder", "buy milk"),
    ("Add pay rent to my to-do list.", "new_reminder", "pay rent"),
    ("Take a note that the meeting moved to Thursday.", "new_note", "the meeting moved to Thursday"),
    ("Jot down idea for a voice controlled Laya demo.", "new_note", "idea for a voice controlled Laya demo"),
    ("Pause the music.", "play_pause", None),
    ("Play some music.", "play_pause", None),
    ("Resume playback.", "play_pause", None),
    ("Turn it up.", "volume_up", None),
    ("Make it louder.", "volume_up", None),
    ("Turn the volume down.", "volume_down", None),
    ("Mute the sound.", "volume_down", None),
    ("Lock the screen.", "lock_screen", None),
    ("Lock my computer.", "lock_screen", None),
    ("Take a screenshot.", "screenshot", None),
    ("Grab a screen capture.", "screenshot", None),
    ("I had a really nice lunch today.", "none", None),
    ("Um, never mind.", "none", None),
    ("The quarterly numbers look good.", "none", None),
]



# Written AFTER the descriptions/threshold were tuned on CASES; never used for tuning the
# model-side choice. NOTE: the argument-strip regexes WERE adjusted after seeing these
# (Finder/Chrome/passport), so the argument score here is not held-out.
HELDOUT: list[tuple[str, str, str | None]] = [
    ("Open Messages.", "open_app", "Messages"),
    ("Can you launch Finder.", "open_app", "Finder"),
    ("Go to Terminal.", "open_app", "Terminal"),
    ("Start Zoom.", "open_app", "Zoom"),
    ("Pull up Google Chrome.", "open_app", "Google Chrome"),
    ("Search the web for flights to Lisbon in March.", "web_search", "flights to Lisbon in March"),
    ("Look up how to repot a monstera.", "web_search", "how to repot a monstera"),
    ("Who won the World Series in 2019?", "web_search", None),
    ("Find me reviews of the Framework laptop.", "web_search", "reviews of the Framework laptop"),
    ("Remind me to email the landlord tomorrow.", "new_reminder", "email the landlord tomorrow"),
    ("Add a reminder to book the dentist.", "new_reminder", "book the dentist"),
    ("Put renew passport on my to-do list.", "new_reminder", "renew passport"),
    ("Make a note that the wifi password is on the fridge.", "new_note", "the wifi password is on the fridge"),
    ("Write down that Sam prefers Tuesdays.", "new_note", "that Sam prefers Tuesdays"),
    ("Stop the music.", "play_pause", None),
    ("Unpause.", "play_pause", None),
    ("Pause.", "play_pause", None),
    ("Volume up.", "volume_up", None),
    ("Crank it up.", "volume_up", None),
    ("Quieter please.", "volume_down", None),
    ("Lower the volume.", "volume_down", None),
    ("Lock the Mac.", "lock_screen", None),
    ("Lock it.", "lock_screen", None),
    ("Screenshot.", "screenshot", None),
    ("Capture my screen.", "screenshot", None),
    ("Snap a picture of the screen.", "screenshot", None),
    ("Okay that's it.", "none", None),
    ("My cat is sleeping on the keyboard.", "none", None),
    ("Hmm, let me think.", "none", None),
    ("Thanks.", "none", None),
]
def post(text: str) -> dict[str, Any]:
    data = urllib.parse.urlencode({"text": text, "dry": "1"}).encode()
    with urllib.request.urlopen("http://127.0.0.1:8765/voice", data, timeout=30) as r:
        return cast(dict[str, Any], json.loads(r.read()))


def evaluate(cases: list[tuple[str, str, str | None]], label: str) -> None:
    ok = arg_ok = arg_n = 0
    ms: list[float] = []
    print("\n== %s (%d cases)" % (label, len(cases)))
    for text, want, want_arg in cases:
        r = post(text)
        hit = r["action"] == want
        ok += hit
        ms.append(r["decide_ms"])
        note = ""
        if want_arg is not None and hit:
            arg_n += 1
            arg_ok += r["argument"].lower() == want_arg.lower()
            if r["argument"].lower() != want_arg.lower():
                note = "  arg=%r (want %r)" % (r["argument"], want_arg)
        print("%s %-50s -> %-13s p=%.2f  (want %-12s)%s"
              % ("OK " if hit else "BAD", text[:50], r["action"], r["probability"], want, note))
    ms.sort()
    print("%s: action accuracy %d/%d | argument accuracy %d/%d | decide p50 %.1f ms max %.1f ms"
          % (label, ok, len(cases), arg_ok, arg_n, ms[len(ms) // 2], ms[-1]))


if __name__ == "__main__":
    evaluate(CASES, "tuning set")
    evaluate(HELDOUT, "held-out set")
