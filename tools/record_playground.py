"""Record playground/playground.gif: the browser page downloading, building and searching yi-128-ip, with the waits
sped up (the page's own clock shows the real time).

Needs Google Chrome, and Playwright and Pillow in a Python of their own (not the package's .venv):
    uv venv /tmp/rec --python 3.13 && uv pip install --python /tmp/rec playwright pillow
    /tmp/rec/bin/python tools/record_playground.py
It starts the page's server on port 8036 with .venv's Python. yi-128-ip must not be downloaded or built yet, so the
recording shows every step: move playground/data/yi-128-ip.hdf5 and playground/work/yi-128-ip/ away first.
"""
import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "playground" / "playground.gif"
PORT = 8036
NAME = "yi-128-ip"
WIDTH, HEIGHT, SCALE = 1440, 860, 0.7
WAIT_EVERY_S, WAIT_FRAME_MS = 1.0, 140   # while a job runs: one frame a second, shown for 140 ms (about 7x faster)

CURSOR = """
const c = document.createElement('div'); c.id = 'rec-cursor';
c.style.cssText = 'position:fixed;width:22px;height:22px;border-radius:50%;border:3px solid #d9480f;' +
  'background:rgba(217,72,15,.18);pointer-events:none;z-index:9999;transition:all .35s;left:640px;top:400px;margin:-11px';
document.body.append(c);
const n = document.createElement('div'); n.id = 'rec-note';
n.style.cssText = 'position:fixed;right:16px;bottom:14px;padding:4px 10px;border-radius:6px;background:#1d1f21;' +
  'color:#fff;font:600 13px -apple-system,sans-serif;z-index:9999;display:none';
document.body.append(n);
"""


def job():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/job") as r:
        return json.load(r)


def main():
    if (ROOT / "playground" / "data" / f"{NAME}.hdf5").exists() or (ROOT / "playground" / "work" / NAME).exists():
        raise SystemExit(f"{NAME} is already downloaded or built; move it away first so the recording shows each step")
    server = subprocess.Popen([str(ROOT / ".venv" / "bin" / "python"), "-I", str(ROOT / "playground" / "play.py"),
                               "ui", "--port", str(PORT)], stdout=subprocess.DEVNULL)
    frames = []
    try:
        time.sleep(2)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="light")
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.wait_for_selector("#datasets tbody tr")
            page.evaluate(CURSOR)

            def snap(ms):
                img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                frames.append((img.resize((int(WIDTH * SCALE), int(HEIGHT * SCALE)), Image.LANCZOS), ms))

            def note(text):
                page.evaluate("t => { const n = document.getElementById('rec-note'); n.textContent = t; "
                              "n.style.display = t ? 'block' : 'none'; }", text)

            def click(selector, hold_ms=900):
                el = page.locator(selector).first
                el.scroll_into_view_if_needed()
                b = el.bounding_box()
                page.evaluate("([x, y]) => { const c = document.getElementById('rec-cursor'); c.style.display = 'block'; "
                              "c.style.left = x + 'px'; c.style.top = y + 'px'; }", [b["x"] + b["width"] / 2, b["y"] + b["height"] / 2])
                time.sleep(0.4)
                snap(500)
                el.click()
                time.sleep(0.3)
                snap(hold_ms)

            def wait_job():
                note("waiting: sped up about 7x")
                while True:
                    time.sleep(WAIT_EVERY_S)
                    j = job()
                    snap(WAIT_FRAME_MS)
                    if j and j["state"] != "running":
                        break
                if j["state"] != "done":
                    raise SystemExit(f"the {j['kind']} failed: {j['error']}")
                time.sleep(1.2)   # the page refreshes its cards after the job ends
                note("")

            def show(selector, hold_ms):
                page.evaluate("s => { document.getElementById('rec-cursor').style.display = 'none'; "
                              "document.querySelector(s).scrollIntoView({block: 'start'}); }", selector)
                time.sleep(0.3)
                snap(hold_ms)

            snap(1800)
            click(f"text={NAME}", 1200)
            click("#b-get")
            wait_job()
            snap(1200)
            click("#b-build")
            wait_job()
            show("#job-card", 2500)
            show("#build-card", 3500)
            page.evaluate("window.scrollTo(0, 0)")
            click("#b-search")
            wait_job()
            show("#search-card", 4000)
            click("#b-q", 600)
            show("#q-card", 4500)
            browser.close()
    finally:
        server.terminate()
    images = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:], duration=[ms for _, ms in frames], loop=0,
                   optimize=True)
    print(f"wrote {OUT}: {len(frames)} frames, {sum(ms for _, ms in frames) / 1000:.0f} s, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
