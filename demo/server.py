"""Step 34: the GooAQ demo's server (part 4). After demo/prepare.py:
    demo/.venv/bin/python -I demo/server.py [--port 8034]
then open http://127.0.0.1:8034/ . It holds the model that made GooAQ's numbers, our index of the full GooAQ items
(demo/work/vindex) and HNSW on the same items (demo/work/hnsw.bin).
GET /search?q=<question> turns the question into numbers, then searches both indexes on one core, each 25 times
(the time shown is the middle one). The reply gives each index's 10 closest items with their answer texts.
It listens on this machine only: GooAQ's text is not for commercial use.
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # -I leaves the script folder out
from prepare import MODEL, ROOT, WORK, model   # noqa: E402

RUNS = 25


class Demo:
    def __init__(self):
        import hnswlib
        import vindex
        self.model = model()
        self.idx = vindex.Index.load(WORK / "vindex")
        h = json.loads((WORK / "hnsw.json").read_text())
        self.hnsw = hnswlib.Index(space="ip", dim=768)
        self.hnsw.load_index(str(WORK / "hnsw.bin"))
        self.hnsw.set_ef(max(int(h["effort"]), 10))
        self.hnsw.set_num_threads(1)
        self.hnsw_settings = h
        s29 = json.loads((ROOT / "validate" / "results" / "step29.json").read_text())
        self.target = next(c["data"]["ours"]["target"] for c in s29["checks"]
                           if c["check"].startswith("gooaq: at full size"))   # the target the index picked (step 29)
        self.item_answer = np.load(WORK / "item_answer.npy", mmap_mode="r")
        self.blob = np.memmap(WORK / "answers.bin", dtype=np.uint8, mode="r")
        self.at = np.load(WORK / "answers_at.npy", mmap_mode="r")
        self.lock = threading.Lock()
        self.info = {"items": int(len(self.item_answer)), "with_text": int((np.asarray(self.item_answer) >= 0).sum()),
                     "dim": 768, "model": MODEL, "vindex_target": self.target, "hnsw": h}
        self.numbers("warm up")

    def text(self, item):
        a = int(self.item_answer[item])
        return None if a < 0 else bytes(self.blob[self.at[a]:self.at[a + 1]]).decode("utf-8")

    def numbers(self, q):
        return self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def search(self, q, numbers=False):
        with self.lock:
            t = time.perf_counter()
            v = self.numbers(q)
            embed = time.perf_counter() - t
            out = {"q": q, "embed_ms": embed * 1000}
            for name, run in (("vindex", lambda: self.idx.search(v, k=10, target=self.target, threads=1)[:2]),
                              ("hnsw", lambda: self.hnsw.knn_query(v, k=10, num_threads=1))):
                times = []
                for _ in range(RUNS):
                    t = time.perf_counter()
                    ids, d = run()
                    times.append(time.perf_counter() - t)
                out[name] = {"ms": float(np.median(times)) * 1000, "results": [
                    {"item": int(i), "similarity": float(1 - x), "text": self.text(int(i))}
                    for i, x in zip(np.asarray(ids)[0], np.asarray(d)[0]) if i >= 0]}
            out["settings"] = {"vindex_target": self.target, "hnsw": self.hnsw_settings}
            if numbers:   # the question's numbers, for validate/step34.py
                out["numbers"] = v[0].tolist()
            return out


def handler(demo):
    page = (HERE / "index.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self.send(200, page, "text/html; charset=utf-8")
            if u.path == "/health":
                return self.send(200, b"ok", "text/plain")
            if u.path == "/favicon.ico":   # no icon; answered so the browser logs no error
                return self.send(204, b"", "image/x-icon")
            if u.path == "/info":
                return self.send(200, json.dumps(demo.info).encode(), "application/json")
            if u.path == "/search":
                a = parse_qs(u.query)
                q = (a.get("q") or [""])[0].strip()[:500]
                if not q:
                    return self.send(400, b'{"error": "type a question"}', "application/json")
                return self.send(200, json.dumps(demo.search(q, a.get("numbers") == ["1"])).encode(), "application/json")
            self.send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8034)
    a = ap.parse_args()
    demo = Demo()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(demo))
    print(f"ready on http://127.0.0.1:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
