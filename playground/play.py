"""Try vindex on a public dataset, from the command line or a browser page.

    .venv/bin/python playground/play.py list                  # the datasets, and what is done on each
    .venv/bin/python playground/play.py run NAME              # download, build, search: all three
    .venv/bin/python playground/play.py get NAME              # or one at a time
    .venv/bin/python playground/play.py build NAME
    .venv/bin/python playground/play.py search NAME
    .venv/bin/python playground/play.py show NAME             # what the last build and search found
    .venv/bin/python playground/play.py ui                    # the same in a browser: http://127.0.0.1:8035/

See playground/README.md.
"""
import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I leaves the script's folder out
import lab  # noqa: E402

# ---------- the command line ----------


def progress_line(p, started):
    """One line: time so far, the part of the job, its step, and how far it has got."""
    parts = [f"{time.time() - started:6.0f}s", p.get("phase") or "starting"]
    if p.get("step"):
        parts.append(p["step"])
    line = "  ".join(parts[:2]) + ("  |  " + parts[2] if len(parts) > 2 else "")
    if p.get("total"):
        if p.get("phase") == "downloading":
            line += f"  {p['done'] / 1e6:,.0f} of {p['total'] / 1e6:,.0f} MB"
        line += f"  {100 * p['done'] / p['total']:3.0f}%"
    return line


def printer():
    """A `report` for lab's jobs: one line, rewritten in place in a terminal, a new line per step otherwise."""
    started, last = time.time(), [None, ""]
    tty = sys.stdout.isatty()

    def report(p):
        line = progress_line(p, started)
        key = (p.get("phase"), p.get("step"))
        if tty:
            sys.stdout.write("\r" + line[:110].ljust(len(last[1])))
            last[1] = line[:110]
        elif key != last[0]:
            print(line, flush=True)
        last[0] = key
        sys.stdout.flush()

    def end():
        if tty and last[1]:
            sys.stdout.write("\n")
    return report, end


def run_job(fn, *a, **kw):
    report, end = printer()
    try:
        return fn(*a, report=report, **kw)
    finally:
        end()


def show_list():
    print(f"{'dataset':36} {'kind':40} {'items':>11} {'dims':>5} {'GB':>6}  {'notes':17} done")
    for d in lab.datasets():
        done = ", ".join(k for k in ("downloaded", "built", "searched") if d[k])
        notes = " ".join(t for t, on in (("in-report", d["in_report"]), ("unlike", d["unlike"])) if on)
        items = f"{d['items']:,}" if d["items"] else "?"
        print(f"{d['name']:36} {d['kind']:40} {items:>11} {d['dims'] or '?':>5} {d['gb']:6.2f}  {notes:17} {done}")
    print("\nin-report: one of the 13 datasets in REPORT.md.  unlike: questions look unlike the items (e.g. text searching photos).")


def show_build(b):
    m = b["memory"]
    print(f"\nbuilt {b['items']:,} items of {b['dims']} numbers ({b['metric']}) in {b['build_seconds']:,.0f} s on {b['threads']}")
    print(f"  sample questions: {b['sample_questions']};  test questions: {b['test_questions']}")
    print(f"  memory: {m['total_bytes'] / 1e6:,.0f} MB ({m['total_bytes'] / b['items']:.0f} bytes per item); "
          f"full numbers {m['full_numbers_on_ssd_bytes'] / 1e6:,.0f} MB can stay on SSD")
    if b.get("chosen"):
        c = b["chosen"]
        print(f"  settings picked: {c['chosen_r']} links per item, slack {c['chosen_alpha']:.1f}, "
              f"{c['chosen_code_bits']}-bit copies, re-check {c['chosen_rerank']}")
    s = b["stats"]
    ratio = f"{s['ood_ratio']:.2f}" if s.get("ood_ratio") is not None else "not measured"
    print(f"  questions unlike items: ratio {ratio}, question links {'on' if s.get('question_links') else 'off'};"
          f"  items no search can reach: {s.get('unreachable')}")


def show_search(r):
    print(f"\n{r['questions']:,} test questions, {r['threads']} thread(s), true top {lab.K} from the file or by checking every item")
    print(f"  {'effort':16} {'found':>8} {'questions/s':>12} {'checks':>9}")
    for row in r["runs"]:
        how = f"target {row['target']}" if "target" in row else f"list_size {row['list_size']}"
        print(f"  {how:16} {row['found']:8.1%} {row['questions_per_s']:12,} {row['checks']:9,.0f}")


def cli(a):
    if a.cmd == "list":
        return show_list()
    if a.cmd == "show":
        i = lab.info(a.name)
        if not i["build"]:
            sys.exit(f"{a.name} is not built yet")
        show_build(i["build"])
        if i["search"]:
            show_search(i["search"])
        return
    if a.cmd in ("get", "run"):
        r = run_job(lab.download, a.name)
        print("already downloaded:" if r.get("already_there") else "downloaded:", r["file"])
    if a.cmd in ("build", "run"):
        show_build(run_job(lab.build, a.name, threads=a.threads, sample_questions=a.sample_questions, metric=a.metric))
    if a.cmd in ("search", "run"):
        targets = a.target if a.target is not None else ([] if a.list_size else [0.90, 0.95, 0.99])
        show_search(run_job(lab.search, a.name, targets=targets, list_sizes=a.list_size or [], threads=a.search_threads))


# ---------- the browser page ----------


class Job:
    """The one job running for the page (a download, build or search), and how far it has got."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = None

    def start(self, kind, name, fn, **kw):
        with self.lock:
            if self.now and self.now["state"] == "running":
                raise RuntimeError(f"busy: {self.now['kind']} {self.now['name']} is running")
            self.now = {"kind": kind, "name": name, "state": "running", "started": time.time(),
                        "progress": {}, "result": None, "error": None}
        job = self.now

        def report(p):
            job["progress"] = dict(p)

        def run():
            try:
                job["result"] = fn(name, report=report, **kw)
                job["state"] = "done"
            except Exception as e:   # shown on the page
                traceback.print_exc()
                job["error"] = str(e)
                job["state"] = "failed"
            job["ended"] = time.time()

        threading.Thread(target=run, daemon=True).start()

    def view(self):
        j = self.now
        if not j:
            return None
        return {**j, "elapsed": (j.get("ended") or time.time()) - j["started"]}


def handler(job):
    page = (Path(__file__).resolve().parent / "ui.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, kind="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            a = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, page, "text/html; charset=utf-8")
                if u.path == "/favicon.ico":   # no icon; answered so the browser logs no error
                    return self.send(204, b"", "image/x-icon")
                if u.path == "/api/datasets":
                    return self.send(200, lab.datasets())
                if u.path == "/api/job":
                    return self.send(200, job.view())
                if u.path == "/api/info":
                    return self.send(200, lab.info(a.get("name", "")))
                if u.path == "/api/question":
                    return self.send(200, lab.question(a.get("name", ""), int(a.get("i", 0)), float(a.get("target", 0.95))))
                return self.send(404, {"error": "not found"})
            except (ValueError, FileNotFoundError) as e:
                return self.send(400, {"error": str(e)})

        def do_POST(self):
            # A page on another site cannot send this header to us without our leave, so it cannot start a 60 GB download.
            if self.headers.get("X-Playground") != "1":
                return self.send(403, {"error": "missing X-Playground header"})
            u = urlparse(self.path)
            try:
                size = int(self.headers.get("Content-Length") or 0)
                if size > 10_000:
                    raise ValueError("request too big")
                body = json.loads(self.rfile.read(size) or b"{}")
                name = lab.check_name(body.get("name"))
                if u.path == "/api/download":
                    job.start("download", name, lab.download)
                elif u.path == "/api/build":
                    threads = int(body["threads"]) if body.get("threads") else None
                    job.start("build", name, lab.build, threads=threads)
                elif u.path == "/api/search":
                    targets = [float(t) for t in body.get("targets", [0.90, 0.95, 0.99])][:8]
                    sizes = [int(s) for s in body.get("list_sizes", [])][:8]
                    job.start("search", name, lab.search, targets=targets, list_sizes=sizes,
                              threads=int(body.get("threads") or 1))
                else:
                    return self.send(404, {"error": "not found"})
                return self.send(200, job.view())
            except RuntimeError as e:
                return self.send(409, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    return H


def ui(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler(Job()))
    print(f"open http://127.0.0.1:{port}/   (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="Try vindex on a public dataset.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the datasets, and what is done on each")
    for cmd, what in (("get", "download a dataset"), ("build", "build and save an index"),
                      ("search", "search the test questions and check the answers"),
                      ("run", "get, build and search"), ("show", "what the last build and search found")):
        p = sub.add_parser(cmd, help=what)
        p.add_argument("name", type=lab.check_name, help="a name from 'list', or your own file's name in playground/data/")
        if cmd in ("build", "run"):
            p.add_argument("--threads", type=int, help="cores for the build (default: all)")
            p.add_argument("--sample-questions", type=int, default=lab.SAMPLE_QUESTIONS,
                           help=f"sample questions for the build (default {lab.SAMPLE_QUESTIONS:,})")
            p.add_argument("--metric", choices=("cos", "ip", "l2"), help="only for a file with no 'distance'")
        if cmd in ("search", "run"):
            p.add_argument("--target", type=float, nargs="+", help="accuracy targets (default 0.90 0.95 0.99)")
            p.add_argument("--list-size", type=int, nargs="+", help="fixed efforts instead of (or with) targets")
            p.add_argument("--search-threads", type=int, default=1, help="cores for searching (default 1)")
    p = sub.add_parser("ui", help="the browser page")
    p.add_argument("--port", type=int, default=8035)
    a = ap.parse_args()
    if a.cmd == "ui":
        return ui(a.port)
    try:
        cli(a)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
