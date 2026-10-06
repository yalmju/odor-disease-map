"""Local browser application. Binds only to loopback; no remote services required."""
import argparse
import csv
import json
import mimetypes
import sqlite3
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .cli import run
from .data import load_dataset

WEB = Path(__file__).with_name("web")


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class Studio:
    def __init__(self, root, library=None):
        self.root = Path(root).resolve()
        self.library = Path(library).resolve() if library else None
        self.lock = threading.Lock(); self.jobs = {}; self.datasets = {}; self.active = None
        self.root.mkdir(parents=True, exist_ok=True)
        candidates = [("dream", self.root / "data/dream.json", "DREAM · 공개 혼합물"),
                      ("demo", self.root / "examples/demo.json", "소프트웨어 예제 · 가상 거리")]
        if not candidates[1][1].exists(): candidates[1] = ("demo", WEB / "demo.json", candidates[1][2])
        for key, path, name in candidates:
            if path.exists(): self.register(key, path, name)
        for path in sorted((self.root / "data/uploads").glob("*.json")):
            try: self.register(path.stem, path, "불러온 데이터 · " + path.stem[:8])
            except (ValueError, KeyError, TypeError): pass
        for path in sorted((self.root / "runs").glob("*/report.json")):
            if path.parent.is_symlink(): continue
            report = read_json(path)
            matching = next((key for key, d in self.datasets.items() if d["sha"] == report.get("input_sha256")), None)
            self.jobs[path.parent.name] = {"id": path.parent.name, "status": "complete", "dataset": matching, "report": report}

    def register(self, key, path, name):
        import hashlib
        doc, _, audit = load_dataset(path)
        self.datasets[key] = {"id": key, "name": name, "path": path, "doc": doc, "audit": audit,
                              "sha": hashlib.sha256(path.read_bytes()).hexdigest()}

    def state(self):
        overview, hmdb, sources, user = {}, {}, [], {}
        if self.library:
            overview = read_json(self.library / "analysis/analysis_metrics.json", {})
            hmdb = read_json(self.library / "analysis/hmdb_summary.json", {})
            sources = read_json(self.library / "metadata/availability.json", [])
            user = read_json(self.library / "analysis/user_voclib_summary.json", {})
        with self.lock:
            return {"datasets": [{"id": d["id"], "name": d["name"], "molecules": len(d["doc"]["molecules"]),
                                  "mixtures": len(d["doc"]["mixtures"]), "audit": d["audit"]} for d in self.datasets.values()],
                    "runs": [dict(job) for job in self.jobs.values()], "active": self.active,
                    "library": {"connected": bool(self.library and self.library.exists()), "molecules": overview.get("registry", {}).get("unique_exact_inchikey"),
                                "hmdb": hmdb.get("records"), "hmdb_version": list(hmdb.get("versions", {})), "sources": sources, "user": user}}

    def import_dataset(self, doc):
        if not isinstance(doc, dict): raise ValueError("최상위 JSON은 객체여야 합니다.")
        key = "upload-" + uuid.uuid4().hex[:12]
        path = self.root / "data/uploads" / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        try:
            with self.lock: self.register(key, path, "불러온 데이터 · " + key[-6:])
        except Exception:
            path.unlink(); raise
        return {"id": key, "audit": self.datasets[key]["audit"]}

    def start(self, body):
        key = body["dataset"]
        if key not in self.datasets: raise ValueError("데이터셋을 선택해 주세요.")
        ridge, seed, perplexity = float(body.get("ridge", .05)), int(body.get("seed", 42)), float(body.get("perplexity", 30))
        import math
        if not math.isfinite(ridge) or not 0 < ridge <= 100 or not 0 <= seed < 2**32 or not math.isfinite(perplexity) or not 0 < perplexity <= 500:
            raise ValueError("설정 범위: ridge 0 초과~100, seed 0~2³²−1, perplexity 0 초과~500")
        if len(self.datasets[key]["doc"]["mixtures"]) > 2000:
            raise ValueError("앱 t-SNE는 최대 2,000개 혼합물을 지원합니다. 명시적으로 부분집합을 만들어 주세요.")
        with self.lock:
            if self.active: raise ValueError("이미 분석 중입니다. 완료 후 다시 실행해 주세요.")
            rid = "app-" + uuid.uuid4().hex[:12]
            self.active = rid
            self.jobs[rid] = {"id": rid, "dataset": key, "status": "running"}
        def work():
            try:
                report = run(self.datasets[key]["path"], self.root / "runs" / rid, ridge, True, seed, perplexity)
                with self.lock: self.jobs[rid].update(status="complete", report=report)
            except Exception as exc:
                with self.lock: self.jobs[rid].update(status="failed", error=str(exc))
            finally:
                with self.lock: self.active = None
        threading.Thread(target=work, daemon=True).start()
        return {"id": rid}

    def artifact(self, rid, name):
        allowed = {"report.json", "model.json", "predictions.json", "embedding.json", "embedding.npz", "embedding.png"}
        if rid not in self.jobs or name not in allowed: raise ValueError("Unknown result")
        path = (self.root / "runs" / rid / name).resolve()
        if not path.is_relative_to((self.root / "runs").resolve()): raise ValueError("Invalid result path")
        return path

    def embedding(self, rid):
        path = self.artifact(rid, "embedding.json")
        points = read_json(path, [])
        ds = self.datasets.get(self.jobs[rid].get("dataset"))
        groups = {}
        if ds:
            for pair in ds["doc"]["pairs"]:
                for mid in [pair["a"], pair["b"]]: groups.setdefault(mid, set()).add(pair["split"])
        for point in points:
            mid = point["mixture_id"]
            point["group"] = "/".join(sorted(groups.get(mid, {"unknown"})))
            point["components"] = ds["doc"]["mixtures"].get(mid, {}).get("components", []) if ds else []
            if ds:
                point["components"] = [{**c, "smiles": ds["doc"]["molecules"].get(c["molecule_id"])} for c in point["components"]]
        return {"points": points, "report": self.jobs[rid].get("report", {})}

    def search(self, query, offset=0):
        if not self.library or not (self.library / "analysis/odor_library.sqlite").exists():
            return {"rows": [], "connected": False}
        query = query[:200]; offset = max(0, min(offset, 200000))
        path = self.library / "analysis/odor_library.sqlite"
        con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            # Search source rows first, then collapse exact structures; names are source-reported.
            rows = con.execute("SELECT inchikey, MAX(name) AS name, MAX(canonical_smiles) AS smiles, GROUP_CONCAT(DISTINCT source) AS sources FROM molecule_source_records WHERE inchikey IS NOT NULL AND (name LIKE ? OR inchikey LIKE ? OR canonical_smiles LIKE ?) GROUP BY inchikey ORDER BY inchikey LIMIT 51 OFFSET ?", ("%"+query+"%", "%"+query+"%", "%"+query+"%", offset)).fetchall()
            return {"rows": [dict(r) for r in rows[:50]], "has_more": len(rows)>50, "connected": True, "offset": offset}
        finally: con.close()


def handler_for(studio):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, status=200, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes): body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(body)

        def host_ok(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.host_ok(): return self.send({"error": "Local host required"}, 403)
            url = urlparse(self.path); q = parse_qs(url.query)
            try:
                if url.path == "/api/state": return self.send(studio.state())
                if url.path in {"/api/smellnet", "/api/dream-benchmark"}:
                    path = studio.root / ("runs/smellnet-v1/report.json" if url.path == "/api/smellnet" else "runs/dream-v1/report.json")
                    if not path.exists(): return self.send({"error": "분석 결과를 준비 중입니다."}, 404)
                    return self.send(path.read_bytes())
                if url.path == "/api/atlas":
                    path = studio.root / "data/group_atlas.json"
                    if not path.exists(): return self.send({"error": "그룹 해석 데이터가 아직 생성되지 않았습니다. build_group_atlas.py를 먼저 실행하세요."}, 404)
                    return self.send(path.read_bytes())
                if url.path == "/api/search": return self.send(studio.search(q.get("q", [""])[0], int(q.get("offset", [0])[0])))
                if url.path == "/api/embedding": return self.send(studio.embedding(q.get("run", [""])[0]))
                if url.path.startswith("/api/file/"):
                    parts = url.path.split("/")
                    if len(parts) != 5: raise ValueError("Invalid result")
                    p = studio.artifact(parts[3], parts[4])
                    return self.send(p.read_bytes(), content_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                assets = {"/": "index.html", "/app.js": "app.js", "/meaning.js": "meaning.js", "/hmdb.js": "hmdb.js", "/benchmark.js": "benchmark.js", "/style.css": "style.css", "/demo.json": "demo.json", "/sers-template.json": "sers-template.json"}
                if url.path not in assets: return self.send({"error": "Not found"}, 404)
                p = WEB / assets[url.path]
                return self.send(p.read_bytes(), content_type=(mimetypes.guess_type(p.name)[0] or "text/plain") + "; charset=utf-8")
            except (ValueError, KeyError, FileNotFoundError) as exc: return self.send({"error": str(exc)}, 400)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.host_ok() or (origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}):
                return self.send({"error": "Same-origin local request required"}, 403)
            if self.headers.get_content_type() != "application/json": return self.send({"error": "JSON required"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000: return self.send({"error": "JSON file limit: 20 MB"}, 413)
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/import": return self.send(studio.import_dataset(body), 201)
                if self.path == "/api/run": return self.send(studio.start(body), 202)
                return self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc: return self.send({"error": str(exc)}, 400)

        def log_message(self, fmt, *args): pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description="VOCLIB Studio local app")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--library", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Studio(args.root, args.library)))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"VOCLIB Studio: {url}", flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
