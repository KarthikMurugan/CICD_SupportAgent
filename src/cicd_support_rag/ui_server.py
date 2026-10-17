"""Serve the local PipelineHub chat UI. Run: python -m cicd_support_rag.ui_server

The server binds to loopback. Credentials stay in the Python process. Browser
requests use same-origin JSON; model work runs serially in a background worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from cicd_support_rag.bot import ask

PROJECT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"
STAGES = {"retrieve": "drafting", "draft": "checking", "verify": "finalizing",
          "answer": "finalizing", "clarify": "finalizing", "escalate": "finalizing"}


def contextual_question(question, history):
    """Carry user-provided details forward after a clarification, without treating
    previous assistant output as knowledge-base evidence.
    """
    if history and history[-1].get("status") == "needs_clarification":
        previous = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
        if previous:
            return f"Original support request: {previous}\nAdditional details from the user: {question}"
    return question


class ChatService:
    def __init__(self, answerer=ask):
        self.answerer = answerer
        self.jobs = {}
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipelinehub-chat")

    def submit(self, question, history):
        with self.lock:
            now = time.monotonic()
            expired = [key for key, job in self.jobs.items()
                       if job["status"] in {"done", "error"} and now - job["created"] > 3600]
            for key in expired:
                del self.jobs[key]
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 4:
                raise ValueError("A few questions are already waiting. Please try again shortly.")
            job_id = secrets.token_urlsafe(18)
            self.jobs[job_id] = {"status": "queued", "stage": "queued", "created": now}
        self.worker.submit(self._run, job_id, question, history)
        return job_id

    def _run(self, job_id, question, history):
        def progress(node):
            with self.lock:
                self.jobs[job_id]["stage"] = STAGES.get(node, "checking")

        with self.lock:
            self.jobs[job_id].update(status="running", stage="retrieving")
        try:
            result = self.answerer(contextual_question(question, history), on_stage=progress)
            result["question"] = question
            with self.lock:
                self.jobs[job_id].update(status="done", stage="done", result=result)
        except Exception:
            # Do not expose exception strings, credentials, or request internals.
            with self.lock:
                self.jobs[job_id].update(status="error", stage="error", error=
                    "The answer couldn't be completed. Check the configured answer provider and Pinecone connection, then try again.")

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return {k: v for k, v in job.items() if k != "created"} if job else None


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def trusted_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def json_response(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.trusted_host():
                return self.json_response(403, {"error": "This interface is available only on localhost."})
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/health":
                load_dotenv(PROJECT / ".env")
                ready = False
                try:
                    from cicd_support_rag.generation import generation_client, NebiusClient
                    client = generation_client()
                    if isinstance(client, NebiusClient):
                        ready = client.ready()
                    else:
                        with urllib.request.urlopen(client.base_url + "/api/tags", timeout=3) as response:
                            models = json.load(response).get("models", [])
                        ready = any(m.get("name") == client.model for m in models)
                except Exception:
                    pass
                return self.json_response(200, {
                    "model_ready": ready,
                    "knowledge_configured": bool(os.environ.get("PINECONE_API_KEY") and (PROJECT / "data/embedded/manifest.json").exists()),
                })
            if path.startswith("/api/jobs/"):
                job = service.get(path.removeprefix("/api/jobs/"))
                return self.json_response(200 if job else 404, job or {"error": "This request is no longer available. Please send it again."})
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                      "/styles.css": ("styles.css", "text/css")}
            if path not in assets:
                return self.json_response(404, {"error": "Page not found"})
            name, content_type = assets[path]
            body = (WEB / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.trusted_host():
                return self.json_response(403, {"error": "Local requests only"})
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if origin and origin not in allowed:
                return self.json_response(403, {"error": "Use the local chat page to send a question."})
            if self.headers.get("X-PipelineHub-Client") != "web" or self.headers.get_content_type() != "application/json":
                return self.json_response(415, {"error": "A JSON chat request is required."})
            if self.path != "/api/chat":
                return self.json_response(404, {"error": "Endpoint not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 40000:
                    raise ValueError("Please shorten your question.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Enter a support question.")
                question = payload.get("question")
                history = payload.get("history", [])
                if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                    raise ValueError("Enter a question between 1 and 4,000 characters.")
                if not isinstance(history, list) or len(history) > 8:
                    raise ValueError("Conversation context is too long.")
                for item in history:
                    if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"} or not isinstance(item.get("content"), str) or len(item["content"]) > 4000:
                        raise ValueError("Invalid conversation context.")
                job_id = service.submit(question.strip(), history)
                self.json_response(202, {"job_id": job_id})
            except (ValueError, UnicodeError, TypeError):
                self.json_response(400, {"error": "Please enter a question of up to 4,000 characters and try again."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(PROJECT)
    service = ChatService()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    except OSError as error:
        parser.exit(2, f"Could not start the chat interface on port {args.port}: {error}\n")
    print(f"PipelineHub chat is ready at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.worker.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
