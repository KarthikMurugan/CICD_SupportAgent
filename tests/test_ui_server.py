import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import time
import unittest

from cicd_support_rag.ui_server import ChatService, contextual_question, make_handler


class UIServerTests(unittest.TestCase):
    def setUp(self):
        self.questions = []

        def answerer(question, on_stage):
            self.questions.append(question)
            on_stage("retrieve")
            return {"answer": "A supported answer [doc-0001]", "status": "answered", "retrieved_sources": []}

        self.service = ChatService(answerer=answerer)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.service.worker.shutdown(wait=True)

    def request(self, method, path, payload=None, extra_headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Content-Type": "application/json", "X-PipelineHub-Client": "web"}
        headers.update(extra_headers or {})
        conn.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=headers)
        response = conn.getresponse()
        status, body = response.status, response.read()
        conn.close()
        return status, body

    def test_async_chat_returns_the_answer(self):
        status, body = self.request("POST", "/api/chat", {"question": "What is PipelineHub?", "history": []})
        self.assertEqual(status, 202)
        job_id = json.loads(body)["job_id"]
        for _ in range(50):
            _, body = self.request("GET", "/api/jobs/" + job_id)
            result = json.loads(body)
            if result["status"] == "done":
                break
            time.sleep(.01)
        self.assertEqual(result["result"]["status"], "answered")
        self.assertEqual(self.questions, ["What is PipelineHub?"])

    def test_unknown_paths_cannot_expose_credentials_or_source_files(self):
        for path in ("/.env", "/../.env", "/project-config.json"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_cross_origin_or_untrusted_host_is_rejected(self):
        request = {"question": "Hello"}
        self.assertEqual(self.request("POST", "/api/chat", request, {"Origin": "https://untrusted.example"})[0], 403)
        self.assertEqual(self.request("GET", "/", extra_headers={"Host": "untrusted.example"})[0], 403)

    def test_empty_and_non_string_questions_are_rejected(self):
        for question in ("", " ", 5, ["not a question"]):
            self.assertEqual(self.request("POST", "/api/chat", {"question": question})[0], 400)

    def test_clarification_carries_forward_user_details(self):
        history = [{"role": "user", "content": "My pipeline failed."},
                   {"role": "assistant", "content": "Which step?", "status": "needs_clarification"}]
        result = contextual_question("Deployment, with error 403", history)
        self.assertIn("My pipeline failed.", result)
        self.assertIn("Deployment, with error 403", result)
        self.assertNotIn("Which step?", result)

    def test_internal_errors_do_not_leak_to_browser(self):
        def fail(*_args, **_kwargs):
            raise RuntimeError("SECRET_SHOULD_NOT_APPEAR")
        service = ChatService(answerer=fail)
        try:
            job_id = service.submit("What is PipelineHub?", [])
            for _ in range(50):
                result = service.get(job_id)
                if result["status"] == "error":
                    break
                time.sleep(.01)
            self.assertEqual(result["status"], "error")
            self.assertNotIn("SECRET_SHOULD_NOT_APPEAR", json.dumps(result))
        finally:
            service.worker.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
