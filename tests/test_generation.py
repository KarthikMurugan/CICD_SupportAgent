import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from pydantic import BaseModel
from cicd_support_rag.generation import NebiusClient, OllamaClient, GenerationError, generation_client


class Output(BaseModel):
    message: str


class NebiusTests(unittest.TestCase):
    def client(self):
        with patch.dict(os.environ, {'NEBIUS_API_KEY': 'test-key'}):
            return NebiusClient()

    def test_structured_request_and_validation(self):
        response = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
            'message': {'content': '{"message":"ok"}'}}], 'usage': {'total_tokens': 12}}).encode())
        with patch('urllib.request.urlopen', return_value=response) as request:
            client = self.client()
            self.assertEqual(client.complete(Output, 'Use evidence only', {'question': 'test'}).message, 'ok')
            sent = request.call_args.args[0]
            body = json.loads(sent.data)
            self.assertEqual(sent.full_url, 'https://api.tokenfactory.nebius.com/v1/chat/completions')
            self.assertEqual(body['response_format']['json_schema']['schema'], Output.model_json_schema())
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(client.last_usage['total_tokens'], 12)

    def test_truncated_or_invalid_output_is_rejected(self):
        for finish, content in [('length', '{"message":"partial"}'), ('stop', 'not json')]:
            response = io.BytesIO(json.dumps({'choices': [{'finish_reason': finish,
                'message': {'content': content}}]}).encode())
            with patch('urllib.request.urlopen', return_value=response), self.assertRaises(GenerationError):
                self.client().complete(Output, 'system', {})

    def test_auth_errors_do_not_leak_server_body(self):
        error = HTTPError('https://api.tokenfactory.nebius.com/v1/chat/completions', 401,
                          'SECRET', {}, io.BytesIO(b'SECRET'))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(GenerationError) as raised:
                self.client().complete(Output, 'system', {})
        self.assertNotIn('SECRET', str(raised.exception))
        self.assertIn('authentication', str(raised.exception))

    def test_explicit_provider_selection(self):
        with patch.dict(os.environ, {'GENERATION_PROVIDER': 'nebius', 'NEBIUS_API_KEY': 'test-key'}):
            self.assertIsInstance(generation_client(), NebiusClient)
        with patch.dict(os.environ, {'GENERATION_PROVIDER': 'ollama'}):
            self.assertIsInstance(generation_client(), OllamaClient)
        with patch.dict(os.environ, {'GENERATION_PROVIDER': 'unknown'}), self.assertRaises(ValueError):
            generation_client()
