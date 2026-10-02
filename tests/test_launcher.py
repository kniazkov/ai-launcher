import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'lib/launcher.py')
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)

FAKE_OLLAMA = r'''#!/usr/bin/env python3
import http.server, json, os
from pathlib import Path
host, port = os.environ['OLLAMA_HOST'].split(':')
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'version': 'test', 'models': [{'name': 'qwen3-coder:30b'}]}).encode())
    def do_POST(self):
        obj = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200); self.end_headers()
        if self.path == '/api/show':
            self.wfile.write(b'{"capabilities":["completion","tools"]}')
        elif self.path == '/api/create':
            Path(os.environ['TMPDIR'], 'create.json').write_text(json.dumps(obj))
            self.wfile.write(b'{"status":"success"}')
        else:
            Path(os.environ['OLLAMA_MODELS']).mkdir(parents=True, exist_ok=True)
            Path(os.environ['OLLAMA_MODELS'], 'request.json').write_text(json.dumps(obj))
            self.wfile.write(b'{"message":{"role":"assistant","content":"OK"},"done":true}\n')
http.server.HTTPServer((host, int(port)), Handler).serve_forever()
'''
FAKE_CN = r'''#!/usr/bin/env python3
import os, sys, json
from pathlib import Path
record = {'cwd': os.getcwd(), 'args': sys.argv[1:], 'global_dir': os.environ['CONTINUE_GLOBAL_DIR']}
Path(os.environ['TMPDIR'], 'cn.json').write_text(json.dumps(record))
print('OK')
'''

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ai test ', dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        self.state = self.root / '.ai-launcher'
        self.model = 'qwen3-coder-30b'

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, *args):
        return ai.parser().parse_args(['--model', self.model, *args])

    def fake_runtime(self):
        for name, content in [('ollama/bin/ollama', FAKE_OLLAMA),
                              ('continue-cli/node_modules/.bin/cn', FAKE_CN)]:
            path = self.state / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            path.chmod(0o755)

    def test_models_have_identical_entrypoints(self):
        self.assertEqual(len(ai.MODELS), 9)
        for model in ai.MODELS:
            path = ROOT / 'models' / (model + '.sh')
            subprocess.run(['bash', '-n', str(path)], check=True)
            result = subprocess.run(['bash', str(path), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for command in ('install', 'configure', 'chat', 'code'):
                self.assertIn(command, result.stdout)

    def test_profiles_are_independent(self):
        ai.configure(self.state, self.model, self.arguments('configure', '--context', '32768', '--max-tokens', '4096'))
        self.assertEqual(ai.load_settings(self.state, self.model)['context'], 32768)
        self.assertEqual(ai.load_settings(self.state, 'qwen3-14b'), ai.DEFAULTS)
        ai.configure(self.state, self.model, self.arguments('configure', '--threads', '6'))
        result = ai.load_settings(self.state, self.model)
        self.assertEqual(result['threads'], 6)
        self.assertEqual(result['max_tokens'], 4096)

    def test_invalid_settings_do_not_overwrite(self):
        ai.configure(self.state, self.model, self.arguments('configure', '--context', '16384'))
        before = ai.settings_path(self.state, self.model).read_bytes()
        for flags in [('--context', '1024'), ('--max-tokens', '16000'),
                      ('--temperature', 'nan'), ('--threads', '-1'), ('--timeout', '0')]:
            with self.assertRaises(ai.Failure):
                ai.configure(self.state, self.model, self.arguments('configure', *flags))
            self.assertEqual(ai.settings_path(self.state, self.model).read_bytes(), before)

    def test_config_syncs_options(self):
        cfg = {**ai.DEFAULTS, 'context': 32768, 'max_tokens': 4096, 'temperature': 0.0, 'threads': 4}
        doc = ai.continue_config(self.model, cfg, 12345)['models'][0]
        self.assertEqual(doc['defaultCompletionOptions']['contextLength'], 32768)
        self.assertEqual(doc['defaultCompletionOptions']['maxTokens'], 4096)
        self.assertEqual(doc['requestOptions']['extraBodyProperties']['options'], ai.options(cfg))
        self.assertEqual(ai.options(cfg)['temperature'], 0.0)

    def test_wrapper_works_from_project_and_after_move(self):
        ai.install_launchers(self.root, self.state)
        moved = self.root / 'moved install'
        moved.mkdir()
        shutil.move(str(self.state), moved)
        shutil.move(str(self.root / 'bin'), moved)
        project = self.root / 'some project'
        project.mkdir()
        wrapper = moved / 'bin' / (self.model + '.sh')
        result = subprocess.run(['bash', str(wrapper), 'configure', '--context', '16384'],
                                cwd=project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(ai.load_settings(moved / '.ai-launcher', self.model)['context'], 16384)
        self.assertFalse((project / '.ai-launcher').exists())

    def test_server_chat_and_cleanup(self):
        self.fake_runtime()
        with ai.server(self.state, self.model, ai.DEFAULTS) as (port, env):
            self.assertEqual(env['OLLAMA_NO_CLOUD'], '1')
            with patch('builtins.input', side_effect=['Hello', '/bye']):
                ai.chat(port, self.model, ai.DEFAULTS)
            payload = json.loads((self.state / 'models/request.json').read_text())
            self.assertEqual(payload['model'], 'qwen3-coder:30b')
            self.assertEqual(payload['options']['num_ctx'], 8192)
            self.assertEqual(payload['options']['num_predict'], 1024)
        with self.assertRaises(OSError):
            ai.api(port, '/api/version', timeout=1)

    def test_code_cwd_permissions_and_generated_config(self):
        self.fake_runtime()
        project = self.root / 'project'
        project.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(project)
            with ai.server(self.state, self.model, ai.DEFAULTS) as (port, env):
                ai.code(port, self.state, self.model, ai.DEFAULTS, env,
                        self.arguments('code', '--read-only', '--allow-shell'))
            record = json.loads((self.state / 'tmp/cn.json').read_text())
            self.assertEqual(record['cwd'], str(project))
            pairs = list(zip(record['args'], record['args'][1:]))
            self.assertIn(('--exclude', 'Bash'), pairs)
            self.assertIn(('--exclude', 'Write'), pairs)
            self.assertIn(('--exclude', 'Fetch'), pairs)
            cfg = json.loads((self.state / 'profiles' / self.model / 'continue.yaml').read_text())
            self.assertEqual(cfg['models'][0]['defaultCompletionOptions']['contextLength'], 8192)
            self.assertTrue(Path(record['global_dir']).is_dir())
            created = json.loads((self.state / 'tmp/create.json').read_text())
            self.assertEqual(created['from'], 'qwen3-coder:30b')
            self.assertEqual(created['parameters']['num_ctx'], 8192)
            self.assertEqual(cfg['models'][0]['model'], created['model'])
        finally:
            os.chdir(previous)

    def test_install_keeps_settings_and_skips_present_model(self):
        self.fake_runtime()
        ai.configure(self.state, self.model, self.arguments('configure', '--context', '32768'))
        before = ai.settings_path(self.state, self.model).read_bytes()
        with patch.object(ai, 'install_runtimes'), patch.object(ai, 'install_launchers'), \
             patch.object(ai.os, 'geteuid', return_value=1000), \
             patch.object(ai.subprocess, 'check_output', return_value='v22.0.0\n'), \
             patch.object(ai, 'run') as run_mock:
            ai.install(self.root, self.state, self.model)
            run_mock.assert_not_called()
        self.assertEqual(ai.settings_path(self.state, self.model).read_bytes(), before)

    def test_lock_refuses_parallel_session(self):
        with ai.locked(self.state / 'operation.lock'):
            with self.assertRaises(ai.Failure):
                with ai.locked(self.state / 'operation.lock'):
                    pass

    def test_bootstrap_installs_missing_package_through_apt(self):
        tools = self.root / 'tools'
        tools.mkdir()
        log = self.root / 'apt.log'
        for name in ('python3', 'curl', 'tar', 'xz', 'git', 'ip', 'unshare', 'runuser'):
            file = tools / name
            file.write_text('#!/bin/sh\nexit 0\n')
            file.chmod(0o755)
        apt = tools / 'apt-get'
        apt.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_APT_LOG"\n')
        apt.chmod(0o755)
        sudo = tools / 'sudo'
        sudo.write_text('#!/bin/sh\nexec "$@"\n')
        sudo.chmod(0o755)
        env = {**os.environ, 'PATH': str(tools), 'TEST_APT_LOG': str(log)}
        subprocess.run(['/bin/bash', '-c', 'source "$1"; ai_dependencies', '_',
                        str(ROOT / 'lib/bootstrap.sh')], env=env, check=True)
        calls = log.read_text().splitlines()
        self.assertEqual(calls[0], 'update')
        self.assertTrue(calls[1].startswith('install -y'))
        self.assertIn('zstd', calls[1])

if __name__ == '__main__':
    unittest.main()
