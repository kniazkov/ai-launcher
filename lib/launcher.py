#!/usr/bin/env python3
"""Portable Linux Ollama/Continue launcher; Python 3.9+, standard library only."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

SOURCE = Path(__file__).resolve().parent.parent
MODELS = json.loads((SOURCE / 'lib/models.json').read_text())
CONTINUE_VERSION = '1.5.47'  # Version used successfully in the original WSL setup.
DEFAULTS = dict(context=8192, max_tokens=1024, temperature=0.2, threads=0,
                timeout=1800, load_timeout=600)
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Failure(Exception):
    pass


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Failure('Эта установка уже занята. Заверши другую сессию/установку.')
        yield


def validate(settings, model):
    for key in ('context', 'max_tokens', 'threads', 'timeout', 'load_timeout'):
        if type(settings.get(key)) is not int:
            raise Failure(f'{key}: требуется целое число.')
    if not 2048 <= settings['context'] <= MODELS[model]['context_max']:
        raise Failure(f"Контекст должен быть от 2048 до {MODELS[model]['context_max']}.")
    if not 1 <= settings['max_tokens'] <= settings['context'] // 2:
        raise Failure('max-tokens должен быть > 0 и не больше половины контекста.')
    if settings['threads'] < 0:
        raise Failure('threads не может быть отрицательным; 0 означает автовыбор.')
    if not 1 <= settings['timeout'] <= 86400 or not 1 <= settings['load_timeout'] <= 86400:
        raise Failure('Таймауты должны быть от 1 до 86400 секунд.')
    if (not isinstance(settings['temperature'], (int, float)) or
            not math.isfinite(settings['temperature']) or
            not 0 <= settings['temperature'] <= 1):
        raise Failure('temperature должна быть от 0 до 1.')
    return settings


def load_settings(state, model):
    path = state / 'profiles' / model / 'settings.json'
    settings = DEFAULTS.copy()
    if path.exists():
        settings.update(json.loads(path.read_text()))
    return validate(settings, model)


def settings_path(state, model):
    return state / 'profiles' / model / 'settings.json'


def options(settings):
    result = dict(num_ctx=settings['context'], num_predict=settings['max_tokens'],
                  temperature=settings['temperature'])
    if settings['threads']:
        result['num_thread'] = settings['threads']
    return result


def configured_tag(model):
    return f'ai-launcher-{model}:latest'


def continue_config(model, settings, port):
    # JSON is valid YAML; one source of truth feeds both Continue and Ollama.
    return {
        'name': f'Local {MODELS[model]["tag"]}', 'version': '1.0.0', 'schema': 'v1',
        'models': [{
            'name': MODELS[model]['tag'], 'provider': 'ollama',
            'model': configured_tag(model), 'apiBase': f'http://127.0.0.1:{port}',
            'capabilities': ['tool_use'], 'roles': ['chat', 'edit', 'apply'],
            'defaultCompletionOptions': {
                'contextLength': settings['context'], 'maxTokens': settings['max_tokens'],
                'temperature': settings['temperature'],
            },
            'requestOptions': {
                # Continue CLI passes this directly to the OpenAI-compatible SDK,
                # which expects milliseconds; our public option uses seconds.
                'timeout': settings['timeout'] * 1000,
                'extraBodyProperties': {'options': options(settings)},
            },
        }],
    }


def api(port, route, payload=None, timeout=10):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}{route}',
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with LOCAL_HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def environment(state, model, settings, port):
    env = os.environ.copy()
    # Do not let inherited Ollama settings bind to the LAN, change model storage,
    # or expose cloud endpoints. Installation keeps HTTPS proxy support.
    for key in list(env):
        if key.startswith(('OLLAMA_', 'CONTINUE_')):
            env.pop(key)
    env.update(
        OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=str(state / 'models'),
        OLLAMA_NO_CLOUD='1', OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
        OLLAMA_CONTEXT_LENGTH=str(settings['context']), OLLAMA_KEEP_ALIVE='5m',
        OLLAMA_LOAD_TIMEOUT=f'{settings["load_timeout"]}s',
        CONTINUE_GLOBAL_DIR=str(state / 'profiles' / model / 'continue'),
        TMPDIR=str(state / 'tmp'), npm_config_cache=str(state / 'npm-cache'),
        npm_config_userconfig=str(state / 'npmrc'),
        PATH=str(state / 'node/bin') + os.pathsep + env.get('PATH', ''),
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost',
    )
    for directory in ('tmp', 'npm-cache', f'profiles/{model}/continue'):
        (state / directory).mkdir(parents=True, exist_ok=True)
    return env


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def server(state, model, settings):
    binary = state / 'ollama/bin/ollama'
    if not binary.is_file():
        raise Failure('Ollama отсутствует. Сначала выполни install.')
    port = free_port()
    env = environment(state, model, settings, port)
    log_dir = state / 'logs' / model
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = log_dir / f'{stamp}-{os.getpid()}.log'
    print(f'Ollama: {log_path}', flush=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen([str(binary), 'serve'], env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise Failure(f'Ollama завершилась. Посмотри {log_path}')
                try:
                    api(port, '/api/version', timeout=1)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise Failure(f'Ollama не запустилась за 60 секунд: {log_path}')
                    time.sleep(0.25)
            yield port, env
        finally:
            # Kill only the process group created by this launcher, including runners.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def download(url, target):
    partial = target.with_name(target.name + '.part')
    run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
         '--proto', '=https', '--proto-redir', '=https', '--output', partial, url])
    os.replace(partial, target)


def architecture():
    machine = platform.machine()
    if machine == 'x86_64':
        return 'amd64', 'x64'
    if machine in ('aarch64', 'arm64'):
        return 'arm64', 'arm64'
    raise Failure('Поддерживаются только Linux x86_64 и ARM64.')


def install_runtimes(state, env):
    ollama_arch, node_arch = architecture()
    downloads = state / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    if not (state / 'ollama/bin/ollama').is_file():
        archive = downloads / 'ollama.tar.zst'
        download(f'https://ollama.com/download/ollama-linux-{ollama_arch}.tar.zst', archive)
        with tempfile.TemporaryDirectory(dir=state / 'tmp') as temp:
            run(['tar', '--use-compress-program=zstd', '-xf', archive, '-C', temp])
            if not (Path(temp) / 'bin/ollama').is_file():
                raise Failure('Архив Ollama не содержит bin/ollama.')
            # A completed runtime becomes visible only after successful extraction.
            os.replace(temp, state / 'ollama')
        archive.unlink()
    if not (state / 'node/bin/node').is_file():
        sums = downloads / 'SHASUMS256.txt'
        base = 'https://nodejs.org/dist/latest-v22.x/'
        download(base + sums.name, sums)
        matches = re.findall(r'^([a-f0-9]{64})\s+(node-v22\.\d+\.\d+-linux-' +
                             re.escape(node_arch) + r'\.tar\.xz)$', sums.read_text(), re.M)
        if len(matches) != 1:
            raise Failure('Не удалось определить официальный архив Node.js 22.')
        digest, name = matches[0]
        archive = downloads / name
        download(base + name, archive)
        sha = hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != digest:
            raise Failure('SHA256 архива Node.js не совпал с SHASUMS256.txt.')
        with tempfile.TemporaryDirectory(dir=state / 'tmp') as temp:
            run(['tar', '-xJf', archive, '--strip-components=1', '-C', temp])
            if not (Path(temp) / 'bin/node').is_file():
                raise Failure('Не найден bin/node.')
            os.replace(temp, state / 'node')
        archive.unlink()
    cli = state / 'continue-cli/node_modules/.bin/cn'
    if not cli.is_file():
        (state / 'npmrc').touch(exist_ok=True)
        run([state / 'node/bin/npm', 'install', '--prefix', state / 'continue-cli',
             '--no-audit', '--no-fund', '--save-exact', f'@continuedev/cli@{CONTINUE_VERSION}'],
            env=env, cwd=state)
    return cli


def install_launchers(root, state):
    target = state / 'app'
    if target.resolve() != SOURCE:
        for directory in ('lib', 'models'):
            shutil.copytree(SOURCE / directory, target / directory, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (root / 'bin').mkdir(exist_ok=True)
    for model in MODELS:
        wrapper = root / 'bin' / f'{model}.sh'
        # Paths are computed at launch, so the complete installation can be moved.
        text = ('#!/usr/bin/env bash\nset -Eeuo pipefail\n'
                'AI_LAUNCHER_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"\n'
                'export AI_LAUNCHER_HOME\n'
                f'exec bash "$AI_LAUNCHER_HOME/.ai-launcher/app/models/{model}.sh" "$@"\n')
        if wrapper.exists() and 'AI_LAUNCHER_HOME' not in wrapper.read_text():
            raise Failure(f'Не буду перезаписывать посторонний файл: {wrapper}')
        wrapper.write_text(text)
        wrapper.chmod(0o755)


def install(root, state, model):
    if os.geteuid() == 0:
        raise Failure('Запусти install обычным пользователем. Для системных пакетов используется sudo.')
    with locked(state / 'operation.lock'):
        settings = load_settings(state, model)
        if not settings_path(state, model).exists():
            atomic_json(settings_path(state, model), settings)
        env = environment(state, model, settings, free_port())
        cli = install_runtimes(state, env)
        install_launchers(root, state)
        with server(state, model, settings) as (port, server_env):
            tag = MODELS[model]['tag']
            present = any(m['name'] in (tag, tag + ':latest')
                          for m in api(port, '/api/tags')['models'])
            if not present:
                available = shutil.disk_usage(state).free / 1e9
                if available < MODELS[model]['size_gb'] + 2:
                    raise Failure(f'Недостаточно места: свободно {available:.1f} ГБ; '
                                  f'на веса нужно примерно {MODELS[model]["size_gb"]} ГБ + запас.')
                run([state / 'ollama/bin/ollama', 'pull', tag], env=server_env)
            manifest = api(port, '/api/show', {'model': tag})
            atomic_json(state / 'profiles' / model / 'model-info.json', manifest)
        # Record the actually installed versions; no automatic runtime upgrades.
        versions = {'continue': CONTINUE_VERSION,
                    'node': subprocess.check_output([str(state / 'node/bin/node'), '--version'],
                                                     text=True).strip()}
        atomic_json(state / 'versions.json', versions)
        print(f'Установлено: {MODELS[model]["tag"]}\nПапка: {root}\n'
              f'Чат: bash "{root}/bin/{model}.sh" chat\n'
              f'Из папки проекта: bash "{root}/bin/{model}.sh" code')


def configure(state, model, args):
    with locked(state / 'operation.lock'):
        current = load_settings(state, model)
        updates = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k, None) is not None}
        result = validate({**current, **updates}, model)
        if updates:
            atomic_json(settings_path(state, model), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f'Настройки: {settings_path(state, model)}')


def chat(port, model, settings):
    print('Локальный чат. /bye — выход; /clear — очистить историю. Файлы автоматически не читаются.')
    history = []
    while True:
        try:
            text = input('>>> ').strip()
        except EOFError:
            break
        if text in ('/bye', '/exit'):
            break
        if text == '/clear':
            history.clear()
            continue
        if not text:
            continue
        messages = history + [{'role': 'user', 'content': text}]
        payload = {'model': MODELS[model]['tag'], 'messages': messages,
                   'stream': True, 'options': options(settings)}
        request = urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        answer, thinking = '', False
        with LOCAL_HTTP.open(request, timeout=settings['timeout']) as response:
            for line in response:
                event = json.loads(line)
                if event.get('error'):
                    raise Failure(event['error'])
                msg = event.get('message', {})
                if msg.get('thinking'):
                    if not thinking:
                        print('[Thinking] ', end='', flush=True)
                        thinking = True
                    print(msg['thinking'], end='', flush=True)
                content = msg.get('content', '')
                if content:
                    if thinking:
                        print('\n[Ответ] ', end='', flush=True)
                        thinking = False
                    print(content, end='', flush=True)
                    answer += content
        print()
        history = messages + [{'role': 'assistant', 'content': answer}]
        # Local chat is intentionally simple: Ollama may trim oldest context;
        # /clear explicitly starts over. Nothing is saved to a transcript file.


def code(port, state, model, settings, env, args):
    info = api(port, '/api/show', {'model': MODELS[model]['tag']})
    if 'tools' not in info.get('capabilities', []):
        raise Failure('Эта сборка модели не объявляет поддержку tools. chat доступен; '
                      'для code выбери Qwen3 или Qwen3 Coder.')
    # Some Continue releases discover context from /api/show, overriding YAML.
    # A lightweight derived tag exposes the same parameters there, without
    # duplicating the base weights or changing another model's configuration.
    created = api(port, '/api/create', {
        'model': configured_tag(model), 'from': MODELS[model]['tag'],
        'parameters': options(settings), 'stream': False,
    }, timeout=settings['timeout'])
    if created.get('status') != 'success':
        raise Failure(f'Не удалось создать настроенный тег модели: {created}')
    workspace_id = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:16]
    env = env.copy()
    env['CONTINUE_GLOBAL_DIR'] = str(state / 'profiles' / model / 'continue' / workspace_id)
    Path(env['CONTINUE_GLOBAL_DIR']).mkdir(parents=True, exist_ok=True)
    config = state / 'profiles' / model / 'continue.yaml'
    atomic_json(config, continue_config(model, settings, port))
    command = [state / 'continue-cli/node_modules/.bin/cn', '--config', config]
    if not Path(command[0]).is_file():
        raise Failure('Continue не установлен. Повтори install.')
    for tool in ('Fetch', 'UploadArtifact'):
        command += ['--exclude', tool]
    command += ['--ask' if args.allow_shell and not args.read_only and not args.prompt else '--exclude', 'Bash']
    for tool in ('Write', 'Edit', 'MultiEdit'):
        command += ['--exclude' if args.read_only or args.prompt else '--ask', tool]
    if args.resume:
        command += ['--resume']
    if args.prompt:
        command += ['-p', args.prompt]
    print(f'Проект: {Path.cwd()}\nКонтекст: {settings["context"]}; '
          f'ответ: {settings["max_tokens"]}. Конфиг: {config}', flush=True)
    run(command, env=env)


def offline_command(root, model, args):
    username = pwd.getpwuid(os.getuid()).pw_name
    if os.geteuid() == 0:
        raise Failure('Чат и агент запускаются обычным пользователем, не root.')
    required = ('sudo', 'unshare', 'ip', 'runuser')
    if any(shutil.which(cmd) is None for cmd in required):
        raise Failure('Для изоляции требуются sudo, unshare, ip и runuser. Выполни install.')
    original_ns = os.readlink('/proc/self/ns/net')
    command = ['sudo', 'unshare', '--net', '--', 'bash', str(SOURCE / 'lib/netns.sh'),
               username, sys.executable, str(SOURCE / 'lib/launcher.py'),
               '--model', model, args.command, '--home', str(root),
               '--internal-parent-netns', original_ns]
    if args.command == 'code':
        for name in ('read_only', 'allow_shell', 'resume'):
            if getattr(args, name):
                command += ['--' + name.replace('_', '-')]
        if args.prompt:
            command += ['--prompt', args.prompt]
    return command


def parser():
    p = argparse.ArgumentParser(description='Локальные модели: install / configure / chat / code')
    p.add_argument('--model', choices=MODELS, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('install', 'configure', 'chat', 'code'):
        sp = sub.add_parser(name)
        sp.add_argument('--home', type=Path, help='Папка установки; install по умолчанию использует cwd')
        if name == 'configure':
            for key in ('context', 'max_tokens', 'threads', 'timeout', 'load_timeout'):
                sp.add_argument('--' + key.replace('_', '-'), type=int)
            sp.add_argument('--temperature', type=float)
        if name in ('chat', 'code'):
            sp.add_argument('--internal-parent-netns', help=argparse.SUPPRESS)
        if name == 'code':
            sp.add_argument('--read-only', action='store_true', help='Запретить правки')
            sp.add_argument('--allow-shell', action='store_true', help='Разрешить команды с подтверждением')
            sp.add_argument('--resume', action='store_true')
            sp.add_argument('--prompt', help='Один диагностический запрос без TUI, запись запрещена')
    return p


def main():
    args = parser().parse_args()
    if platform.system() != 'Linux':
        raise Failure('Нужен Linux или WSL2.')
    if args.home:
        root = args.home.expanduser().resolve()
    elif args.command == 'install':
        root = Path.cwd().resolve()
    else:
        root = Path(os.environ.get('AI_LAUNCHER_HOME', str(SOURCE))).expanduser().resolve()
    state = root / '.ai-launcher'
    if args.command == 'install':
        install(root, state, args.model)
    elif args.command == 'configure':
        configure(state, args.model, args)
    else:
        if not settings_path(state, args.model).exists():
            raise Failure(f'Модель не установлена в {root}. Сначала выполни install.')
        if not args.internal_parent_netns:
            print('Запуск без внешней IP-сети; sudo нужен только для создания namespace.', flush=True)
            run(offline_command(root, args.model, args))
            return
        if os.readlink('/proc/self/ns/net') == args.internal_parent_netns:
            raise Failure('Сетевая изоляция не сработала.')
        with locked(state / 'operation.lock'):
            settings = load_settings(state, args.model)
            with server(state, args.model, settings) as (port, env):
                if args.command == 'chat':
                    chat(port, args.model, settings)
                else:
                    code(port, state, args.model, settings, env, args)


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print('\nОстановлено.', file=sys.stderr)
        sys.exit(130)
    except (Failure, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        sys.exit(1)
