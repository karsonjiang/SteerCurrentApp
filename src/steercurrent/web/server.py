"""Small same-origin HTTP server with a bounded, serial simulation queue."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import multiprocessing
from pathlib import Path
from queue import Queue, Full
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .simulation import simulate, validate_request

_STATIC = Path(__file__).with_name('static')
_MAX_BODY = 64_000


class Jobs:
    """Keep results private to unguessable job IDs; execute just one solve at a time."""

    def __init__(self, executor):
        self.executor = executor
        self.pending = Queue(maxsize=3)
        self.records = {}
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, data):
        with self.lock:
            # Do not retain browser results indefinitely in a small server's memory.
            completed = [key for key, value in self.records.items()
                         if value['status'] in ('done', 'error')]
            for key in completed:
                if len(self.records) >= 8 or time.monotonic()-self.records[key]['created'] > 900:
                    del self.records[key]
            key = uuid4().hex
            self.records[key] = {'status': 'queued', 'created': time.monotonic()}
            try:
                self.pending.put_nowait((key, data))
            except Full:
                del self.records[key]
                raise ValueError('The calculation queue is full. Please try again shortly.')
            return key

    def get(self, key):
        with self.lock:
            record = self.records.get(key)
            if record is None or (record['status'] in ('done', 'error') and
                                  time.monotonic()-record['created'] > 900):
                self.records.pop(key, None)
                return None
            return {name: value for name, value in record.items() if name != 'created'}

    def _run(self):
        while True:
            item = self.pending.get()
            if item is None:
                self.pending.task_done()
                return
            key, data = item
            with self.lock:
                self.records[key]['status'] = 'running'
            try:
                result = self.executor.submit(simulate, data).result()
                # Encode once; repeated polls must not serialize a large mesh again.
                encoded = json.dumps({'status': 'done', 'result': result}, allow_nan=False).encode()
                update = {'status': 'done', 'encoded': encoded}
            except Exception as error:
                update = {'status': 'error', 'message': str(error) or type(error).__name__}
            with self.lock:
                self.records[key].update(update)
            self.pending.task_done()

    def close(self):
        """Finish accepted work and stop the queue thread before closing the executor."""
        self.pending.join()
        self.pending.put(None)
        self.worker.join()


def make_server(host, port, jobs):
    """Create a server; no network sockets or worker processes start at import time."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Closing a tab must not interrupt another visitor's calculation.

        def do_GET(self):
            path = urlsplit(self.path).path
            if path.startswith('/api/jobs/'):
                record = jobs.get(path.removeprefix('/api/jobs/'))
                if record is None:
                    return self._send(404, {'message': 'Calculation expired or was not found. Run it again.'})
                return self._send(200, record.get('encoded', record))
            if path == '/api/health':
                return self._send(200, {'status': 'ok'})
            # Explicit asset allowlist prevents reading arbitrary repository files.
            name = 'index.html' if path == '/' else path.removeprefix('/')
            if name not in {'index.html', 'app.js', 'geometry.js', 'style.css', 'presentation.json'}:
                return self._send(404, {'message': 'Not found.'})
            file = _STATIC / name
            return self._send(200, file.read_bytes(), mimetypes.guess_type(name)[0] or 'text/plain')

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self._send(403, {'message': 'Use the app from this server address.'})
            path = urlsplit(self.path).path
            if path not in ('/api/validate', '/api/jobs'):
                return self._send(404, {'message': 'Not found.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= _MAX_BODY:
                    return self._send(413, {'message': 'Circuit request is too large or empty.'})
                if self.headers.get_content_type() != 'application/json':
                    return self._send(415, {'message': 'Send application/json.'})
                data = json.loads(self.rfile.read(length))
                geometry = validate_request(data)
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
                return self._send(400, {'message': str(error)})
            if path == '/api/validate':
                return self._send(200, {'geometry': geometry.to_dict()})
            try:
                key = jobs.submit(data)
            except ValueError as error:
                return self._send(429, {'message': str(error)})
            return self._send(202, {'id': key})

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else '') not in ('200', '202'):
                super().log_message(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Launch the local SteerCurrent circuit playground.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    # Spawn gives the numerical worker a fresh process, with Gmsh on its main thread.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = Jobs(executor)
        server = make_server(args.host, args.port, jobs)
        print(f'SteerCurrent is ready at http://{args.host}:{server.server_port}', flush=True)
        print('Press Ctrl+C to stop. Calculations run one at a time.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            jobs.close()
