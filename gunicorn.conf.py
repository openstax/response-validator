from os import getenv

accesslog = '-'
bind = getenv('GUNICORN_BIND', '127.0.0.1:8000')
threads = int(getenv('GUNICORN_THREADS', 2))
umask = int(getenv('GUNICORN_UMASK', 7))
worker_class = getenv('GUNICORN_WORKER_CLASS', 'gevent')
