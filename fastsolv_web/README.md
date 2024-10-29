# `fastsolv_web`
This directory contains a single python script, `fastsolv_web.py`, which renders and serves a simple flask app to provide access to `fastsolv` via the browser.

To serve the app locally, use flask.
To serve the app via the internet, install `waitress` with `pip` and run `waitress-serve --host 127.0.0.1 --port 5000 fastsolv_web:app` (assuming that `nginx` is running and redirecting HTTPS traffic to 5000, and that the DNS record on the domain has been configured correctly).
Other faster WSGIs are incompatible with this web application because, behind the scenes, `mordred` uses multiprocessing to calculate molecular descriptors and this breaks uWSGI and gunicorn.
