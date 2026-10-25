# Test guide

Run the complete suite from the repository root:

```bash
.venv/bin/python -m pytest -q
```

Run one calculation while working on it, for example:

```bash
.venv/bin/python -m pytest -q tests/test_conduction.py
```

## Web app tests

`test_web.py` is part of the standard pytest suite. It checks request limits,
calculation jobs, separate visitors, the presentation circuit, and whether
the browser's loop shapes can be used by the Python solver.

If Node is installed, pytest also runs `web_geometry.mjs`. That test checks
the JavaScript rules for reshaping loops, stopping at constraints, and keeping
resistors attached as the loop changes. Node is needed only to run this test;
there are no npm packages to install. You can run it directly from the
repository root with `node tests/web_geometry.mjs`.

For an optional check of the actual browser interface, install Playwright,
start the app in one terminal, and run the browser test in another:

```bash
.venv/bin/python -m pip install -e '.[browser-test]'
.venv/bin/python -m steercurrent
```

In another terminal:

```bash
.venv/bin/python tests/browser_smoke.py
```

`browser_smoke.py` uses a locally installed Chrome browser by default. It
checks dragging, undo, field views, separate tabs, and a narrow mobile viewport.
Screenshots go to `/tmp/steercurrent-browser` by default. Playwright and Node
are test tools; neither is needed to run the app.
