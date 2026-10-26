"""The web adapter preserves core fields, rejects oversized work, and isolates jobs."""
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
import pytest

from steercurrent.web.server import Jobs, same_origin
from steercurrent.web.simulation import simulate, validate_request

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def request_data():
    return {'geometry': json.loads((ROOT / 'tests/orthogonal_loop.json').read_text())}


def test_presentation_demo_concentrates_dissipation_in_resistors():
    demo = json.loads((ROOT / 'src/steercurrent/web/static/presentation.json').read_text())
    assert [r['id'] for r in demo['config']['resistors']] == ['R1', 'R2']
    result = simulate({'geometry': demo})
    assert result['summary']['source_power'] == pytest.approx(result['summary']['joule_power'], rel=1e-8)
    assert result['summary']['resistor_power'] / result['summary']['joule_power'] > .9


def test_simulation_returns_finite_browser_fields(request_data):
    result = simulate(request_data)
    json.dumps(result, allow_nan=False)
    nv, nt = len(result['vertices']), len(result['triangles'])
    assert np.shape(result['potential']) == (nv,)
    assert np.shape(result['magnetic']) == (nv,)
    for field in ('electric', 'current', 'energy'):
        assert np.shape(result[field]) == (nt, 2)
    power = np.asarray(result['power_exchange'])
    assert power.shape == (nt,)
    assert np.any(power > 0) and np.any(power < 0)
    assert np.shape(result['toy_electric']) == (nt, 2)
    assert len(result['toy_charges']['edges']) == len(result['toy_charges']['density'])
    assert result['summary']['source_power'] == pytest.approx(result['summary']['joule_power'], rel=1e-8)
    assert json.loads(json.dumps(result['geometry'])) == request_data['geometry']
    for trace in result['charges'].values():
        assert np.shape(trace['edges']) == (len(trace['density']), 2)
    # A second visitor's battery changes only that visitor's calculation.
    zero = deepcopy(request_data)
    zero['geometry']['config']['battery']['emf'] = 0
    other = simulate(zero)
    np.testing.assert_allclose(other['potential'], 0, atol=1e-12)
    np.testing.assert_allclose(other['power_exchange'], 0, atol=1e-12)
    assert result['summary']['current'] > 0


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(resolution='unbounded'),
    lambda d: d['geometry']['config']['battery'].update(emf=11),
    lambda d: d['geometry']['config']['battery'].update(emf=float('nan')),
    lambda d: d['geometry']['config'].update(wire_resistivity=1e-8),
    lambda d: d['geometry']['config'].update(length=100),
])
def test_invalid_requests_fail_before_meshing(request_data, mutation):
    mutation(request_data)
    with pytest.raises(ValueError):
        validate_request(request_data)


def test_post_origin_accepts_https_proxy_and_rejects_other_sites():
    host = 'steercurrent.onrender.com'
    assert same_origin(f'https://{host}', host)
    assert same_origin(f'http://{host}', host)
    assert not same_origin('https://another.example', host)


def test_queue_is_bounded_and_recovers_after_failure():
    started, release = threading.Event(), threading.Event()

    class Executor:
        def submit(self, function, data):
            future = Future()
            started.set()
            assert release.wait(5), 'Test did not release the worker.'
            if data.get('fail'):
                future.set_exception(ValueError('Invalid test circuit'))
            else:
                future.set_result(data)
            return future

    jobs = Jobs(Executor())
    try:
        first = jobs.submit({'visitor': 'one'})
        assert started.wait(5)
        second = jobs.submit({'visitor': 'two'})
        failed = jobs.submit({'fail': True})
        last = jobs.submit({'visitor': 'three'})
        assert jobs.get(first)['status'] == 'running'
        assert jobs.get(second)['status'] == 'queued'
        with pytest.raises(ValueError, match='queue is full'):
            jobs.submit({})
        release.set()
        jobs.pending.join()
        assert json.loads(jobs.get(first)['encoded'])['result'] == {'visitor': 'one'}
        assert json.loads(jobs.get(second)['encoded'])['result'] == {'visitor': 'two'}
        assert jobs.get(failed)['message'] == 'Invalid test circuit'
        assert jobs.get(last)['status'] == 'done'
        assert jobs.get('unknown') is None
    finally:
        release.set()
        jobs.close()


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is optional; needed only for browser geometry unit tests.')
def test_browser_geometry_rules():
    subprocess.run(['node', str(ROOT / 'tests/web_geometry.mjs')], check=True, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is optional')
@pytest.mark.parametrize('kind', ['rectangle', 'step', 'u', 'stairs'])
def test_editor_presets_work_with_core(kind):
    """Each editor shape must remain usable by the Python solver after a resize."""
    script = """
      import {readFileSync} from 'node:fs';
      const source=readFileSync('src/steercurrent/web/static/geometry.js');
      const {preset,handles,reshape}=await import(`data:text/javascript;base64,${source.toString('base64')}`);
      const original=preset(process.argv[1]);
      const top=handles(original).find(h=>h.id==='top');
      const resized=reshape(original,'top',[top.point[0],top.point[1]+.04]);
      console.log(JSON.stringify([original,resized]));
    """
    original, resized = json.loads(subprocess.check_output(
        ['node', '--input-type=module', '-e', script, kind], cwd=ROOT, text=True))
    for geometry in (original, resized):
        validate_request({'geometry': geometry})
    assert resized['config']['length'] != original['config']['length']
    result = simulate({'geometry': resized})
    assert result['summary']['current'] > 0
    assert result['summary']['source_power'] == pytest.approx(result['summary']['joule_power'], rel=1e-8)
