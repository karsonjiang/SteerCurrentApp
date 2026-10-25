"""Optional real-browser checks. Start the web app before running this script.

    .venv/bin/python tests/browser_smoke.py --url http://127.0.0.1:8000

Requires the browser-test extra and Chrome, or Playwright's installed Chromium.
Screenshots are written outside the repository unless --screenshots is supplied.
"""
import argparse
from pathlib import Path
import tempfile

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--screenshots', type=Path, default=Path(tempfile.gettempdir()) / 'steercurrent-browser')
    parser.add_argument('--channel', default='chrome')
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        failures, submitted, completed = [], [], []
        page.on('pageerror', lambda error: failures.append(str(error)))
        page.on('request', lambda request: submitted.append(request.post_data_json)
                if request.method == 'POST' and request.url.endswith('/api/jobs') else None)

        def collect_result(response):
            if '/api/jobs/' in response.url and response.status == 200:
                data = response.json()
                if data.get('status') == 'done':
                    completed.append(data['result'])

        page.on('response', collect_result)

        def settle_view():
            page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')

        def ready():
            expect(page.locator('#freshness')).to_have_text('Fields up to date', timeout=30000)
            expect(page.locator('#status')).to_have_text('Ready.', timeout=30000)

        def point(x, y):
            box = page.locator('#canvas').bounding_box()
            zoom = min(box['width']/18, box['height']/14)
            return box['x'] + box['width']/2 + (x-6)*zoom, box['y'] + box['height']/2 - y*zoom

        def drag(start, end):
            page.mouse.move(*point(*start))
            page.mouse.down()
            page.mouse.move(*point(*end), steps=10)
            page.mouse.up()

        page.goto(args.url)
        ready()
        expect(page.locator('body')).to_have_class('presentation')
        expect(page.locator('#physics-panel')).to_be_hidden()
        expect(page.locator('#configuration-panel')).to_be_hidden()
        assert page.locator('.stage').bounding_box()['width'] / page.locator('.workspace').bounding_box()['width'] > .75
        before = len(submitted)
        original_width = page.locator('#canvas').bounding_box()['width']
        page.locator('#toggle-controls').click()
        expect(page.locator('#configuration-panel')).to_be_visible()
        settle_view()
        opened_width = page.locator('#canvas').bounding_box()['width']
        assert opened_width < original_width
        page.locator('#toggle-controls').click()
        expect(page.locator('#configuration-panel')).to_be_hidden()
        settle_view()
        assert page.locator('#canvas').bounding_box()['width'] > opened_width
        page.locator('#toggle-physics').click()
        expect(page.locator('#physics-panel')).to_be_visible()
        settle_view()
        assert page.locator('#canvas').bounding_box()['width'] < original_width
        page.locator('#close-physics').click()
        page.locator('#numerical-panel > summary').click()
        expect(page.locator('#numerical-elements')).to_have_text(f"{completed[-1]['summary']['triangles']:,}")
        expect(page.locator('#numerical-vertices')).to_have_text(f"{len(completed[-1]['vertices']):,}")
        expect(page.locator('#numerical-time')).to_have_text(f"{completed[-1]['summary']['seconds']:.3f} s")
        expect(page.locator('#numerical-state')).to_have_text('Complete · current')
        page.locator('#numerical-panel > summary').click()
        settle_view()
        assert len(submitted) == before
        assert page.locator('button.component.selected').get_attribute('data-component') == 'battery'
        expect(page.locator('#story-count')).to_have_text('1 / 7')
        # Each Next click opens the next idea and its corresponding field.
        page.locator('#demo-shape').select_option('u')
        ready()
        assert submitted[-1]['geometry']['shape']['points'][5][1] == 4
        page.locator('#demo-shape').select_option('step')
        ready()
        story_views = ['Current density J', 'Electric field E', 'Battery-only field',
                       'Surface charge σ', 'Power exchange', 'Energy flow S', 'Energy flow S']
        expect(page.locator('#story-explanation')).to_be_hidden()
        expect(page.locator('#story-reveal')).to_be_hidden()
        for step in range(7):
            expect(page.locator('#story-count')).to_have_text(f'{step+1} / 7')
            expect(page.locator('#view-label')).to_have_text(story_views[step])
            if step == 0:
                expect(page.locator('#legend')).to_contain_text('conventional current')
                page.screenshot(path=str(args.screenshots / 'story_current_wire.png'), full_page=True)
            if step:
                if not page.locator('#guide-explanation').evaluate('element => element.open'):
                    page.locator('#guide-explanation > summary').click()
                expect(page.locator('#story-explanation')).to_be_visible()
            if step == 1:
                expect(page.locator('#legend')).to_contain_text('positive charge')
                page.screenshot(path=str(args.screenshots / 'story_electric_wire.png'), full_page=True)
            if step == 2:
                expect(page.locator('#story-conclusion')).to_contain_text('misses some bends')
                expect(page.locator('#story-conclusion')).to_be_hidden()
                page.locator('#story-reveal').click()
                expect(page.locator('#story-conclusion')).to_be_visible()
                expect(page.locator('#story-reveal')).to_be_hidden()
                page.screenshot(path=str(args.screenshots / 'story_battery_only.png'), full_page=True)
                page.locator('[data-story-view="electric"]').click()
                expect(page.locator('#view-label')).to_have_text('Electric field E')
                page.locator('[data-story-view="toy"]').click()
                expect(page.locator('#view-label')).to_have_text('Battery-only field')
                page.locator('#mode-explore').click()
                expect(page.locator('#view-label')).to_have_text('Electric field E')
                expect(page.locator('[data-view="toy"]')).to_have_count(0)
                page.locator('#mode-present').click()
                expect(page.locator('#view-label')).to_have_text('Battery-only field')
            if step == 3:
                expect(page.locator('#story-conclusion')).to_be_hidden()
                expect(page.locator('#story-reveal')).to_be_hidden()
                expect(page.locator('#legend')).to_contain_text('thicker/darker')
                expect(page.locator('#legend')).to_contain_text('magenta +')
                page.screenshot(path=str(args.screenshots / 'story_charge_density.png'), full_page=True)
            if step == 4:
                expect(page.locator('#legend')).to_contain_text('absorption')
            if step == 6:
                expect(page.locator('#view-label')).to_have_text('Energy flow S')
                # The comparison controls stay visible even when the equation is expanded.
                assert page.locator('#story-views').evaluate('''element => {
                    const body = element.closest('.story-body').getBoundingClientRect();
                    const controls = element.getBoundingClientRect();
                    return controls.top >= body.top && controls.bottom <= body.bottom;
                }''')
                for field, label in [('electric', 'Electric field E'),
                                     ('magnetic', 'Magnetic field Bz'),
                                     ('energy', 'Energy flow S')]:
                    button = page.locator(f'[data-story-view="{field}"]')
                    button.click()
                    expect(button).to_have_class('active')
                    expect(page.locator('#view-label')).to_have_text(label)
                page.screenshot(path=str(args.screenshots / 'story_energy.png'), full_page=True)
            else:
                page.locator('#story-next').click()
        page.locator('#mode-explore').click()
        expect(page.locator('body')).to_have_class('explore')
        page.locator('#toggle-controls').click()
        expect(page.locator('#shape')).to_be_visible()
        expect(page.locator('#field-select option')).to_have_count(8)
        expect(page.locator('#field-select optgroup')).to_have_count(3)
        page.locator('#toggle-physics').click()
        for view in ['potential', 'electric', 'current', 'current_strength', 'charge', 'power_exchange', 'magnetic', 'energy']:
            page.locator('#field-select').select_option(view)
            if view == 'current_strength':
                expect(page.locator('#lesson-title')).to_have_text('What is current crowding?')
                expect(page.locator('#legend')).to_contain_text('Red = current crowding above even flow')
                expect(page.locator('#formula msqrt')).to_have_count(1)
                expect(page.locator('#formula mover')).to_have_count(1)
                expect(page.locator('#formula .equation-terms dt .vector-symbol')).to_have_count(1)
            expect(page.locator('#formula math')).to_have_count(1)
            expect(page.locator('#formula .equation-meaning')).not_to_be_empty()
            expect(page.locator('#formula .equation-terms dt').first).not_to_be_empty()
            if view == 'power_exchange':
                expect(page.locator('#resistor-readout-row')).to_be_visible()
                expect(page.locator('#resistor-readout')).not_to_have_text('—')
            else:
                expect(page.locator('#resistor-readout-row')).to_be_hidden()
            page.screenshot(path=str(args.screenshots / f'{view}.png'), full_page=True)
        page.locator('[data-component="R1"]').click()
        before_current = completed[-1]['summary']['current']
        page.locator('#resistivity').focus()
        page.locator('#resistivity').press('End')
        ready()
        assert submitted[-1]['geometry']['config']['resistors'][0]['resistivity'] == 60
        assert completed[-1]['summary']['current'] < before_current
        page.locator('#undo').click()
        ready()
        page.locator('#resolution').select_option('fine')
        ready()
        fine_count = completed[-1]['summary']['triangles']
        expect(page.locator('#numerical-resolution')).to_have_text('Finer mesh')
        page.locator('#resolution').select_option('quick')
        ready()
        assert completed[-1]['summary']['triangles'] < fine_count
        before = len(submitted)
        page.locator('#fit').click()
        settle_view()
        original_image = page.locator('#canvas').evaluate('element => element.toDataURL()')
        page.locator('#show-mesh').check()
        assert page.locator('#canvas').evaluate('element => element.toDataURL()') != original_image
        page.locator('#show-mesh').uncheck()
        page.locator('#show-handles').uncheck()
        assert page.locator('#canvas').evaluate('element => element.toDataURL()') != original_image
        page.locator('#show-handles').check()
        page.locator('#zoom-in').click()
        assert page.locator('#canvas').evaluate('element => element.toDataURL()') != original_image
        page.locator('#zoom-out').click()
        page.locator('#fit').click()
        assert page.locator('#canvas').evaluate('element => element.toDataURL()') == original_image
        box = page.locator('#canvas').bounding_box()
        page.mouse.move(box['x'] + 30, box['y'] + 50)
        page.mouse.down()
        page.mouse.move(box['x'] + 60, box['y'] + 70, steps=5)
        page.mouse.up()
        assert page.locator('#canvas').evaluate('element => element.toDataURL()') != original_image
        page.mouse.wheel(0, -120)
        page.locator('#fit').click()
        assert len(submitted) == before
        page.locator('#close-physics').click()
        page.locator('#toggle-controls').click()
        settle_view()
        page.locator('#fit').click()
        drag((7, 4), (7.3, 4))
        ready()
        assert submitted[-1]['geometry']['config']['resistors'][0]['start'] > 10.1
        page.locator('#undo').click()
        ready()
        page.locator('#toggle-controls').click()
        page.locator('#toggle-physics').click()
        settle_view()
        # Drag the selected resistor along the upper wire, away from its end dots.
        page.locator('#fit').click()
        drag((7, 4), (7.35, 4))
        ready()
        moved = submitted[-1]['geometry']
        assert moved['config']['resistors'][0]['start'] > 10.2
        assert moved['config']['length'] == 40
        page.locator('#undo').click()
        ready()
        assert submitted[-1]['geometry']['config']['resistors'][0]['start'] == 10

        # Numeric resistor edits clear turns just as dragging does.
        page.locator('[data-component="R2"]').click()
        page.locator('#resistor-start').fill('24')
        page.locator('#resistor-start').press('Tab')
        ready()
        expect(page.locator('#error')).to_be_hidden()
        r2 = submitted[-1]['geometry']['config']['resistors'][1]
        assert r2['start'] < 23.5 and r2['end'] < 27.5
        page.locator('#undo').click()
        ready()
        page.locator('#resistor-length').fill('10')
        page.locator('#resistor-length').press('Tab')
        ready()
        expect(page.locator('#error')).to_be_hidden()
        r2 = submitted[-1]['geometry']['config']['resistors'][1]
        assert r2['start'] == 26 and r2['end'] > 36.5
        page.locator('#undo').click()
        ready()
        page.locator('[data-component="R1"]').click()

        # Shape handles resize the loop; resistor ends follow the wire.
        original = submitted[-1]['geometry']['config']
        for name, anchor, direction, delta in [
            ('top', (4,4), (0,1), (0,.3)),
            ('bottom', (6,-4), (0,-1), (0,-.3)),
            ('right', (12,-2), (1,0), (.3,0)),
            ('notch', (8,0), (1,1), (-.3,-.3)),
        ]:
            page.locator('#fit').click()
            box = page.locator('#canvas').bounding_box()
            zoom = min(box['width']/18, box['height']/14)
            offset = .5 + 16/zoom
            start = (anchor[0]+direction[0]*offset, anchor[1]+direction[1]*offset)
            before = len(submitted)
            drag(start, (start[0]+delta[0], start[1]+delta[1]))
            ready()
            assert len(submitted) > before, f'{name} handle did not submit an edit'
            changed = submitted[-1]['geometry']
            expected_length = 40 if name == 'notch' else 40.6
            assert abs(changed['config']['length']-expected_length) < .15, (name, changed['config']['length'])
            assert changed['config']['battery'] == original['battery']
            assert all(r['end'] > r['start'] for r in changed['config']['resistors'])
            points = changed['shape']['points']
            if name == 'top': assert points[1][1] > 4.2
            if name == 'bottom': assert points[-1][1] < -4.2
            if name == 'right': assert points[4][0] > 12.2
            if name == 'notch': assert points[3][0] < 7.8
            page.locator('#undo').click()
            ready()

        # An overlarge drag stops at the first allowed corner position, without an error.
        page.locator('#fit').click()
        box = page.locator('#canvas').bounding_box()
        zoom = min(box['width']/18, box['height']/14)
        start = (8 + .5 + 16/zoom, .5 + 16/zoom)
        drag(start, (start[0], start[1]+5))
        ready()
        stopped = submitted[-1]['geometry']['shape']['points'][3][1]
        assert 0 < stopped < 4
        expect(page.locator('#error')).to_be_hidden()
        page.locator('#undo').click()
        ready()

        # A new shape starts fresh; returning to it restores the last edit.
        original = submitted[-1]['geometry']['config']
        saved_shapes = {}
        for kind in ['u', 'stairs', 'step']:
            page.locator('#shape').select_option(kind)
            ready()
            assert submitted[-1]['geometry']['config'] == original
            assert len(submitted[-1]['geometry']['shape']['points']) == (7 if kind == 'step' else 9)
            if kind != 'step':
                # Exercise a new corner handle through its accessible numeric control.
                handle = 'indent left' if kind == 'u' else 'upper corner'
                control = page.locator(f'input[data-handle="{handle}"][data-axis="1"]')
                page.locator('summary').filter(has_text='How shape handles work').click()
                control.fill(str(float(control.input_value()) + .04))
                control.press('Tab')
                ready()
                edited = submitted[-1]['geometry']['config']
                assert edited['battery'] == original['battery']
                assert all(r['end'] > r['start'] for r in edited['resistors'])
                assert abs(edited['length']-(39.92 if kind == 'u' else 40)) < 1e-8
                saved_shapes[kind] = submitted[-1]['geometry']
                page.screenshot(path=str(args.screenshots / f'{kind}.png'), full_page=True)
                page.locator('summary').filter(has_text='How shape handles work').click()
        for kind in ['u', 'stairs', 'step']:
            page.locator('#shape').select_option(kind)
            ready()
            if kind in saved_shapes:
                assert submitted[-1]['geometry'] == saved_shapes[kind]
            else:
                assert submitted[-1]['geometry']['config'] == original

        # Automatic mode has no competing manual action.
        expect(page.locator('#calculate')).to_be_disabled()
        page.locator('#auto').uncheck()
        expect(page.locator('#calculate')).to_be_enabled()
        page.locator('#auto').check()
        expect(page.locator('#calculate')).to_be_disabled()

        # Invalid component overlap leaves the accepted circuit and fields intact.
        before = len(submitted)
        page.locator('#resistor-start').fill('0')
        page.locator('#resistor-start').press('Tab')
        expect(page.locator('#error')).to_be_visible()
        expect(page.locator('#resistor-start')).to_have_value('10.00')
        assert len(submitted) == before

        page.locator('[data-component="battery"]').click()
        page.locator('#reverse').click()
        ready()
        assert submitted[-1]['geometry']['config']['battery']['polarity'] == -1
        page.locator('#shape').select_option('rectangle')
        ready()
        assert len(submitted[-1]['geometry']['shape']['points']) == 5
        assert submitted[-1]['geometry']['config'] == original
        # Switching away from a resized rectangle restores the saved U loop.
        page.locator('#fit').click()
        box = page.locator('#canvas').bounding_box()
        zoom = min(box['width']/18, box['height']/14)
        drag((12 + .5 + 16/zoom, 0), (6 + .5 + 16/zoom, 0))
        ready()
        page.locator('[data-component="R1"]').click()
        page.locator('#resistor-start').fill('4.6')
        page.locator('#resistor-start').press('Tab')
        ready()
        page.locator('#shape').select_option('u')
        ready()
        expect(page.locator('#error')).to_be_hidden()
        assert submitted[-1]['geometry'] == saved_shapes['u']
        for _ in range(3):
            page.locator('#undo').click()
            ready()
        expect(page.locator('#shape')).to_have_value('rectangle')
        page.locator('#add-resistor').click()
        ready()
        assert len(submitted[-1]['geometry']['config']['resistors']) == 3
        page.locator('#remove').click()
        ready()
        assert len(submitted[-1]['geometry']['config']['resistors']) == 2

        # A changed model hides old fields until explicitly recalculated.
        page.locator('#auto').uncheck()
        page.locator('[data-component="battery"]').click()
        page.locator('#reverse').click()
        expect(page.locator('#freshness')).to_have_text('Edits not calculated')
        expect(page.locator('#numerical-state')).to_have_text('Previous result · stale')
        expect(page.locator('#current-readout')).to_have_text('—')
        page.locator('#calculate').click()
        ready()

        # Separate browser sessions start with their own untouched example.
        other = browser.new_page()
        other.goto(args.url)
        expect(other.locator('#freshness')).to_have_text('Fields up to date', timeout=30000)
        expect(other.locator('#shape')).to_have_value('step')
        other.close()

        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(args.screenshots / 'mobile.png'), full_page=True)
        assert not page.evaluate('() => document.documentElement.scrollWidth > innerWidth')
        page.locator('#mode-present').click()
        expect(page.locator('#story-count')).to_have_text('7 / 7')
        expect(page.locator('#view-label')).to_have_text('Energy flow S')
        page.locator('#reset').click()
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        ready()
        expect(page.locator('#story-count')).to_have_text('1 / 7')
        expect(page.locator('#demo-shape')).to_have_value('step')
        page.screenshot(path=str(args.screenshots / 'story_mobile.png'), full_page=True)
        assert not page.evaluate('() => document.documentElement.scrollWidth > innerWidth')
        # New loop replaces one saved shape; Reset demo forgets every shape.
        page.set_viewport_size({'width': 1440, 'height': 1000})
        page.locator('#mode-explore').click()
        page.locator('#auto').uncheck()
        page.locator('#shape').select_option('u')
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        page.locator('#loop-length').fill('44')
        page.locator('#new-loop').click()
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        expect(page.locator('#components .component')).to_have_count(2)
        page.locator('#shape').select_option('step')
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        page.locator('#shape').select_option('u')
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        expect(page.locator('#loop-length')).to_have_value('44')
        expect(page.locator('#components .component')).to_have_count(2)
        page.locator('#reset').click()
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        ready()
        page.locator('#auto').uncheck()
        page.locator('#shape').select_option('u')
        expect(page.locator('#status')).to_contain_text('Circuit updated.')
        expect(page.locator('#loop-length')).to_have_value('40')
        expect(page.locator('#components .component')).to_have_count(3)
        page.locator('#auto').check()
        ready()
        before = len(submitted)
        for width, height in [(1280, 720), (1024, 768), (760, 900), (390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            page.locator('#mode-present').click()
            if page.locator('#toggle-physics').get_attribute('aria-expanded') == 'true':
                page.locator('#toggle-physics').click()
            if page.locator('#toggle-controls').get_attribute('aria-expanded') == 'true':
                page.locator('#toggle-controls').click()
            page.locator('#guide-panel > summary').click()
            settle_view()
            expanded_height = page.locator('#canvas').bounding_box()['height']
            page.locator('#guide-panel > summary').click()
            page.locator('#numerical-panel > summary').click()
            settle_view()
            assert page.locator('#canvas').bounding_box()['height'] >= 230
            assert page.locator('#canvas').bounding_box()['height'] <= expanded_height
            page.locator('#story-next').scroll_into_view_if_needed()
            expect(page.locator('#story-next')).to_be_visible()
            page.locator('#numerical-panel > summary').click()
            page.locator('#toggle-controls').click()
            page.locator('#toggle-physics').click()
            expect(page.locator('#configuration-panel')).to_be_visible()
            expect(page.locator('#physics-panel')).to_be_visible()
            assert not page.evaluate('() => document.documentElement.scrollWidth > innerWidth')
            page.screenshot(path=str(args.screenshots / f'layout_{width}.png'), full_page=True)
        assert len(submitted) == before
        assert not failures, failures
        print(f'Browser checks passed. Screenshots: {args.screenshots}')
        browser.close()


if __name__ == '__main__':
    main()
