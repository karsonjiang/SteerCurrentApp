# Local web app

From the repository root, using the existing Python environment:

```bash
.venv/bin/python -m steercurrent
```

Open **http://127.0.0.1:8000**. Stop the server with Ctrl+C. An alternative port
can be selected with `--port 8001`. After installing the project, the equivalent
entry point is `.venv/bin/steercurrent-web`.
No additional runtime Python package is needed.

## Automatic calculation and mesh resolution

The demo calculates automatically on opening. Disable **Automatically update
fields** to make several changes before calculating. Quick and finer meshes
have fixed resolution limits for interactive use.

## Modes

### Guided

The app opens in **Guided** mode. Seven steps show current, the electric field,
a battery-only field, surface charge, power exchange, and energy flow. The
walkthrough is docked below the canvas; collapse **Guided walkthrough** to give
the simulation more space. Scenes 3, 4, and 7 have comparison buttons for
switching between relevant fields on the same geometry.

### Explore

Choose **Explore** at any point to select a field without resetting the circuit.
The compact **View** selector groups all eight views into Fields, Transport,
and Materials & interactions. In addition to showing the same views as **Guided**
mode, the selector shows potential and current strength, which highlights current
hotspots. Returning to Guided mode re-enables automatic calculation.

## Shared workspace

The circuit canvas stays in place across modes. The shape selector, drag handles,
and resistor dragging work in both modes and at every Guided step. The
**Circuit parameters** toolbar button opens or collapses the configuration
panel in either mode.
**Physics & equations** beside the active view opens the explanation panel,
which starts collapsed. It contains the full scientific explanation, equations
and symbol definitions, current simulation readouts, and expandable experiments
and model information.

Drag empty space to pan, scroll to zoom, or use the zoom buttons. **Fit view**
resets only the camera. Toggle the mesh or editing handles independently. Move
the pointer over the circuit to inspect field values.

Potential and magnetic color maps interpolate nodal values linearly within each
triangle. Vector arrows show sampled triangle values; their direction is
physical, their length is fixed, and their brightness indicates relative
magnitude on a logarithmic scale. Energy arrows use triangle-average S. Colors
and arrow brightness are normalized separately for each calculation, so use
probe values when comparing strengths. All values use the existing model's
units, with integrated current and power per unit depth.

## Circuit editing

In the configuration panel, select the battery or a resistor. Drag a resistor
body to move it along the loop; drag its end dots to change its length. Numeric
controls provide the same operations. Resistors can span corners, provided
their ends have sufficient clearance.

The first visit to each loop shape starts a fresh 40-unit circuit. Switching
back restores that shape's latest accepted edits. Green circular handles
outside the wire reshape rectangular, L-shaped, U-shaped, and stepped loops.
Their arrows show the permitted movement; hover over a handle for a hint:

- The top and bottom handles change the loop's height and total length.
- The right handle changes its width and total length.
- The L-shaped and stepped loops' inside corners move while keeping their turns.
- The U-shaped loop's corner handles change indentation width and depth.
  Changing the depth also changes the loop length.

The battery remains straight at `(0, 0)`. Dragging a handle can change the
loop length. Each resistor end stays on its original wire section as that
section moves or stretches, so the resistor's length may change. The edited
loop stays within its chosen shape family. A handle stops at the nearest allowed
position. This editor supports four shape families and loop lengths 24–80.

**New loop** replaces the selected shape's saved circuit at the entered starting
length and resets its components. **Reset example** clears all saved shapes and
restores the default L-shaped loop. **Undo** restores the previous
accepted geometry or material setting. Mesh-detail and display changes are not
part of geometry undo history.

The server accepts up to six resistors, loop lengths 24–80, and bounded
material parameters. Detailed editing instructions are under **Geometry help**,
**Interaction help**, and the selected component's help section.

## How it is connected

- `simulation.py` validates browser inputs, calls existing meshing/calculation
  functions, and exports finite arrays plus summary values for the readouts.
- `server.py` serves local static assets (under `web/static`) and JSON endpoints
  (e.g., `/api/jobs` to start a calculation).
  Several people can open the app and edit their own circuits at the same time.
  The server handles their browser requests concurrently and keeps calculation results under separate job IDs.
  One dedicated worker process runs calculations one at a time, isolating Gmsh's global state.
  Up to three calculations can wait in the queue; after that, a new request gets a “queue is full” response.
  This is designed for a small demo, and users may notice a wait if several calculate at once.
- `static/geometry.js` contains the browser's shape and component editing rules.
- `static/app.js` handles interaction, rendering, and result revisions.

`POST /api/validate` checks an edit without meshing. `POST /api/jobs` validates
and queues a calculation; `GET /api/jobs/<id>` polls its status/result.
At most three jobs wait behind the running job. Completed jobs are bounded in
memory and expire after 15 minutes. Each browser owns its configuration and
receives its result through an unguessable job ID.

## Testing

See the [test guide](../../../tests/README.md).
