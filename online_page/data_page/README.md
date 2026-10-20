# Motion Data Atlas

The web app and service code live in online_page/data_page.
Canonical HumanML3D, SONIC, and MotionX files stay under dataset.
No dataset media or SQLite index belongs in Git or GitHub Pages.

## Boundaries

- server.py serves the read-only API for samples, text, motion, video,
  summary statistics, and derived SMPL-H meshes.
- static/ contains the full Motion Data Atlas and Viewer pages.
- build_index.py builds the private SQLite index from dataset metadata.
- dataset/data_page/data holds index.sqlite, summary.json, and
  task_stats.json on the server.
- runtime/cloudflare-viewer/deps holds server-only Python dependencies.

GitHub Pages publishes the public summary snapshot in dataset-page.
tools/site/build_public_site.py uses an explicit static allowlist and
excludes this dynamic app. The full Viewer needs its Python API.

## Run

The existing server entry point is:
  /wangbenyou-sulongjie/caimeng/runtime/cloudflare-viewer/start-public-viewer.sh

It starts the service on 127.0.0.1:8765 and sets these environment
variables:

- MOTION_DATA_ROOT: canonical dataset root.
- MOTION_DATA_INDEX_DIR: private SQLite and summary JSON directory.
- MOTION_DATA_PAGE_DEPS: server Python dependency directory.
- MOTION_SMPLH_MODEL: licensed SMPL-H model file.

If the index is missing, run build_index.py with --data-root and
--output-dir pointing at the private dataset locations. The index stores
dataset paths and must not be copied into the public site.

On viewer.caimeng.online, the root path redirects to /viewer. Other hosts
retain the documentation page at the root. /viewer serves the full browser,
and /health provides the health check. The app binds only to loopback; the
existing tunnel configuration handles external access.

## Viewer controls and root motion

The public viewer is https://viewer.caimeng.online/viewer.

- The default world view preserves horizontal root displacement relative to
  the first frame, root height changes, and yaw. The previous viewer subtracted
  the current root X/Z every frame and therefore hid horizontal translation.
- In-place view hides only horizontal translation. Both modes use one floor
  offset for the entire clip, so vertical root motion remains visible.
- The trajectory is projected onto the ground. Path length is cumulative
  horizontal travel; net displacement is the first-to-last horizontal distance.
  Small root jitter can accumulate, especially in long stationary clips.
- The 263D input has already been preprocessed. Coordinates are not absolute
  positions in the original camera scene.
- The SMPL-H API returns meshes centered horizontally, with height and yaw
  already applied. The frontend adds only the matching frame's X/Z displacement
  in world mode. Mesh playback is sampled at approximately 5 FPS.
- Whole-clip fit and front/side/top presets help inspect travel. Camera follow
  keeps the body in view while the ground trajectory still moves relative to it.

The sample sidebar supports ID search, media filters, 40-item pages, page
numbers, previous/next sample across pages, and random selection across all
filtered results. Copied links preserve dataset, sample, page, search, filter,
and root mode. Text is searchable and expandable; frame/event labels seek the
motion. Videos open only when present and play independently because their
time alignment has not been verified.

Shortcuts outside text inputs: Space plays/pauses, arrow keys step frames,
J/K select the next/previous sample, `/` focuses search, and Escape closes
expanded text. Native controls retain their own Space behavior.

## Frontend checks

From the repository root:

```sh
node --test online_page/data_page/tests/*.test.mjs
node --check online_page/data_page/static/viewer.js
node --check online_page/data_page/static/viewer-state.mjs
git diff --check
```

Coordinate transforms and navigation live in `static/viewer-state.mjs`.
They never modify the decoded source buffer. Static files can be deployed to
the existing service without restarting the backend or rebuilding its index.

Requests include bounded retries for transient network/server failures and a
timeout that covers reading the response body. Switching samples cancels the
previous request. Persistent failures show the HTTP/error detail and an in-page
retry button. Module initialization failures are reported separately from data
service failures, so the page does not silently remain in the connecting state.
