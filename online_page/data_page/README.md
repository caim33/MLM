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

The root path serves the documentation page. /viewer serves the full
browser, and /health provides the health check. The app binds only to
loopback; the existing tunnel configuration handles external access.
