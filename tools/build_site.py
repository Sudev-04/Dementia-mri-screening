"""Build the server-less demo site (for GitHub Pages or any static host).

    python tools/build_site.py site

The result runs entirely in the visitor's browser: screening is scored by the same
JavaScript the offline app uses, and scans are measured locally by scanlocal.js.
There is no sign-in, no database and no upload.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.measure import load_reference  # noqa: E402
from common.scoring import load_bundle  # noqa: E402


def build(out: Path) -> Path:
    static = ROOT / "app" / "static"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(static, out, ignore=shutil.ignore_patterns("package.json"))

    index = (out / "index.html").read_text(encoding="utf-8")
    index = index.replace('<html lang="en">', '<html lang="en" data-static="1">').replace('"/static/', '"')
    assert 'data-static="1"' in index and "/static/" not in index
    (out / "index.html").write_text(index, encoding="utf-8")

    manifest = json.loads((out / "manifest.webmanifest").read_text(encoding="utf-8"))
    manifest.update(start_url="./", scope="./")
    for icon in manifest["icons"]:
        icon["src"] = icon["src"].replace("/static/", "")
    (out / "manifest.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sw = (out / "sw.js").read_text(encoding="utf-8")
    sw = sw.replace('"/static/', '"').replace('  "/",', '  "./",').replace('"/api/bundle"', '"bundle.json", "reference.json"')
    sw = sw.replace("alzscreen-${VERSION}", "alzscreen-site-${VERSION}")
    assert "/static/" not in sw and '"./"' in sw
    (out / "sw.js").write_text(sw, encoding="utf-8")

    (out / "bundle.json").write_text(json.dumps(load_bundle()), encoding="utf-8")
    (out / "reference.json").write_text(json.dumps(load_reference()), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out


if __name__ == "__main__":
    target = build(Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve())
    print(f"built {target} ({sum(1 for p in target.rglob('*') if p.is_file())} files)")
