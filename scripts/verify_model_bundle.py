"""Check extracted model files against the repository manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest_path = ROOT / 'docs/model_manifest.json'
    if not manifest_path.exists():
        raise SystemExit('Missing docs/model_manifest.json. Obtain the complete project release.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f'Missing: {name}')
        elif path.stat().st_size != expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            failures.append(f'Hash mismatch: {name}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified all {len(manifest['files'])} model files.")

if __name__ == '__main__':
    main()
