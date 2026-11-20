"""Build a separate, hash-verified inference bundle without training checkpoints."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = [ROOT / 'artifacts/baseline.joblib', ROOT / 'artifacts/neural.joblib']
    for folder in ('colab/bert_model', 'artifacts/student_int8'):
        files.extend(p for p in sorted((ROOT / folder).iterdir()) if p.is_file()
                     and p.name not in {'training_args.bin', 'training_report.json', 'reload_check.json'})
    manifest = {'format_version': 1, 'files': {}}
    for path in files:
        manifest['files'][path.relative_to(ROOT).as_posix()] = {
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT / 'docs').mkdir(exist_ok=True)
    manifest_path = ROOT / 'docs/model_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    output = ROOT / 'release'
    output.mkdir(exist_ok=True)
    bundle = output / 'support-router-models.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        for name, item in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item['sha256'], name
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (output / 'support-router-models.zip.sha256').write_text(f'{digest}  {bundle.name}\n')
    print(f'Verified {len(files)} files; {bundle.stat().st_size / 1e6:.1f} MB: {bundle}')

if __name__ == '__main__':
    main()
