"""Verify the complete handoff and original runtime using only Python stdlib."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(read, names):
    manifest = json.loads(read('HANDOFF_MANIFEST.json'))
    listed = set()
    for row in manifest['files']:
        p = PurePosixPath(row['path'])
        if p.is_absolute() or '..' in p.parts or row['path'] in listed:
            raise ValueError('Invalid or duplicate manifest path')
        raw = read(row['path'])
        assert len(raw) == row['bytes'] and digest(raw) == row['sha256'], row['path']
        listed.add(row['path'])
    assert set(names) == listed | {'HANDOFF_MANIFEST.json'}, 'Extra/missing files'
    prefix = '01_ARM实机程序/runtime/'
    frozen_raw = read(prefix + 'manifest.json')
    assert digest(frozen_raw) == '09c4795d9457f6aff117643234edb6c7fcdd12395a5d7268ec11c32aba0005dd'
    frozen = json.loads(frozen_raw)
    for row in frozen['files'] + frozen['models']:
        assert digest(read(prefix + row['path'])) == row['sha256'], row['path']
    return dict(verified=True, files=len(names), frozen_sources=len(frozen['files']),
                models=len(frozen['models']), inference_started=False, hardware_accessed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            prefix = 'ARM_最终交接/'
            items = [i for i in archive.infolist() if not i.is_dir()]
            assert len({i.filename for i in items}) == len(items)
            for i in items:
                p = PurePosixPath(i.filename)
                assert not p.is_absolute() and '..' not in p.parts and i.filename.startswith(prefix)
            result = verify(lambda name: archive.read(prefix + name),
                            [i.filename[len(prefix):] for i in items])
    else:
        root = args.root.resolve()
        paths = [p for p in root.rglob('*') if p.is_file()]
        assert all(root in p.resolve().parents for p in paths)
        result = verify(lambda name: (root / name).read_bytes(),
                        [p.relative_to(root).as_posix() for p in paths])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
