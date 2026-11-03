"""Verify the entire unpacked ARM package using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            members = archive.namelist()
            assert len(members) == len(set(members)), "Duplicate ZIP entries"
            manifests = [n for n in members if n.endswith("/MANIFEST.json")]
            assert len(manifests) == 1
            prefix = manifests[0][:-len("MANIFEST.json")]
            manifest = json.loads(archive.read(manifests[0]))
            expected = {prefix+r["path"] for r in manifest["files"]} | {manifests[0]}
            assert set(members) == expected, "Unexpected/missing ZIP entries"
            for record in manifest["files"]:
                name = prefix+record["path"]
                data = archive.read(name)
                assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], name
        print(json.dumps(dict(verified=True, files=len(manifest["files"]), archive=str(args.zip))))
    else:
        root = args.root.resolve()
        manifest = json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
        for record in manifest["files"]:
            path = (root/record["path"]).resolve()
            assert root in path.parents and path.is_file(), record["path"]
            assert path.stat().st_size == record["bytes"], record["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["path"]
        print(json.dumps(dict(verified=True, files=len(manifest["files"]), root=str(root))))


if __name__ == "__main__":
    main()
