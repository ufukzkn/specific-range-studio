"""Export core runtime sources and a verified bundle, without workstation environments."""
from pathlib import Path
import argparse
import json
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.research_bundle import sha256, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/research_bundle_portable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_bundle(args.bundle)
    if args.output.exists():
        parser.error("Choose a new output file; existing packages are preserved")
    entries = {}
    for folder in ["src", "scripts", "docs", "deploy", "tests"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                entries[str(path.relative_to(ROOT).as_posix())] = path
    for name in ["README.md", ".gitignore", ".dockerignore", "requirements-edge.txt", "requirements.txt",
                 "One_Engine_Data.xlsx", "Two_Engine_Data.xlsx"]:
        entries[name] = ROOT / name
    for path in args.bundle.rglob("*"):
        if path.is_file():
            entries["artifacts/research_bundle_portable/" + path.relative_to(args.bundle).as_posix()] = path
    manifest = {"files": {name: sha256(path) for name, path in sorted(entries.items())},
                "notes": ["Core runtime export; no .git history or workstation virtual environment.",
                          "Dataset Builder demo assets and archived Raporlar omitted.",
                          "Included models are labelled in the bundle manifest; check before scientific use."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC error: {bad}")
    print(json.dumps({"output": str(args.output.resolve()), "file_count": len(entries)+1,
                      "bytes": args.output.stat().st_size, "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
