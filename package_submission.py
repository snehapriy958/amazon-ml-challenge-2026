"""Submission Packaging Utility for Amazon ML Challenge 2026.
Builds the exact required submission ZIP structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import argparse
from pathlib import Path
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge submission ZIP.")
    parser.add_argument("--team-name", type=str, required=True, help="Official team name for submission.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the submission zip.")
    args = parser.parse_args()

    team_name = args.team_name.strip()
    zip_filename = Path(args.output_dir) / f"{team_name}_submission.zip"

    print("=" * 75)
    print(f"=== PACKAGING SUBMISSION ARCHIVE: {zip_filename.name} ===")
    print("=" * 75)

    # Required files and directories
    matching_file = Path("output/matching_results.tsv")
    candidate_file = Path("output/candidate_pairs.tsv")
    doc_file = Path("Documentation_template.md")
    code_dir = Path("code/business_entity_resolution")

    assert matching_file.exists(), f"Missing required file: {matching_file}"
    assert candidate_file.exists(), f"Missing required file: {candidate_file}"
    assert doc_file.exists(), f"Missing required file: {doc_file}"
    assert code_dir.exists(), f"Missing required directory: {code_dir}"

    t0 = time.time()
    print(f"Creating {zip_filename} with ZIP_DEFLATED (level 6)...", flush=True)

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Output files
        print(f"  Adding {matching_file} ({matching_file.stat().st_size / (1024**2):.1f} MB)...", flush=True)
        zf.write(matching_file, arcname="output/matching_results.tsv")

        print(f"  Adding {candidate_file} ({candidate_file.stat().st_size / (1024**3):.2f} GB)...", flush=True)
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")

        # 2. Documentation template
        print(f"  Adding {doc_file}...", flush=True)
        zf.write(doc_file, arcname="Documentation_template.md")

        # 3. Code package
        print(f"  Adding {code_dir} source files...", flush=True)
        for path in code_dir.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                rel_path = path.relative_to(code_dir.parent.parent)
                print(f"    + {rel_path}", flush=True)
                zf.write(path, arcname=str(rel_path).replace("\\", "/"))

    zip_size_gb = zip_filename.stat().st_size / (1024 ** 3)
    print("=" * 75)
    print(f"SUCCESS: Package created in {time.time()-t0:.2f}s!")
    print(f"Archive: {zip_filename.resolve()}")
    print(f"Archive Size: {zip_size_gb:.2f} GB")
    print("=" * 75)


if __name__ == "__main__":
    main()
