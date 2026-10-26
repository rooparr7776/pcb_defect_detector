#!/usr/bin/env python3
"""
Audit NanoNetsPCBChecker images and XML annotations.

Run from the folder containing NanoNetsPCBChecker:
    python3 inspect_nanonets_dataset.py
or:
    python3 inspect_nanonets_dataset.py /path/to/NanoNetsPCBChecker/DLAssignment

This script does NOT train a model and does NOT use XML annotations to make
detector predictions. It only helps identify clean reference candidates and
known defect examples for evaluation.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip()
        try:
            rotated = int(float(obj.findtext("rotated", default="0") or 0))
        except ValueError:
            rotated = 0
        objects.append((name, rotated))
    return objects


def main():
    default_root = Path(__file__).resolve().parent / "NanoNetsPCBChecker" / "DLAssignment" / "DLAssignment" / "errors"
    root = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_root

    if not root.exists():
        print(f"ERROR: Dataset folder not found:\n  {root}")
        print("Pass the correct DLAssignment path as the first argument.")
        sys.exit(1)

    rows = []
    for image in sorted(root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        if "errors" in {p.lower() for p in image.parts}:
            continue
        xml_path = image.with_suffix(".xml")
        if not xml_path.exists():
            continue
        try:
            objects = read_xml(xml_path)
        except (ET.ParseError, OSError) as exc:
            print(f"WARNING: Could not read {xml_path.name}: {exc}")
            continue

        names = {name for name, _ in objects}
        rotated = sorted(name for name, flag in objects if flag == 1)
        errors_dir_image = root / "errors" / image.name
        # Some clones may place errors/ in a nested directory.
        marked_matches = list(root.rglob("errors/" + image.name))
        marked_exists = bool(marked_matches) or errors_dir_image.exists()

        rows.append({
            "image": image,
            "xml": xml_path,
            "count": len(objects),
            "names": names,
            "rotated": rotated,
            "marked": marked_exists,
        })

    if not rows:
        print(f"No image/XML pairs found under: {root}")
        sys.exit(2)

    print(f"\nDataset folder: {root}")
    print(f"Annotated image/XML pairs found: {len(rows)}")
    print("\nIMAGE SUMMARY")
    print("-" * 100)
    for row in rows:
        flags = []
        if row["count"] == 60 and not row["rotated"]:
            flags.append("CLEAN-REFERENCE CANDIDATE (verify visually)")
        if row["count"] < 60:
            flags.append(f"FEWER THAN 60 BOXES ({60 - row['count']} fewer; possible missing components)")
        if row["rotated"]:
            flags.append("ROTATED=" + ",".join(row["rotated"]))
        if row["marked"]:
            flags.append("matching image found under errors/ (visual aid only)")
        note = " | ".join(flags) if flags else "no simple flag"
        print(f"{row['image'].name:30} objects={row['count']:2}  {note}")

    # Select a likely clean reference by annotation count and rotated flags.
    candidates = [r for r in rows if r["count"] == 60 and not r["rotated"]]
    if not candidates:
        print("\nNo image with exactly 60 objects and zero rotated flags was found.")
        print("Do not guess a reference; inspect the XMLs and images manually.")
        return

    reference = candidates[0]
    print("\nPAIRWISE DEFECT HINTS")
    print("-" * 100)
    print(f"Using candidate reference: {reference['image'].name}")
    print("IMPORTANT: this is only a candidate. Open the image and confirm it is defect-free.")
    for row in rows:
        if row is reference:
            continue
        missing = sorted(reference["names"] - row["names"])
        extra = sorted(row["names"] - reference["names"])
        if missing or extra or row["rotated"]:
            print(f"\nTest: {row['image'].name}")
            print(f"  Names absent relative to reference: {', '.join(missing) if missing else 'none'}")
            print(f"  Extra/different names: {', '.join(extra) if extra else 'none'}")
            print(f"  XML rotated flags: {', '.join(row['rotated']) if row['rotated'] else 'none'}")
            if row["marked"]:
                print("  A matching errors/ visualization exists; inspect it, but don't use it as detector input.")

    print("\nNEXT STEP")
    print("1. Visually verify the clean-reference candidate.")
    print("2. Use that image as Reference PCB and a different image as Test PCB.")
    print("3. Compare predictions against XML only after the detector has made predictions.")
    print("4. Do not report accuracy from one or two images; collect multiple clean and defective cases.")


if __name__ == "__main__":
    main()
