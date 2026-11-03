"""Prepare reviewed type/shape trials without altering earlier sealed datasets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from PIL import Image

from prepare_domestic_direction_pycharm import ROOT, digest, save

OUT = ROOT / "runs/signal_type_direction_repair_20261022"
DOMESTIC = ROOT / "runs/domestic_direction_20261022"
OLD = ROOT / "runs/verified_annotation_20261022/attribute_manifest_conservative.json"
SMALL = ROOT / "runs/signal_small_type_20261019/corrected_v2/manifest.json"
TYPE_NAMES = ["vehicle_signal", "pedestrian_signal", "bicycle_signal", "other_object"]


def rgb_digest(path):
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "manifest.json"
    if target.exists():
        data = json.loads(target.read_text(encoding="utf8"))
        for name, sha in data["input_manifests"].items():
            if digest(name) != sha: raise RuntimeError("Frozen dataset input changed: " + name)
        return data
    domestic = json.loads((DOMESTIC / "manifest.json").read_text(encoding="utf8"))
    old = json.loads(OLD.read_text(encoding="utf8"))
    small = json.loads(SMALL.read_text(encoding="utf8"))
    device = json.loads((OUT / "device_review_rows.json").read_text(encoding="utf8"))
    audit_path = ROOT / "runs/direction_path_relation_20261021/visual_direction_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))["records"]
    sources = {r["source_image"] for r in old["records"] if not r.get("synthetic")}
    sources.update(r["source_image"] for r in small["records"])
    sources.update(r["source_image"] for r in device)
    sources.update(r["image"] for r in audit if r["direction_label"] is not None)
    print("TYPE_DIRECTION_CANONICAL_SOURCES", len(sources), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_sha = dict(zip(sorted(sources), pool.map(rgb_digest, sorted(sources))))
    protected = {source_sha[r["image"]] for r in audit if r["direction_label"] is not None}
    rows = []
    for r in domestic["records"]:
        role = {"train": "train", "calibration": "domestic_calibration", "test": "domestic_review"}[r["split"]]
        rows.append(dict(image=r["crop"], role=role, type_id=0, direction_id=r["direction_id"],
            source_rgb_sha256=r["source_pixel_sha256"], source_image=r["source_image"],
            recording_group=r["recording_group"], origin="domestic_author", synthetic=False,
            previous_direction_id=r["id"], target_origin="author motor signal class; reviewed arrow mapping"))
    for r in old["records"]:
        if r.get("synthetic") or r["type_id"] < 0: continue
        rows.append(dict(image=r["image"], role="train" if r["attribute_split"] == "train" else "original_validation",
            type_id=r["type_id"], direction_id=r["direction_id"] if r["type_id"] == 0 else -100,
            source_rgb_sha256=source_sha[r["source_image"]], source_image=r["source_image"],
            recording_group="legacy/"+source_sha[r["source_image"]], origin="final_conservative_attributes", synthetic=False,
            target_origin="final reviewed type/direction; ambiguous public_009 and public_014 excluded"))
    for r in small["records"]:
        if r.get("synthetic") or r["origin"] == "original_type_manifest": continue
        rows.append(dict(image=r["image"], role=r["role"], type_id=r["type_id"], direction_id=-100,
            source_rgb_sha256=source_sha[r["source_image"]], source_image=r["source_image"],
            recording_group="legacy/"+source_sha[r["source_image"]], origin="corrected_small_type_review", synthetic=False,
            target_origin="corrected explicit motor/bicycle/digit-only review; other_display mapped to other_object"))
    # The twenty contact crops were inspected in this task: training vehicle
    # trim/plates/signs/backgrounds and five existing camera-device checks. None
    # is relabeled as pedestrian, bicycle, or an unlit vehicle signal.
    for r in device:
        rows.append(dict(image=r["image"], role="train" if r["split"] == "train" else "device_review",
            type_id=3, direction_id=-100, source_rgb_sha256=source_sha[r["source_image"]], source_image=r["source_image"],
            recording_group="legacy/"+source_sha[r["source_image"]], origin="visually_reviewed_non_signal_object",
            synthetic=False, proposal_id=r["proposal_id"], target_origin="native crop contact inspected; true non-signal object"))
    by_crop = {}; exclusions = []
    for row in rows:
        if Path(row["image"]).name in {"public_009.png", "public_014.png"}:
            exclusions.append(dict(image=row["image"], reason="previously ambiguous public crop")); continue
        row["image_sha256"] = digest(row["image"])
        key = row["image_sha256"]
        if key in by_crop:
            previous = by_crop[key]
            if previous["type_id"] != row["type_id"]: raise RuntimeError("Conflicting known type labels for identical crop")
            if previous["role"] == "train" and row["role"] != "train": by_crop[key] = row
            exclusions.append(dict(image=row["image"], reason="identical crop deduplicated")); continue
        by_crop[key] = row
    rows = list(by_crop.values())
    held_sources = {r["source_rgb_sha256"] for r in rows if r["role"] != "train"}
    final = []
    for row in rows:
        if row["role"] == "train" and (row["source_rgb_sha256"] in held_sources or row["source_rgb_sha256"] in protected):
            exclusions.append(dict(image=row["image"], reason="full source reserved for validation/known public diagnostic")); continue
        row["id"] = f"repair_{len(final):04d}"
        final.append(row)
    train_sources = {r["source_rgb_sha256"] for r in final if r["role"] == "train"}
    assert not train_sources & held_sources and not train_sources & protected
    assert all(not r["synthetic"] for r in final)
    inputs = [DOMESTIC / "manifest.json", OLD, SMALL, OUT / "device_review_rows.json", audit_path]
    result = dict(records=final, type_names=TYPE_NAMES, input_manifests={str(p): digest(p) for p in inputs},
        counts={s:dict(Counter(TYPE_NAMES[r["type_id"]] for r in final if r["role"] == s)) for s in sorted({r["role"] for r in final})},
        direction_train_counts=dict(Counter(r["direction_id"] for r in final if r["role"] == "train" and r["direction_id"] >= 0)),
        exclusions=exclusions, protected_known_direction_source_count=len(protected),
        train_source_disjoint_from_all_evaluation=True, train_excludes_known_public_direction_sources=True,
        unmapped_66_class_data_used=False, physical_lane_head_labels=False,
        scope="Existing recording-separated domestic crops + corrected legacy type/digit/device labels; all evaluation materials known development/regression, not blind site-independent validation")
    save(target, result)
    print("TYPE_DIRECTION_MANIFEST", result["counts"], result["direction_train_counts"], flush=True)
    return result


if __name__ == "__main__": prepare()
