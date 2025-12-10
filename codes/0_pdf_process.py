import os
import json
import argparse
from typing import Any, Dict, List

# -----------------------------------------------------------------
# Helpers for "standard" S2 PDF JSON (remove *_spans + meta fields)
# -----------------------------------------------------------------
STANDARD_DROP_KEYS = {
    "cite_spans", "ref_spans", "eq_spans",          # span lists
    "authors", "bib_entries", "year", "venue",      # metadata
    "identifiers", "_pdf_hash", "header",           # misc
}

def _clean_standard(obj: Any) -> Any:
    """Recursively remove the keys above from vanilla S2 JSON."""
    if isinstance(obj, dict):
        # drop keys in-place
        for k in STANDARD_DROP_KEYS:
            obj.pop(k, None)
        # recurse
        return {k: _clean_standard(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_standard(x) for x in obj]
    return obj


# -----------------------------------------------------------------
# Helpers for "dolphin-ocr" JSON  (keep only label / text / page_no)
# -----------------------------------------------------------------
def _dolphin_keep_element(el: Dict[str, Any]) -> Dict[str, Any]:
    return {k: el[k] for k in ("label", "text") if k in el}

def _clean_dolphin(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bbox, reading_order, etc. – keep only minimal fields."""
    if "pages" in data:                 # multi-page
        pages = []
        for pg in data["pages"]:
            cleaned_pg = {
                "elements": [_dolphin_keep_element(e) for e in pg.get("elements", [])]
            }
            if "page_number" in pg:
                cleaned_pg["page_number"] = pg["page_number"]
            pages.append(cleaned_pg)
        return {"pages": pages}
    else:                               # flat list
        return {
            "elements": [_dolphin_keep_element(e) for e in data.get("elements", [])]
        }


# -----------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", required=True)
    parser.add_argument("--output_json_path", required=True)
    parser.add_argument(
        "--input_json_type",
        default="standard",
        choices=["standard", "dolphin-ocr"],
        help="Format of the input JSON file",
    )
    args = parser.parse_args()

    # 1. read
    with open(args.input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. clean
    if args.input_json_type == "standard":
        cleaned = _clean_standard(data)
    else:  # dolphin-ocr
        cleaned = _clean_dolphin(data)

    # 3. write
    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {args.output_json_path}")

# -----------------------------------------------------------------
if __name__ == "__main__":
    main()