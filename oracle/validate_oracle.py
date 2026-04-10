import json
import os
import sys

REQUIRED_FIELDS = ["g", "P_t", "R_t", "e_t", "C", "F_t", "K_t"]


def validate(filepath):
    with open(filepath) as f:
        data = json.load(f)

    parsed = data.get("parsed")
    report = {
        "file": os.path.basename(filepath),
        "fields": {},
        "passed": True,
        "parse_error": parsed is None
    }

    if parsed is None:
        print(f"  WARNING: could not parse JSON in {filepath} - check raw_response")
        report["passed"] = False
        return report

    for field in REQUIRED_FIELDS:
        value = parsed.get(field)
        if value is None:
            status = "MISSING"
        elif value == [] or value == "":
            # only g and e_t must always have content
            if field in ["g", "e_t"]:
                status = "EMPTY - check this"
                report["passed"] = False
            else:
                status = "ok (empty)"
        else:
            status = "ok"
        report["fields"][field] = status
        if status == "MISSING":
            report["passed"] = False

    return report


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")

    if not os.path.exists(output_dir):
        print(f"No outputs directory found at {output_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    if not files:
        print("No output files found in oracle/outputs/")
        sys.exit(0)

    print(f"Validating {len(files)} file(s)...\n")
    all_passed = True

    for filename in sorted(files):
        filepath = os.path.join(output_dir, filename)
        result = validate(filepath)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {filename}")
        for field, val in result["fields"].items():
            marker = "  " if "ok" in val else "!!"
            print(f"  {marker} {field}: {val}")
        print()
        if not result["passed"]:
            all_passed = False

    print("All files passed." if all_passed else "Some files failed - review above.")
