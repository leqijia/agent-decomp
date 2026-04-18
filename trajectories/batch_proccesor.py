import argparse
from pathlib import Path
from typing import List

from trajectories.result_parser import main as result_parser_main


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch run trajectories.result_parser over all JSON files in an input directory."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to write success and error files")

    args = parser.parse_args(argv[1:] if argv is not None else None)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"]
    )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total JSON files: {len(json_files)}")

    for index, json_file in enumerate(json_files, start=1):
        print(f"Processing file no {index} : {json_file.name}")
        try:
            result_parser_main(["trajectories.result_parser", str(json_file), str(output_dir)])
        except SystemExit as exc:
            print(f"Failed processing {json_file.name}: exited with code {exc.code}")
        except Exception as exc:
            print(f"Failed processing {json_file.name}: {exc}")


if __name__ == "__main__":
    main()


