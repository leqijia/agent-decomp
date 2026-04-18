import sys
import json
from .validate import output_mapper
from pathlib import Path
from datetime import datetime


def get_output_files(input_path, output_dir):
    p = Path(input_path)
    out_dir = Path(output_dir)

    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_{now.microsecond // 1000:03d}"

    success_file = out_dir / f"{p.stem}_success_{timestamp}{p.suffix}"
    error_file = out_dir / f"{p.stem}_error_{timestamp}{p.suffix}"

    return str(success_file), str(error_file)


def main(argv: list | None = None) -> None:
    """Run the result parser.

    argv: list of args (like sys.argv). If None, uses sys.argv.
    """
    if argv is None:
        argv = sys.argv

    if len(argv) < 3:
        print("Usage: result_parser.py <input.json> <output_directory>")
        return

    input_path = argv[1]
    output_dir = argv[2]

    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            success, error = get_output_files(input_path, output_dir)
            output = output_mapper(success, error)
            output.normalize(data)
    except json.JSONDecodeError as e:
        print("Malformed JSON:", e)
    except FileNotFoundError:
        print("File not found error")


if __name__ == "__main__":
    main()


