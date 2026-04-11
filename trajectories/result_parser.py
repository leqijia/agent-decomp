import sys
import json
from .validate import output_mapper
from pathlib import Path
from datetime import datetime
def get_output_files(input_path):
    p = Path(input_path)

    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S") + f"_{now.microsecond // 1000:03d}"

    success_file = p.with_name(f"{p.stem}_success_{timestamp}{p.suffix}")
    error_file = p.with_name(f"{p.stem}_error_{timestamp}{p.suffix}")

    return str(success_file), str(error_file)
try:
    with open(sys.argv[1],'r') as file:
        data = json.load(file)
        success, error = get_output_files(sys.argv[1])
        output = output_mapper( success , error )
        output.normalize(data)
except json.JSONDecodeError as e:
    print("Malformed JSON:",e)
except FileNotFoundError:
    print("File not found error")


