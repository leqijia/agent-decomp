import sys
import json
from fields import output_mapper
try:
    with open(sys.argv[1],'r') as file:
        data = json.load(file)
        output = output_mapper()
        output.normalize(data)
except json.JSONDecodeError as e:
    print("Malformed JSON:",e)
except FileNotFoundError:
    print("File not found error")
