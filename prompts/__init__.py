import yaml
import glob
from pathlib import Path

prompt_dict = {}
for file in Path("prompts").glob("*.yml"):
    fname = file.stem
    try:
        prompt_dict[fname] = yaml.load(open(file, encoding='utf-8'), Loader=yaml.FullLoader)
    except yaml.scanner.ScannerError as e:
        print("Error with ", file)

print(f'prompt_dict:\n{prompt_dict.keys()}')
