import yaml
from pathlib import Path


story_dict = {}

for file in Path("Persona").glob("*.yml"):
    fname = file.stem
    try:
        story_dict[fname] = yaml.load(open(file, encoding='utf-8'), Loader=yaml.FullLoader)
    except UnicodeDecodeError as e:
        story_dict[fname] = yaml.load(open(file), Loader=yaml.FullLoader)
    except yaml.scanner.ScannerError as e:
        print("Error with ", file)
