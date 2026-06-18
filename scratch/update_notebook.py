import json

notebook_path = "notebooks/07_statistical_validation.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

modified = False
for cell in data.get("cells", []):
    if cell.get("cell_type") in ("markdown", "code"):
        source = cell.get("source", [])
        new_source = []
        for line in source:
            if "option_b_evaluation" in line:
                new_line = line.replace("option_b_evaluation", "scientific_validation")
                new_source.append(new_line)
                modified = True
                print(f"Replaced in cell {cell.get('id')}: {line.strip()} -> {new_line.strip()}")
            else:
                new_source.append(line)
        cell["source"] = new_source

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print("Successfully updated notebooks/07_statistical_validation.ipynb")
else:
    print("No occurrences of option_b_evaluation found in notebooks/07_statistical_validation.ipynb")
