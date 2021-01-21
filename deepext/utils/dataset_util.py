from typing import Dict, List, Tuple


def create_label_list_and_dict(label_file_path: str) -> Tuple[List[str], Dict[str, int]]:
    label_names, label_dict = [], {}
    i = 0
    with open(label_file_path, "r") as file:
        for line in file:
            label_name = line.replace("\n", "")
            label_names.append(label_name)
            label_dict[label_name] = i
            i += 1
    return label_names, label_dict
