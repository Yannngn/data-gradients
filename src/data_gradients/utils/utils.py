import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np

PathLike: TypeAlias = str | Path


def write_json(path: PathLike, json_dict: dict):
    """Write a json dictionary to a file.
    :param path:        Path to the file. Can be absolute or relative. Should contain '.json' extension.
    :param json_dict:   Dictionary to be written to the file. Should be serializable.
    """
    full_path = Path(path).absolute()
    dirname = full_path.parent
    dirname.mkdir(parents=True, exist_ok=True)

    def _default(o):
        # Numpy types -> native python types
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        # Fallback to string representation
        return str(o)

    with full_path.open("w") as f:
        json.dump(json_dict, f, indent=4, default=_default)


def class_names(mapping, hist: dict):
    if mapping is None:
        return hist

    new_hist = {}
    for key in list(hist.keys()):
        try:
            new_hist.update({mapping[key]: hist[key]})
        except KeyError:
            new_hist.update({key: hist[key]})
    return new_hist


def fuzzy_keys(params: Mapping) -> list[str]:
    """
    Returns params.key() removing leading and trailing white space, lower-casing and dropping symbols.
    :param params: Mapping, the mapping containing the keys to be returned.
    :return: List[str], list of keys as discussed above.
    """
    return [fuzzy_str(s) for s in params.keys()]


def fuzzy_str(s: str):
    """
    Returns s removing leading and trailing white space, lower-casing and drops
    :param s: str, string to apply the manipulation discussed above.
    :return: str, s after the manipulation discussed above.
    """
    return re.sub(r"[^\w]", "", s).replace("_", "").lower()


def get_fuzzy_mapping_param(name: str, params: Mapping):
    """
    Returns parameter value, with key=name with no sensitivity to lowercase, uppercase and symbols.
    :param name: str, the key in params which is fuzzy-matched and retruned.
    :param params: Mapping, the mapping containing param.
    :return:
    """
    fuzzy_params = {fuzzy_str(key): params[key] for key in params.keys()}
    return fuzzy_params[fuzzy_str(name)]


def copy_files_by_list(file_list: list[str], source_dir: PathLike, dest_dir: PathLike) -> None:
    """Copy a list of files from the source directory to the destination directory.

    :param file_list:   List of filenames to be copied.
    :param source_dir:  Path of the source directory.
    :param dest_dir:    Path of the destination directory.
    """
    for file_name in file_list:
        source_file_path = Path(source_dir) / file_name
        # Ensure parent directory exists
        source_file_path.parent.mkdir(parents=True, exist_ok=True)
        if source_file_path.is_file():
            dest_file_path = Path(dest_dir) / file_name
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_file_path, dest_file_path)


def safe_json_load(path: PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        return {}


def text_to_blue(text: str) -> str:
    return f"\033[34;1m{text}\033[0m"


def text_to_yellow(text: str) -> str:
    return f"\033[33;1m{text}\033[0m"


def text_to_red(text: str) -> str:
    return f"\033[31;1m{text}\033[0m"


def break_text(text: str, line_length: int):
    lines = []
    words = text.split()
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)

        if current_length + len(current_line) + word_length <= line_length:
            current_line.append(word)
            current_length += word_length
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    if current_line:
        lines.append(" ".join(current_line))

    # Add spaces to the end of each line to make them equal in length
    for i in range(len(lines)):
        spaces_needed = line_length - len(lines[i])
        lines[i] += " " * spaces_needed

    return lines


def print_in_box(text_lines: str, box_size: int = 70):
    left = text_to_blue("║  ")
    right = text_to_blue("  ║")
    bottom_left = text_to_blue("╚")
    top_bottom = text_to_blue("═")
    top_left = text_to_blue("╔")
    top_right = text_to_blue("╗")
    bottom_right = text_to_blue("╝")

    lines = break_text(text_lines, box_size)
    top_bottom = top_bottom * (box_size + 4)
    print(top_left + top_bottom + top_right)
    for text in lines:
        print(left + text + right)
    print(bottom_left + top_bottom + bottom_right)
