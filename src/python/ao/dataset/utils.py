from ao.io import yaml_load

import os

from pathlib import Path
from warnings import warn
from dotenv import load_dotenv
from typing import Optional, Union, Dict, Tuple


def strtobool(value: str) -> bool:
    """Converts a string to a boolean. True values are `y`, `yes`, `t`, `true`,
    `on` and `1`; false values are `n`, `no`, `f`, `false`, `off` and `0`.
    Raises ValueError if value is anything else. Reimplemented from
    https://docs.python.org/3/distutils/apiref.html#distutils.util.strtobool
    due to distutils deprecation
    https://peps.python.org/pep-0632/#migration-advice.

    Args:
        value (str): String value to be converted.

    Returns:
        bool: Boolean converted from the string.
    """
    if value.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def get_folder(env: Optional['str'] = None, interactive: bool = True) -> Path:
    data_folder = None
    if env:
        load_dotenv()
        data_folder = os.getenv(env)
    if not data_folder and interactive:
        data_folder = input("Enter folder path: ")
    if not data_folder:
        raise ValueError(f"Could not find folder {env}")
    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise ValueError(
            fr"Specified data folder {data_folder} is not a directory"
            )
    return data_folder


def parse_filename(filename: str) -> dict:
    """Parses a filename into a dictionary. Dictionary items are divided by `;`
    characters. Key and value are separated by `_`. Values are parsed into
    `float`, `bool` or left as string in that order. See
    https://docs.python.org/1/distutils/apiref.html#distutils.util.strtobool
    for details on the boolean conversion.

    Args:
        filename (str): Filename to be parsed. It is supposed to be only the
            name, not a full path.

    Returns:
        dict: Parsed dictionary containing string keys with corresponding
            float, bool and string values.
    """
    parsed = {}
    for item in str(filename).split(';'):
        try:
            key, value = item.split('_')
        except ValueError:
            raise ValueError(
                f"Could not parse {item} from {filename}, item {item} should "
                "be composed by one `key` and one `value` separated by a "
                f"unique underscore but {len(item.split('_')) - 1} `_` were "
                "found."
                )
        # Parse number
        try:
            value = float(value)
        except ValueError:
            # Parse bool
            try:
                value = strtobool(value)
            except ValueError:
                # Leave as string
                pass
        parsed[key] = value
    return parsed


def dict_to_filename(d: dict) -> str:
    """Parses a dictionary into a filename. Dictionary items are divided by `;`
    characters. Key and value are separated by `_`. Key and valued are parsed
    using `str` constructor.

    Args:
        d (dict): Dictionary to be parsed into a string. Parsed values
            can't contain `;` or `_` characters as they are used as separators.

    Raises:
        ValueError: If a parsed key or value contains `;` or `_` characters.

    Returns:
        str: Filename containing the parsed dictionary.
    """
    parsed_items = []
    for item in d.items():
        parsed_item = []
        for value in item:
            parsed_value = str(value)
            if ';' in parsed_value or '_' in parsed_value:
                raise ValueError(
                    f"Could not parse `{value}` from {item}, parsed value "
                    f"`{parsed_value}` contains `;` or `_` characters."
                    )
            parsed_item.append(parsed_value)
        parsed_items.append('_'.join(parsed_item))
    return ';'.join(parsed_items)


def list_data(
    data_folder: Union[str, Path],
    naming: Optional[dict] = None,
    ) -> Tuple[Dict[Path, dict], dict]:
    """Inspects a data folder composed of subdirectories that follow a given
    naming convention.

    Args:
        data_folder (Union[str, Path]): Path to the data folder composed of
            subdirectories that follow the naming convention specified by
            `naming`.

        naming (Optional[dict]): Naming convention to be used. Dictionary keys
            are the expected values to be found in each subfolder of the
            data_folder. If not provided, it will be loaded from a
            `naming.yaml` file in the data_folder. That can be avoided by
            specifying `naming` to `False`.

    Returns:
        (data, naming):
            data (Dict[Path, dict]): Dictionary of subdirectory paths as keys
                and their corresponding parsed filenames as values.

            naming (dict): Naming convention used.
    """
    data_folder = Path(data_folder)
    # Find naming convention
    if naming is None:
        _naming = data_folder / 'naming.yaml'
        if _naming.exists():
            naming = yaml_load(_naming)
    # Parse subfolders
    data = {}
    for d in data_folder.iterdir():
        if not d.is_dir():
            continue
        try:
            content = parse_filename(d.name)
        except ValueError as e:
            warn(str(e))
            continue
        # Validate data with naming convention
        if naming:
            if content.keys() != naming.keys():
                warn(
                    f"{d} does not meet the naming convention that requires "
                    f"{naming.keys()}. It has {content.keys()}."
                    )
                continue
        data[d] = content
    return data, naming