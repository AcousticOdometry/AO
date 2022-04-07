import os
import yaml

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


def get_data_folder(
        env: Optional['str'] = None, interactive: bool = True
    ) -> Path:
    data_folder = None
    if env:
        load_dotenv()
        data_folder = os.getenv(env)
    if not data_folder and interactive:
        data_folder = input("Enter data folder: ")
    if not data_folder:
        raise ValueError(f"Could not find data folder {env}")
    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise ValueError(
            f"Specified data folder {data_folder} is not a directory"
            )
    return data_folder


def parse_filename(filename: Union[str, Path]) -> dict:
    """Parses a filename into a dictionary. Dictionary items are divided by `;`
    characters. Key and value are separated by `_`. Values are parsed into
    `int`, `bool` or left as string in that order. See
    https://docs.python.org/3/distutils/apiref.html#distutils.util.strtobool
    for details on the boolean conversion.

    Args:
        filename (Union[str, Path]): Filename to be parsed. If a full path is
            given, only its stem will be used (filename without suffix).

    Returns:
        dict: Parsed dictionary containing string keys with corresponding int,
            bool and string values.
    """
    parsed = {}
    for item in Path(filename).stem.split(';'):
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
            value = int(value)
        except ValueError:
            # Parse bool
            try:
                value = strtobool(value)
            except ValueError:
                # Leave as string
                pass
        parsed[key] = value
    return parsed


def list_data(
    data_folder: Union[str, Path],
    naming: Optional[dict] = None,
    ) -> Tuple[Dict[Path, dict], dict]:
    """_summary_

    Args:
        data_folder (Union[str, Path]): _description_
        naming (Optional[dict]): Naming convention to be used. Dictionary keys
            are the expected values to be found in each subfolder of the
            data_folder. If not provided, it will be loaded from a
            `naming.yaml` file in the data_folder. That can be avoided by
            specifying `naming` to `False`.

    Returns:
        (data, naming):
            data (Dict[Path, dict]): _description_
            naming (dict): _description_
    """
    data_folder = Path(data_folder)
    # Find naming convention
    if naming is None and (naming_file :=
                           data_folder / 'naming.yaml').exists():
        with open(naming_file) as f:
            naming = yaml.safe_load(f)
    # Parse subfolders
    data = {}
    for d in data_folder.iterdir():
        if not d.is_dir():
            continue
        try:
            content = parse_filename(d)
        except ValueError as e:
            warn(str(e))
            continue
        # Validate data with naming convention
        if naming:
            if content.keys() != naming.keys():
                warn(
                    f"{d} does not meet the naming convention that requires "
                    f"{naming.keys()}"
                    )
                continue
        data[d] = content
    return data, naming