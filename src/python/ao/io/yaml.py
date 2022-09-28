import yaml

from pathlib import Path


class SafeLoader(yaml.SafeLoader):

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple', SafeLoader.construct_python_tuple
    )


def yaml_dump(data, to_file: Path = None) -> str:
    if to_file:
        with open(to_file, 'w', encoding="utf-8") as f:
            return yaml.dump(data, stream=f, allow_unicode=True)
    return yaml.dump(data, allow_unicode=True)


def yaml_load(from_file: Path) -> dict:
    with open(from_file, 'r', encoding="utf-8") as f:
        return yaml.load(f, SafeLoader)