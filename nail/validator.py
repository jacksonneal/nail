from os.path import isdir


def dir_exists(v: str) -> str:
    assert isdir(v), f"Directory {v} does not exist"
    return v
