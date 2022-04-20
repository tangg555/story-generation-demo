"""
@Desc:
@Reference:
@Notes:

"""


def are_same_strings(string1: str, string2: str):
    if not isinstance(string1, str) or not isinstance(string2, str):
        raise ValueError("input should be strings")
    return string1.strip().lower() == string2.strip().lower()


def rm_extra_spaces(string: str):
    return " ".join(string.strip().split())
