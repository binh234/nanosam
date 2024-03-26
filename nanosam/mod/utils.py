import re


def parse_version(version_str: str):
    def process_version_part(num):
        VERSION_SUFFIXES = ["a", "b", "rc", "post", "dev"]
        try:
            return [int(num)]
        except ValueError:
            # One version part can only contain one of the above suffixes
            for suffix in VERSION_SUFFIXES:
                if suffix in num:
                    return num.partition(suffix)

            # For unrecognized suffixes, just return as-is
            return [num]

    ver_list = []
    for num in version_str.split("."):
        ver_list.extend(process_version_part(num))

    return tuple(ver_list)


def str_to_comparison_op(s):
    if s == "gt" or s == ">":
        return lambda a, b: a > b
    elif s == "ge" or s == ">=":
        return lambda a, b: a >= b
    elif s == "lt" or s == "<":
        return lambda a, b: a < b
    elif s == "le" or s == "<=":
        return lambda a, b: a <= b
    elif s == "eq" or s == "==":
        return lambda a, b: a == b
    else:
        raise ValueError(
            f"{s} is not a valid value. \
            Only ('gt','ge','lt','le','eq',\
            '>','>=','<','<=','==') are allowed"
        )


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version ",
    hard: bool = False,
) -> bool:
    """
    Check current version against the required version or range.
    Match required version up to patch version. Missing components of version string
    is implicitly filled with 0 e.g. 8.2 ==> 8.2.0, 8 ==> 8.0.0
    Args:
        current (str): Current version.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        # check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
    """
    current_version = parse_version(current)
    constraints = []
    pattern = r"([<>!=]{0,2})\s*(\d+(?:\.\d+){0,2})"

    for required_split in required.split(","):
        matched = re.match(pattern, required_split)
        if matched is None:
            raise Exception(f"Unable to match {required_split}")
        op, version = matched.groups()
        op_func = str_to_comparison_op(op or ">=")
        constraints.append((op_func, version))

    result = True
    for op_func, required_version in constraints:
        required_version = parse_version(required_version)
        result = result and op_func(current_version, required_version)

    if not result:
        warning_message = f"WARNING ⚠️ {name}{required} is required, but {name}{current} is currently installed"
        if hard:
            raise ModuleNotFoundError(
                warning_message
            )  # assert version requirements met
    return result
