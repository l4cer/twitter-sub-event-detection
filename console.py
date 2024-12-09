import sys

import traceback

from typing import Any

from datetime import datetime


NORMAL = "\33[0m"

GRAY   = "\33[90m"
RED    = "\33[31m"
GREEN  = "\33[32m"
YELLOW = "\33[33m"
BLUE   = "\33[34m"
VIOLET = "\33[35m"

ITALIC = "\33[3m"
UNDERLINE = "\33[4m"


def log(message: Any) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        message = str(message).format(**globals())

    except KeyError as key:
        traceback_stack = traceback.format_stack()
        print("".join(traceback_stack[:-1]))

        values = [f"{value}{name}{NORMAL}" for name, value in globals().items() if name.isupper()]
        print(f"{RED}[ERROR]{NORMAL} Invalid key {key}. Possible values are " + ", ".join(values))

        sys.exit(1)

    print(f"{GREEN}{timestamp} {BLUE}[LOG]{NORMAL} {message}{NORMAL}")


def skip_lines(num: int = 1) -> None:
    print(num * "\n", end="")
