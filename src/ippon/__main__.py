"""Start the CLI with actionable errors even when a dependency prevents its imports."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from importlib import import_module

from .errors import RuntimeUnavailable


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        run: Callable[[Sequence[str]], int] = import_module(".cli", __package__).main
    except ImportError as error:
        failure = RuntimeUnavailable.of(error)
        if failure is None:  # a fault inside ippon, not the user's installation
            raise
        if "--json" in arguments:
            print(json.dumps(failure.to_json(exit_code=True)))
        print(f"ippon: {failure}", file=sys.stderr)
        return failure.exit_code
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
