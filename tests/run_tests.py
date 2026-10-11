from __future__ import annotations

import pathlib
import sys
import unittest


if __name__ == "__main__":
    repository_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    suite = unittest.defaultTestLoader.discover(str(repository_root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

