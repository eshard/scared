# This file is a trick. By importing your library from context in the test, your ensure that it will always work whatever
# is the test runner, tests execution context or environnement.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scared  # noqa E402
