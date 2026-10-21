import faulthandler
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
faulthandler.enable()
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass

from .cli import main  # noqa: E402

sys.exit(main())
