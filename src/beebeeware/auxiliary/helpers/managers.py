import contextlib
import sys
from io import StringIO


@contextlib.contextmanager
def capture():
    """
    Docstring for capture: decorator to be used to redirect
    sys terminal outputs into the app's window.
    """
    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()  # type: ignore
        out[1] = out[1].getvalue()  # type: ignore
