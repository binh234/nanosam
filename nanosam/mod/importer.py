import importlib
import logging
import sys

from .utils import check_version


def _version_ok(ver, preferred):
    return check_version(ver, preferred)


def lazy_import(name: str, version: str = None):
    """
    Lazily import a module.

    Args:
        name (str):
            The module name to load
        version (str): Optionally the preferred version of the package
            For example, ``'>=0.5.0'`` or ``'==1.8.0'``.

    Returns:
        LazyModule:
            A lazily loaded module. When an attribute is first accessed,
            the module will be imported.
    """

    def import_mod():
        mod = None
        try:
            mod = importlib.import_module(name)
        except ImportError as err:
            logging.critical(
                f"Module: '{name}' is required but could not be imported.\nNote: Error was: {err}\n",
                extra={"type": "module_import"},
            )
            sys.exit(1)

        if (
            version is not None
            and hasattr(mod, "__version__")
            and not _version_ok(mod.__version__, version)
        ):
            logging.warning(
                f"Module: '{name}' version '{mod.__version__}' is installed, but version '{version}' is required.\n",
                extra={"type": "module_import"},
            )

        return mod

    MODULE_VAR_NAME = "module"

    class LazyModule:
        def __init__(self):
            super().__setattr__(MODULE_VAR_NAME, None)

        def _import_mod(self):
            if self.module is None:
                super().__setattr__(MODULE_VAR_NAME, import_mod())
            return self.module

        def __getattr__(self, name):
            module = self._import_mod()
            return getattr(module, name)

        def __setattr__(self, name, value):
            module = self._import_mod()
            return setattr(module, name, value)

    return LazyModule()


def has_mod(modname):
    """
    Checks whether a module is available and usable.

    Args:
        modname (str): The name of the module to check.

    Returns:
        bool:
            Whether the module is available and usable.
            This returns false if importing the module fails for any reason.
    """
    try:
        importlib.import_module(modname)
    except:
        return False
    return True
