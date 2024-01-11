from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppcls.utils import config
from nanosam.tools.paddle.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=False)
    if config["Arch"].get("use_sync_bn", False):
        config["Arch"]["use_sync_bn"] = False
    engine = Engine(config, mode="export")
    engine.export()
