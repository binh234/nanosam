from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppcls.utils import config
from nanosam.tools.paddle.engine import Engine

import logging

logger = logging.getLogger("ppcls")
logger.setLevel(logging.WARN)

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    engine = Engine(config, mode="train")
    engine.train()
