from __future__ import \
    annotations  # ensure compatibility with cluster python version

import pathlib
import re
import sys
import typing
from collections import Counter, defaultdict
from functools import reduce
from itertools import cycle

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# # load scripts in ./evalled
# # e.g., eval callable via f = eval("evalled.script.f")
# for p in pathlib.Path("evalled").glob("*.py"):
#     exec(f"import converters.{p.stem}")


def main(argv):

    # load the path synonynms using in configs
    # e.g., {"DATA": "~/surfdrive/data"}
    with open("path_syns.json", "rb") as f:
        path_syns = orjson.loads(f.read())

    # load the configs - prefer CL specified over default
    # [
    #   {
    #       "desc": "config 1",
    #       "switch": true,
    #       "output_dir": "DATA/project1"
    #   }
    # ]
    try:
        configs: list = get_configs(argv[0], path_syns=path_syns)
    except:
        configs: list = get_configs("counts_configs.json", path_syns=path_syns)

    # iterate over configs
    for config in configs:

        # get config options
        desc = config["desc"]
        print(f"config={desc}")

        switch: bool = config["switch"]  # run config or skip?

        input_dir: pathlib.Path = resolve_fp(config["input"][0], path_syns=path_syns)
        input_pattern: re.Pattern = eval(config["input"][1])

        # config to be run?
        if switch == False:
            print("\tconfig switched off ... skipping")

        else:

            # build metdatadataframe
            fps = gen_dir(input_dir, pattern=input_pattern)
            df = pd.concat((pd.read_csv(fp, index_col=0) for fp in fps))

            for papertitle in set(df.loc[:, "ddd:papertitle"]):
                print(
                    f"{papertitle}, {len(df.loc[df.loc[:, 'ddd:papertitle'] == papertitle, :])}"
                )


def resolve_fp(path: str, path_syns: typing.Union[None, dict] = None) -> pathlib.Path:
    """Resolve path synonyns, ~, and make absolute, returning pathlib.Path.

    Args:
        path (str): file path or dir
        path_syns (dict): dict of
            string to be replaced : string to do the replacing

    E.g.,
        path_syns = {"DATA": "~/documents/data"}

        resolve_fp("DATA/project/run.py")
        # >> user/home/john_smith/documents/data/project/run.py
    """

    # resolve path synonyms
    if path_syns is not None:
        for fake, real in path_syns.items():
            path = path.replace(fake, real)

    # expand user and resolve path
    return pathlib.Path(path).expanduser().resolve()


def get_configs(config_fp_str: str, *, path_syns=None) -> list:
    """Return the configs to run."""

    configs_fp = resolve_fp(config_fp_str, path_syns)

    with open(configs_fp, "rb") as f:
        configs = orjson.loads(f.read())

    return configs


def gen_dir(
    dir_path: pathlib.Path,
    *,
    pattern: re.Pattern = re.compile(".+"),
    ignore_pattern: typing.Union[re.Pattern, None] = None,
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): re.search pattern to match wanted files [default: all files]
        ignore (re.Pattern): re.search pattern to ignore wrt., previously matched files
    """

    for fp in filter(lambda fp: re.search(pattern, str(fp)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp)):
                pass
            else:
                yield fp


if __name__ == "__main__":
    main(sys.argv[1:])  # assumes an alternative config path may be passed to CL
