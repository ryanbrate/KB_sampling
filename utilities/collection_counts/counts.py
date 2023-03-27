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

        metadata_dir: pathlib.Path = resolve_fp(
            config["metadata"][0], path_syns=path_syns
        )
        metadata_pattern: re.Pattern = eval(config["metadata"][1])
        metadata_doc_label_col: str = config["metadata"][2]

        collections_dir: pathlib.Path = resolve_fp(
            config["collections"][0], path_syns=path_syns
        )
        collections_pattern: re.Pattern = eval(config["collections"][1])

        cuts: list[dict] = config["cuts"]

        # config to be run?
        if switch == False:
            print("\tconfig switched off ... skipping")

        else:

            # build metdatadataframe
            print("build a metadata dataframe")
            csvs = gen_dir(
                metadata_dir, pattern=metadata_pattern, ignore_pattern="config"
            )
            df: pd.DataFrame = pd.concat((pd.read_csv(fp, index_col=0) for fp in csvs))

            # build a list of collection doc_labels available
            print("build a list of collection doc_labels available")
            collections_fps = gen_dir(
                collections_dir,
                pattern=collections_pattern,
                ignore_pattern=re.compile(r"config"),
            )

            doc_labels = set()
            for fp in collections_fps:

                with open(fp, "rb") as f:
                    collection = orjson.loads(f.read())

                for doc_label, _ in collection:
                    doc_labels.add(doc_label)

            # get metadata subset of only the collected labels
            print(
                "build a metadata dataframe subset, of only those articles available in the collection"
            )
            mask = [x in doc_labels for x in df.loc[:, metadata_doc_label_col]]
            df_collections: pd.DataFrame = df.loc[mask, :]

            # go through each cut and get the count
            print("get the count of each cut")
            for cut in cuts:

                # get the metadata df wrt., the cut
                masks = []
                for column, pattern in cut.items():
                    mask = [
                        bool(re.search(eval(pattern), str(x)))
                        for x in df_collections.loc[:, column]
                    ]
                    masks.append(mask)

                all_mask = [all(t) for t in zip(*masks)]

                # return count
                print(
                    f"for metadata {cut}, there are {sum(all_mask)} articles",
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
