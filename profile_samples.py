"""
for each config in sample_configs.json, profile the information in output_dir of each config.
specifically, profile the 

run:
```
python3 profile_samples.py
```
"""

import json
import pathlib
import re
from pprint import pp

import pandas as pd

DEBUG = False


def main():

    # ------
    # Load the variable configurations
    # ------

    with open("sample_configs.json", "r") as f:
        configs: list[dict] = json.load(f)

    # ------
    # Iterate over each config
    # ------
    for config in configs:

        print(f"config: {config['name']}")

        # ------
        # Load the config variables of interest
        # ------

        # output from sample.py wrt., current config
        samples_dir = pathlib.Path(config["output_dir"]).expanduser()

        # ------
        # The samples are named according to their parent query
        # decompose the queries into their respective components
        # ------

        # get an iterable of paths to samples (ignore config.*)
        samples_paths: list[pathlib.Path] = [
            p for p in samples_dir.glob("*.json") if p.stem != "config"
        ]
        if DEBUG:
            print(f"\tfirst sample path: \n\t\t{samples_paths[0]}")

        # ------
        # get a list of (content words, papertitle, type, date) corresponding to
        # samples_paths
        # ------

        # map file paths -> file stems (==jsru queries)
        stems: list[str] = [
            re.match(".*/(.*?).json", str(p)).groups()[0] for p in samples_paths
        ]
        if DEBUG:
            print(f"\tfirst sample stem: \n\t\t{stems[0]}")

        # mapping file stems (queries) -> (content_words, papertitle, type, date)
        decomposed_stems: list[tuple[str]] = [decompose_query(fp) for fp in stems]
        if DEBUG:
            print(f"\tfirst sample query components: \n\t\t{decomposed_stems[0]}")

        # ------
        # get a list of ocr count corresponding to samples_paths
        # ------

        counts = [0] * len(samples_paths)
        for i, p in enumerate(samples_paths):

            # open the ocr collection
            with open(p, "r") as f:
                collection = json.load(f)

            # record ocr count wrt, query
            counts[i] = len(collection)

        if DEBUG:
            print(f"\tfirst sample ocr count: \n\t\t{counts[0]}")

        # ------
        # produce a csv of rows = content_words + date VS columns = papertitle + type
        # ------

        # list of unique content_date combinations
        row_indices = sorted(
            list(
                set(
                    [
                        f"{c}_{d}"
                        for c, d in zip(
                            list(zip(*decomposed_stems))[0],
                            list(zip(*decomposed_stems))[3],
                        )
                    ]
                )
            )
        )
        if DEBUG:
            print(f"\ta row indices: \n\t\t{list(row_indices)[0]}")

        # list of unique papertitle_type combinations
        column_indices = sorted(
            list(
                set(
                    [
                        f"{c}_{d}"
                        for c, d in zip(
                            list(zip(*decomposed_stems))[1],
                            list(zip(*decomposed_stems))[2],
                        )
                    ]
                )
            )
        )
        if DEBUG:
            print(f"\tcolumn indices: \n\t\t{list(column_indices)[0]}")

        # ------
        # get the ocr count in each of the samples
        # ------

        # init. a dataframe
        df = pd.DataFrame(0, index=row_indices, columns=column_indices)

        # populate the dataframe
        for (content_words, papertitle, type_, date), count in zip(
            decomposed_stems, counts
        ):
            row = f"{content_words}_{date}"
            col = f"{papertitle}_{type_}"
            df.loc[row, col] += count

        # save this csv
        df.to_csv(samples_dir / "profile.csv")


def decompose_query(query: str) -> tuple[str]:
    """Decompose a query string into its constituent components

    Return:
        (content words:str, papertitle:str, type:str, date_within:str)
    """

    # extract content words
    context_regex = re.compile(
        "(.*?)(?=papertitle|type|date|\s+AND|$)", flags=re.IGNORECASE
    )
    if re.match(context_regex, query):
        content = re.match(context_regex, query).groups()[0]
        if content == "":
            content = None
    else:
        content = None

    # extract papertitle
    papertitle_regex = re.compile("papertitle=(.*?)(?:$|\s+AND)", flags=re.IGNORECASE)
    if re.search(papertitle_regex, query):
        papertitle = re.search(papertitle_regex, query).groups()[0]
    else:
        papertitle = None

    # extract type
    type_regex = re.compile("type=(.*?)(?:$|\s+AND)", flags=re.IGNORECASE)
    if re.search(type_regex, query):
        type_ = re.search(type_regex, query).groups()[0]
    else:
        type_ = None

    # extract date window
    date_regex = re.compile("date within (.*?)(?:$|\s+AND)", flags=re.IGNORECASE)
    if re.search(date_regex, query):
        date_within = re.search(date_regex, query).groups()[0]
    else:
        date_within = None

    return (content, papertitle, type_, date_within)


if __name__ == "__main__":
    main()
