"""
Run:
# python3 sample.py
"""

import concurrent.futures
import itertools
import json
import pathlib
import re
import time
import typing
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import requests
import xmltodict
from tqdm import tqdm

base = "http://jsru.kb.nl/sru/sru?recordSchema=ddd&x-collection=DDD_artikel"


def main():

    # load the path synonyms used in configs
    with open("path_syns.json", "r") as f:
        path_syns: dict = json.load(f)

    # load the configs
    with open("sample_configs.json", "r") as f:
        configs: list[dict] = json.load(f)

    # iterate over each sampling configuration
    for config in configs:

        # config options
        output_dir: pathlib.Path = resolve_fp(config["output_dir"], path_syns=path_syns)
        n_wanted: int = config["n"]  # sample size wanted
        switch: bool = config["switch"]

        # skip the config?
        if switch == False:
            "config: {config['name'] switched off ... skipping}"
            continue

        # generator of all query urls
        queries: typing.Generator = gen_queries(config)

        # ------
        # iterate over queries and populate metadata dict
        # ------
        for query_url in tqdm(queries, desc="{processing queries}"):

            # ------
            # set up containers, fps and variables of interest wrt., query and
            # sampled metadata output
            # ------

            # query part of query url
            query_part: str = re.search(r"query=\((.+)\)", query_url).groups()[0]

            # container to hold sampled record metadata
            metadata: dict = {key: [] for key in config["recordData"]}

            metadata_fp = output_dir / "metadata" / f"{query_part}.csv"
            metadata_fp.parent.mkdir(exist_ok=True, parents=True)

            n_available_: int = n_available(query_url)  # count of query results

            if n_available_ == 0:
                continue

            # ------
            # build a metadata csv wrt., query samples
            # ------

            if metadata_fp.exists() == False:

                print(query_url)

                # ------
                # We sample metadata blocks by startIndex, where startIndex ~ range(n_available)
                # records is an Iterable of (startIndex::int, metadata block::list[dict]) for each startIndex
                # ------
                if n_available_ > n_wanted and n_wanted != -1:  # only a subset wanted

                    # querying metadata pages is slow ... we can speed this up
                    # by getting contiguous blocks
                    contiguous_n: int = config["contiguous_n"]

                    # NOTE: relies on n_wanted%contiguous == 0
                    random_sample_indices = np.random.choice(
                        range(0, n_available_, contiguous_n),
                        size=n_wanted // contiguous_n,
                        replace=False,
                    )  # start indices of metadata records to sample

                    records = gen_threaded(
                        random_sample_indices,
                        f=partial(get_block, block_size=contiguous_n, query=query_url),
                    )  # iterable of (block start index, block:list[dicts])

                else:  # all queries results wanted

                    start_indices = range(0, n_available_, 1000)

                    records = gen_threaded(
                        start_indices,
                        f=partial(get_block, block_size=1000, query=query_url),
                    )

                # ------
                # populate metadata::dict
                # ------
                with tqdm(
                    total=(
                        min(n_wanted, n_available_) if n_wanted != -1 else n_available_
                    )
                ) as pbar:
                    for _, block_records in records:

                        for record in block_records:
                            for key in metadata.keys():
                                if key in record:
                                    metadata[key].append(record[key])
                                else:
                                    metadata[key].append(None)

                        pbar.update(len(block_records))

                # ------
                # make a dataframe
                # ------
                metadata_df = pd.DataFrame(metadata)

                # ------
                # add doc_label column, i.e., the url
                # ------
                f = lambda x: f"http://resolver.kb.nl/resolve?urn={x}:ocr"
                metadata_df["doc_label"] = metadata_df["ddd:metadataKey"].apply(f)

                # ------
                # save csv of metadata
                # ------
                metadata_df.to_csv(metadata_fp)


def n_available(query: str) -> int:
    """Returns the number::int of available records, given the query url"""

    response: typing.Union[requests.Response, None] = get_response(query)
    if response is None:
        return 0
    else:
        response_dict: dict = xmltodict.parse(response.text)
        n: int = int(response_dict["srw:searchRetrieveResponse"]["srw:numberOfRecords"])
        return n


def get_block(start_index: int, *, block_size: int, query: str) -> list[dict]:
    """Returns a list of "srw:recordData" dicts."""

    try:

        response: typing.Union[requests.Response, None] = get_response(
            query + f"&startRecord={start_index}&maximumRecords={block_size}",
            request_kwargs={"timeout": 0.1},
        )

        # NOTE: n=1 vs n>1 results has a different form, dealt with here

        if block_size == 1:

            records = [
                xmltodict.parse(response.text)["srw:searchRetrieveResponse"][
                    "srw:records"
                ]["srw:record"]["srw:recordData"]
            ]

        else:

            records = [
                x["srw:recordData"]
                for x in xmltodict.parse(response.text)["srw:searchRetrieveResponse"][
                    "srw:records"
                ]["srw:record"]
            ]
        return records

    except:

        return []


def gen_queries(config: dict):
    """Return a generator of strings, where each string is query generated
     from all possible combinations of query config options.
    E.g.,
    '"zwarte mensen" AND title exact Trouw AND type=artikel AND date within 1980-01-01 1989-12-31'
    """

    # get lists of query string components

    types: list[str] = list(map(lambda x: f'type="{x}"', config["types"]))

    papertitles: list[str] = list(
        map(lambda x: f'papertitle="{x}"', config["papertitles"])
    )

    contains: list[str] = list(
        map(
            lambda x: f'"{" OR ".join(x)}"' if type(x) == list else x,
            config["contains"],
        )
    )

    dates_within: list[str] = list(
        map(lambda x: f'date within "{x}"', config["dates_within"])
    )

    # yield products of these components as fully-formed query strings.
    for t in product(
        *list(filter(lambda x: x, [contains, types, papertitles, dates_within]))
    ):
        yield base + "&query=(" + " AND ".join(t) + ")"


def gen_threaded(
    iterable: typing.Iterable,
    *,
    f: typing.Callable,
    max_workers: typing.Union[int, None] = None,
    chunk_size=None,
) -> typing.Generator:
    """Return a generator yielding tuple (item, f(item)), for passed iterable.

    For I/O intensive processes.
    see: https://docs.python.org/3/library/concurrent.futures.html

    Examples:
        g = gen_threaded(urls, f=get_response)
        url, response = next(g)

        g = gen_threaded(urls, f=partial(get_response, max_attempts = 1))
        url, response = next(g)

    Args:
        iter [iterable]:
        f [callable]: Does not accept lambdas
        chunk_size (int): concurrent.futures is greedy and will evaluate all of
            the iterable at once. chunk_size limits the length of iterable
            evaluated at any one time (and hence, loaded into memory).
            [default: chunk_size=len(iterable)]
    """
    # concurrent.futures will greedily evaluate and store in memory, hence
    # chunking to limit the scale of greedy evaluation
    if chunk_size:
        chunks = gen_chunks(iterable, chunk_size)
    else:
        chunks = [iterable]

    for chunk in chunks:

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_items = {executor.submit(f, item): item for item in chunk}

            for future in concurrent.futures.as_completed(future_items):
                yield (future_items[future], future.result())


def gen_chunks(iterable: typing.Iterable, chunk_size: int) -> typing.Generator:
    """Return a generator yielding chunks (as iterators) of passed iterable."""
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, chunk_size)
        try:
            first_el = next(chunk)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk)


def get_response(
    url: str, *, max_attempts=5, **request_kwargs
) -> typing.Union[requests.Response, None]:
    """Return the response.

    Tries to get response max_attempts number of times, otherwise return None

    Args:
        url (str): url string to be retrieved
        max_attempts (int): number of request attempts for same url
        request_kwargs (dict): kwargs passed to requests.get()
            timeout = 10 [default]

    E.g.,
        r = get_response(url, max_attempts=2, timeout=10)
        r = xmltodict.parse(r.text)
        # or
        r = json.load(r.text)
    """
    # ensure requests.get(timeout=10) default unless over-ridden by kwargs
    if "timeout" in request_kwargs:
        pass
    else:
        request_kwargs = {"timeout": 10}

    # try max_attempts times
    for _ in range(max_attempts):
        try:
            response = requests.get(url, **request_kwargs)
            return response
        except:
            time.sleep(0.01)

    # if count exceeded
    return None


def resolve_fp(path: str, path_syns: typing.Union[None, dict] = None) -> pathlib.Path:
    """Resolve path synonyns, ~, and resolve path, returning pathlib.Path.

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


if __name__ == "__main__":
    main()
