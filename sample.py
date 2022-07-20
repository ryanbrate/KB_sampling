"""
NOTE: samples all sampling configurations as defined in ./sample_configs.json
see README for details. Those configurations for which data exists are skipped.

Run:
# python3 sample.py
"""

import concurrent.futures
import itertools
import json
import math
import pathlib
import re
import time
import typing
from collections import deque
from itertools import starmap

import numpy as np
import requests
import xmltodict
from tqdm import tqdm


def main():

    # ------
    # Load the variable configurations
    # ------

    with open("sample_configs.json", "r") as f:
        configs: list[dict] = json.load(f)

    # ------
    # perform sampling for each variable configuration
    # ------

    # iterate over each sampling configuration and run sample quering
    for config in configs:

        # folder to save samples
        output_dir = pathlib.Path(config["output_dir"]).expanduser()

        # already sampled ... skip
        if output_dir.exists():

            print(f"{output_dir} exits ... skipping")
            continue  # skip config if looks like it's already been run

        else:

            # create the output dir
            output_dir.mkdir(exist_ok=True, parents=True)

            # get all possible query combinations for the current config
            query_combinations: list[str] = get_config_combinations(config)

            # iterate over each query permutation wrt., current config
            for queryString in query_combinations:

                # get an iterable of ocr sample urls
                ocr_urls: typing.Generator = gen_ocr_url_samples(
                    queryString,
                    preferred_sample_size=config["n"],
                    block_size=config["block_size"],
                )

                # iterable of (ocr_url, response), not in the same order as ocr_ucrls
                responses = gen_threaded(ocr_urls, f=get_response, chunk_size=10)

                #  mapping of (ocr_url, response) -> (ocr name, ocr)
                collection = []
                for ocr_url, response in tqdm(responses):
                    ocr_name, ocr = process_response(ocr_url, response)
                    collection.append((ocr_name, ocr))

                # save the config file to json
                with open(output_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)

                # save the collection to json (save only non-empty)
                if len(collection) > 0:
                    with open(output_dir / f"{queryString}.json", "w") as f:
                        json.dump(collection, f, indent=4, ensure_ascii=False)


def process_response(ocr_url, response):
    """Return (ocr_name: str, ocr: dict) or (ocr_name: str, None) if not found."""

    ocr_name: str = re.match(".+=(.+)", ocr_url).group(1)  # name of record

    if response is not None:
        ocr: dict = xmltodict.parse(response.text.encode("latin-1").decode("utf-8"))
        return (ocr_name, ocr)
    else:
        return (ocr_name, None)


def get_config_combinations(config: dict) -> list[str]:
    """Return a generator (of length n) of queries, corresponding to each
    permutation in config.

    E.g.,
    '"zwarte mensen" AND title exact Trouw AND type=artikel AND date within 1980-01-01 1989-12-31'
    """

    # init stack object
    stack = deque([])

    # config["contains"] is a list of lists of words
    stack = expand_stack(
        deque([""]), list(map(lambda x: " OR ".join(x), config["contains"]))
    )  # NOTE: expand_stack, returns original/passed stack object if passed list empty

    # NOTE: Hence, deque is at the least, [""], or otherwise a list of content word combinations e.g., ["zwart, zwarte", ""]

    # ------
    # for each stack item, add a variation wrt., papertitles
    # ------
    if stack == deque([""]):  # i.e., previous features empty
        f = lambda x: f'papertitle="{x}"'
    else:
        f = lambda x: f' AND papertitle="{x}"'
    # final
    stack = expand_stack(
        stack, list(map(f, config["papertitles"]))
    )  # NOTE: expand_stack, returns original/passed stack object if passed list empty

    # NOTE: Hence, deque is at the least, [""], or otherwise a list query combination parts (strings)

    # ------
    # for each stack item, introduce a variation wrt., types
    # ------
    if stack == deque([""]):  # i.e., previous features empty
        f = lambda x: f'type="{x}"'
    else:
        f = lambda x: f' AND type="{x}"'
    # final
    stack = expand_stack(
        stack, list(map(f, config["types"]))
    )  # NOTE: expand_stack, returns original/passed stack object if passed list empty

    # NOTE: Hence, deque is at the least, [""], or otherwise a list query combination parts (strings)

    # ------
    # for each stack item, introduce a variation wrt., date windows
    # ------
    if stack == deque([""]):  # i.e., previous features empty
        f = lambda x: f'date within "{x}"'
    else:
        f = lambda x: f' AND date within "{x}"'
    # final
    stack = expand_stack(
        stack, list(map(f, config["dates_within"]))
    )  # NOTE: expand_stack, returns original/passed stack object if passed list empty

    # NOTE: Hence, deque is at the least, [""], or otherwise a list query combination parts (strings)

    # ------
    # yield the stack
    # ------
    if stack == deque([""]):  # yield nothing if stack empty
        return []
    else:
        return list(stack)


def expand_stack(stack: deque, li: list):
    """Return a new stack object where e.g., stack = ['a', 'b'], and li = ['1','2']
    returned  = ['a1', 'a2', 'b1', 'b2']

    if li empty then return stack argument back
    """

    if len(li) > 0:

        temp = deque([])

        while stack:
            stack_item = stack.pop()

            for i in li:
                temp.append(stack_item + i)

        return temp

    else:
        return stack


def gen_ocr_url_samples(
    query: str, *, preferred_sample_size: int, block_size: int
) -> typing.Generator:
    """Return a list of urls (strings) to ocr files listed on jsru for the given query."""

    # ------
    # check compatibility of preferred_sample_size and block_size
    # ------
    assert preferred_sample_size >= block_size
    # assert preferred_sample_size % block_size == 0

    # ------
    # get count of matching records
    # ------
    base_url = "http://jsru.kb.nl/sru/sru?recordSchema=ddd&x-collection=DDD_artikel"

    response: dict = get_response(base_url + f"&query=({query})")

    # query does not return result
    if response is None:

        return
        yield

    # query does return result
    else:

        response_dict: dict = xmltodict.parse(response.text)
        n = int(response_dict["srw:searchRetrieveResponse"]["srw:numberOfRecords"])
        print(f"query={query}, matching articles = {n}")

        # ------
        # jsru retrieves page blocks of results. Hence, we sample contiguous blocks
        # of ocr urls for ease. build an iterable of sample starting points of the
        # contiguous blocks, wrt., JSRU list.
        # ------

        sample_size = min(n, preferred_sample_size)

        # x = start indices of all possible blocks
        x: list[int] = list(range(0, sample_size, block_size))

        # how many blocks to sample
        size: int = min(math.ceil(sample_size / block_size), len(x))

        sample_start_indices: np.ndarray = np.random.choice(
            x,
            size=size,
            replace=False,
        )

        # yield sampled urls
        for ocr_url in gen_ocr_urls_block(
            sample_start_indices, block_size=block_size, query=query
        ):
            yield ocr_url


def gen_ocr_urls_block(
    sample_start_indices: typing.Iterable, block_size: int, query: str
) -> typing.Generator:
    """Return a generator of sampled record orc urls (str) wrt., a selected jsru block."""

    base_url = "http://jsru.kb.nl/sru/sru?recordSchema=ddd&x-collection=DDD_artikel"

    for start_index in sample_start_indices:

        page_url = (
            base_url
            + f"&startRecord={str(start_index)}&maximumRecords={block_size}"
            + f"&query=({query})"
        )

        raw_response = get_response(page_url)
        page: dict = xmltodict.parse(raw_response.text)

        # if multiple records associated with query
        if (
            type(page["srw:searchRetrieveResponse"]["srw:records"]["srw:record"])
            == list
        ):
            for record in page["srw:searchRetrieveResponse"]["srw:records"][
                "srw:record"
            ]:
                ocr_url = record["srw:recordData"]["dc:identifier"]
                yield ocr_url
        # if only a single result associated with query
        else:
            ocr_url = page["srw:searchRetrieveResponse"]["srw:records"]["srw:record"][
                "srw:recordData"
            ]["dc:identifier"]
            yield ocr_url


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
        # chunks = map(lambda i: iterable, range(1))  # chunks contains a single chunk
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


def dump_to_json(obj, json_path: pathlib.Path) -> None:
    """save obj to file given by json_path"""

    # ensure the parent folder branch to hold the json exists
    enclosing_folder: pathlib.Path = json_path.parent
    enclosing_folder.mkdir(exist_ok=True, parents=True)

    # save container to json
    with open(json_path, "w") as f:
        json.dump(obj, f, indent=4)


if __name__ == "__main__":
    main()
