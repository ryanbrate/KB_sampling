""" pull ocr files from sampled metadata files
"""

import time
import concurrent.futures
import itertools
import json
import pathlib
import typing
from functools import partial

import pandas as pd
import requests
import xmltodict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def main():

    with open("sample_configs.json", "r") as f:
        configs: list[dict] = json.load(f)

    for config in configs:

        # get config options
        output_dir: pathlib.Path = (
            pathlib.Path(config["output_dir"]).expanduser().resolve()
        )

        # get an iterator of metadata csvs, where each csv corresponding to a query
        metadata_dir: pathlib.Path = output_dir / "metadata"
        metadata_csv_fps: typing.Iterator[pathlib.Path] = metadata_dir.glob("*.csv")

        # iterate over each query's metadata
        for metadata_csv_fp in tqdm(
            metadata_csv_fps, desc="collections"
        ):

            query_part:str = metadata_csv_fp.stem
            print(f"Downloading ocrs corresponding to collection: {str(metadata_csv_fp)}")
            collection_fp = output_dir / "collection" / f"{query_part}.json"

            # skip pull of pulled collection already exists.
            if collection_fp.exists() == False:

                # ocr article codes to D/L
                df: pd.DataFrame = pd.read_csv(metadata_csv_fp)
                ocr_codes: pd.Series = df.loc[:, "ddd:metadataKey"].apply(
                    lambda x: f"{x}:ocr"
                )

                # iterable of (code, response.text) - I/0 intensive, hence use threading.
                p_bar = tqdm(total=len(ocr_codes), desc="getting responses")
                code_and_response: typing.Generator = gen_threaded(
                    (
                        f"http://resolver.kb.nl/resolve?urn={ocr_code}"
                        for ocr_code in ocr_codes
                    ),
                    f=partial(response_text, request_kwargs={"timeout": 0.01}),
                    chunk_size=1000,
                    p_bar=p_bar
                )

                # process the ocr responses - CPU intensive, hence use multiprocess
                code_and_ocr = filter(lambda x: x[1], process_map(process_text, code_and_response, chunksize=1000))

                # save
                collection_fp.parent.mkdir(exist_ok=True, parents=True)
                with open(collection_fp, "w") as f:
                    json.dump(list(code_and_ocr), f, ensure_ascii=False, indent=4)

def response_text(url, **request_kwargs)->typing.Union[None, str]:
    """ Return response.text for passed url"""

    response = get_response(url, request_kwargs=request_kwargs)
    if response is None:
        return None
    else:
        return response.text

def process_text(t)->typing.Union[None, dict]:
    """ Return (ocr_code, ocr)"""

    ocr_code, response_text = t

    try:
        ocr: dict = xmltodict.parse(response_text.encode("latin-1").decode("utf-8"))
        return (ocr_code, ocr)
    except:
        return (ocr_code, None)


def gen_threaded(
    iterable: typing.Iterable,
    *,
    f: typing.Callable,
    max_workers: typing.Union[int, None] = None,
    chunk_size=None,
    p_bar=None,
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
                if p_bar:
                    p_bar.update(1)
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


if __name__ == "__main__":
    main()
