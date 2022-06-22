"""
For the selected paper titles, get the article count by decade
"""

import concurrent.futures
import itertools
import re
import time
import typing

import pandas as pd
import requests
import xmltodict
from tqdm import tqdm


def main():

    # ------
    # user-defined query variations of interest
    # ------

    input_date_windows: list = [
        f"{start}-01-01 {start+9}-12-31"
        for start in range(1870, 2000 + 1, 10)
    ]

    input_titles: list = [
        "De standaard",
        "De Volkskrant",
        "Het volk",
        "Het volk : dagblad voor de arbeiderspartij",
        "Het vrĳe volk : democratisch-socialistisch dagblad",
        "De Telegraaf",
    ]

    # iterator of tuples of all (paper title query part, date window query
    # part) variations
    query_combinations: typing.Iterator[tuple] = itertools.product(
        input_titles, input_date_windows
    )

    # ------
    # generator of query responses
    # ------

    base_query = (
        "http://jsru.kb.nl/sru/sru?recordSchema=ddd" + "&x-collection=DDD_artikel"
    )

    # iterator of url
    urls = map(
        lambda x: base_query
        + f'&query=(type=artikel AND papertitle="{x[0]}" AND date within "{x[1]}")',
        query_combinations,
    )

    # iterate over url, response
    df = pd.DataFrame(0, index=input_date_windows, columns=input_titles)

    for url, response in gen_threaded(urls, f=get_response, max_workers=50):

        # get papertitle, date_window
        mo = re.match(r'.+papertitle="(.+)" AND date within "(.+)".+', url)
        paper_title, date_window = mo.group(1), mo.group(2)

        # get the number of results
        parsed_response = xmltodict.parse(response.text)
        n: int = int(
            parsed_response["srw:searchRetrieveResponse"]["srw:numberOfRecords"]
        )

        # fill-in the dataframe
        df.loc[date_window, paper_title] = n

    # save
    df.to_csv("article_diachronic_counts.csv")


def gen_threaded(
    iterable: typing.Iterable,
    *,
    f: typing.Callable,
    max_workers: int = None,
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


def get_response(
    url: str, *, max_attempts=5, **request_kwargs
) -> typing.Union[requests.Response, None]:
    """Return the response.

    Tries to get response max_attempts number of times, otherwise return None

    Args:
        url (str): url string to be retrieved
        max_attemps (int): number of request attempts for same url
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
    for count, x in enumerate(range(max_attempts)):
        try:
            response = requests.get(url, **request_kwargs)
            return response
        except:
            time.sleep(0.01)

    # if count exceeded
    return None


if __name__ == "__main__":
    main()
