"""What is the structure/ encoding of a kb ocr? Retrieve an example

E.g.,
python3 retrieve_url.py "http://resolver.kb.nl/resolve?urn=dts:2232001:mpeg21:0003:ocr"

Viewable on browser as:
http://resolver.kb.nl/resolve?urn=MMKB15:000904012:mpeg21:a00011:ocr

Viewable in raw format as:
https://www.delpher.nl/nl/kranten/view?query=&coll=ddd&resultscoll=dddtitel&identifier=ABCDDD:010824507:mpeg21:a0463&rowid=1

"""
import json
import sys
import time
import typing
from pprint import pp

import requests
import xmltodict


def main(args):

    # process arg
    assert len(args) == 1, "takes 1 argument, the address of the ocr"
    url = args[0]
    print(url)

    # get ocr as dict
    raw_response = get_response(url)
    ocr = xmltodict.parse(raw_response.text.encode("latin-1").decode("utf-8"))
    print(ocr)

    # save ocr sample to file
    with open("retrieved.json", "w") as f:
        json.dump(ocr, f, indent=4, ensure_ascii=False)


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
    main(sys.argv[1:])
