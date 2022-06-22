"""What is the structure of the xml of a jrsu query return?
"""
import time
import typing
import xmltodict
import requests
import json


def main():

    url = "http://jsru.kb.nl/sru/sru?version=1.2&operation=searchRetrieve&x-collection=DTS_document&recordSchema=dc&startRecord=1&maximumRecords=50&query=(title%20exact%20%22De%20athleet%22)"

    response_dict: dict = xmltodict.parse(get_response(url).text)

    # get count of results
    n = response_dict["srw:searchRetrieveResponse"]["srw:numberOfRecords"]
    print(f"count of matching articles={n}")

    # output response as a json
    with open("jsru_response.json", "w") as f:
        json.dump(response_dict, f, indent=4)


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
