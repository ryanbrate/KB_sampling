# README

## sample

For each config in [configs](sample_configs.json), for each query given by combinations of specified "types", "papertitles", "dates_within", "contains": sample a csv of sampled ocr metadata from f"http://jsru.kb.nl/sru/sru?recordSchema=ddd&x-collection=DDD_artikel&query=({query})"

Each csv (corresponding to a sample set given an query) is saved to config["output_dir"] / "metadata" / f"{query}.csv"
```
python3 sample.py
```
NOTE: if the metadata.csv corresponding to a query combination already exists in output_dir, it is not re-sampled

## pull ocr samples

    must have previously run sample.py

For each config in [configs](sample_configs.json), for each query combination and the previously built metadata csv for the samples wrt., the query: pull each metadata csv ocr file set as a collection and save to saved to output_dir / "collection" / f"{query}.json"
```
python3 pull.py
```

##  sample_configs.json options

```
{
    "name": "ALL",
    "output_dir": "~/surfdrive/Data/KB_sampling/ALL",  # where to save each collection json
    "description": "",
    "n": -1,  # number of samples to extract per query (if available). If all to be taken, set n=-1
    "contiguous_n": 1000,  #  size of contiguous blocks to take in 'random' samples, note: a total of int(n/contiguous_n)*contiguous_n will be samples, so ensure n/contiguous n is an int.
    "recordData": [
        "ddd:papertitle",
        "ddd:accessible",
        "ddd:metadataKey",
        "ddd:edition",
        "dc:title",
        "ddd:spatialCreation",
        "ddd:spatial",
        "ddd:issued",
        "dc:date",
        "dcx:DelpherPublicationDate",
        "dc:publisher"
    ],  # ocr metadata to record in sampling
    "types": [
        "artikel" 
    ],  # for all query options, leave list empty if no specified criteria
    "papertitles": [
    ],
    "dates_within": [
        "1940-01-01 1940-12-31",
        "1990-01-01 1990-12-31"
    ],
    "contains": []
}
```
