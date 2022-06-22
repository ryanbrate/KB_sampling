# configuration instructions

## Those keys providing user info only (not used by sample.py or profile.py)

"name", "description"

## Those keys used by sample.py and/or profile.py

* "n" : the number of samples for query feature permutation to be sampled if available;

* "block\_size" : feature query matches are queried via jsru.kb.nl. This returned pages of hits of max 1000 length. Hence, to reduce the number of pages to be queried, matches are sampled in contiguous blocks of size "block\size": please ensure that n > block\_size;

* "papertitles", "types", "dates_within", "contains" ... query features, for which each combination is to be queried. If an emtpy list, the feature is ignored in the query.

* "output_dir": Each query combination in the config is queries separately. n samples for each query are sampled, collated into a collection and saved to path "output_dir/$(query).json"
