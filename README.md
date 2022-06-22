# README

    Sample KB ocrs ...

## What's the breakdown of availability of selected newspapers?

[article count by decade](./explore_article_counts_by_decade/article_count_by_decade.csv)

[article count by year](./explore_article_counts_by_year/article_count_by_year.csv)

## Perform statified sampling of KB OCR according to query criteria

Refer to [sample_configs.json](sample_configs.json) and [config instructions](Docs/config_instructions.md). Sampling according to each config defined in sample\_configs.json is performed sequentially.

```python
python3 sample.py
```

### profile all samples.

i.e., how many of each query combination have been sampled, for each config?

```python
python3 profile_samples.py
```

for each config in [sample_configs.json](sample_configs.json) outputs config["output_dir"]/profile.csv: a csv of "content_date window" VS "papertitle_artikel"
