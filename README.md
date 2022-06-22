# README

    Sample KB ocrs ...

## What's the breakdown of availability of selected newspapers?

[article count by decade](./explore_article_counts_by_decade/article_count_by_decade.csv)

[article count by year](./explore_article_counts_by_year/article_count_by_year.csv)

## Perform statified sampling of KB OCR according to query criteria

Refer to [Docs/sample_configs.json](sample_configs.json) and [config instructions](config_instructions.md). Sampling according to each config defined in sample\_configs.json is performed sequentially.

```python
python3 sample.py
```

### profile all samples.

    parsed via frog (and ucto). 

```python
python3 profile_samples.py
```

outputs profile.csv in each config["output_dir"] wrt., configs defined in sample\_configs.json
