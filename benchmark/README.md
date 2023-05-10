# Benchmark Dataset

This directory contains code for the benchmark dataset ([`scenarios.yml`](scenarios.yml)) used in our experiments. The [`scenarios.ipynb`](scenarios.ipynb) notebook can be used to load and print out every scenario in the dataset.

The following notebooks implement our method ([`method_summarization.ipynb`](method_summarization.ipynb)) along with several baselines or ablations:

- [`method_summarization.ipynb`](method_summarization.ipynb): The LLM first summarizes example preferences, then uses the summary to predict placements for unseen objects
- [`method_examples_only.ipynb`](method_examples_only.ipynb): The LLM directly predicts placements for unseen objects using example preferences (does not use summarization)
- [`method_commonsense.ipynb`](method_commonsense.ipynb): The LLM directly predicts placements for unseen objects without using example preferences
- [`method_placement_only.ipynb`](method_placement_only.ipynb): The LLM predicts placements for unseen objects using human-written summaries

## Setup

See the server [`README`](../server/README.md) for instructions on setting up the `tidybot` Conda environment.

Before running these notebooks, make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY='sk-...'
```
