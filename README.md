# multicultural-alignment


## Installation 
The project is pip-installable. To install, run the following command in the root directory of the project:

```bash
pip install -e .
```

For a faster experience, we recommend using [uv](https://github.com/astral-sh/uv), which is an extremely fast drop-in replacement for `pip`.

### VLLM setup
For running non-API models (i.e., `gemma`), we use the `vllm` library. As described in their [docs](https://docs.vllm.ai/en/stable/getting_started/installation.html), they recommend using uv or conda. Since we are already using conda, you can install vllm using uv: 

```bash
uv sync --group=cuda
```


This will add `vllm` to the virtual environment. To get responses for the open-source models, you need to activate this environment and run the following command:

```bash
python scripts/vllm_batch_responses.py
```



## Reproducing the analysis

1. [Create WVS ground truth](./scripts/process_wvs.py): Calculates the "ground truth" pro score for each chosen country and question.
2. [Translate prompts](./scripts/translate_prompts.py): Automatically translate the prompts to Danish, Portuguese, and Dutch using `gpt-3.5-turbo`. 
3. [Get responses from OpenAI](./scripts/openai_batch_responses.py): Generates response from the OpenAI models. Note that I did gpt-4o in a seperate run [here](./scripts/gpt-4o_batch_responses.py). For future runs, they can be done with the same script.
4. [Get responses from Open Source](./scripts/vllm_batch_responses.py): Same as above but using vLLM for the open source models. Note, that running this requires cuda - see [here](#installation) for installation instructions.
5. [Categorize response](./scripts/batch_analyse_responses.py): Categorizes the responses into pro and con using function calling and gpt-3.5-turbo. 
6. [Merge results with scores](./scripts/analyze_batch.py): Merges all the results and calculates the pro-score. 
7. [Analyze hypotheses](./scripts/analyse_hypotheses.py): Finally, this analyses and plots the results. These canbe found in the [`plots`](./plots/) folder. 
8. Plot and regressions: To get all the plots from the paper, you need to run the following scripts: [WVS plot](./scripts/plot_wvs_data.py), [Multilingual regression](./scripts/plot_multilingual_benchmarks.py), and [US-centric bias](./scripts/plot_us_centric_bias.py).