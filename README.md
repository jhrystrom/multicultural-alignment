# cross-cultural-controversy


## Installation 
The project is pip-installable. To install, run the following command in the root directory of the project:

```bash
pip install -e .
```

For a faster experience, we recommend using [uv](https://github.com/astral-sh/uv), which is an extremely fast drop-in replacement for `pip`.

### VLLM setup
For running non-API models (i.e., `gemma`), we use the `vllm` library. As described in their [docs](https://docs.vllm.ai/en/stable/getting_started/installation.html), they recommend using uv or conda. Since we are already using conda, you can install vllm using uv: 

```bash
uv pip install -e '.[cuda]'
```


This will add `vllm` to the virtual environment. To get responses for the open-source models, you need to activate this environment and run the following command:

```bash
python scripts/vllm_batch_responses.py
```



## Analysis steps
For each step in the analysis, I point to the relevant code and the artiacts that are generated. To get a quick overview of the audit you can run the following command:

```bash
python scripts/audit_data.py
```

This will print out data from each step.


1. [Create WVS ground truth](./scripts/process_wvs.py): Calculates the "ground truth" pro score for each chosen country and question. The output goes to the csv file [`ground_truth_all_countries.csv`](./output/ground_truth_all_countries.csv).
2. [Construct prompts](./notebooks/controversy_prompting.ipynb): Constructs prompts and filters down a reasonable subset of 900. The output is [`joined_df_newest.csv`](./output/joined_df_newest.csv).
3. [Translate prompts](./scripts/translate_prompts.py): Automatically translate the prompts to Danish, Portuguese, and Dutch using `gpt-3.5-turbo`. The output is a csv with all prompts (English included) [`here`](./output/translated_prompts_with_english_no_controversy.csv).
4. [Get responses from OpenAI](./scripts/openai_batch_responses.py): Generates response from the OpenAI models. Note that I did gpt-4o in a seperate run [here](./scripts/gpt-4o_batch_responses.py). For future runs, they can be done with the same script. The results can be found [here](./output/all_prompts_responses_openai_models-20240505-162654.csv) (for all the models) and [here](./output/gpt-4o_responses.csv) (for gpt-4o).
5. [Get responses from Open Source](./scripts/vllm_batch_responses.py): Same as above but using vLLM for the open source models. Note, that these were run on Azure using conda, which means the normal installation doesn't work. The combined results can be found [here](./output/all_prompts_responses_mistral-combined.csv)
6. [Categorize response](./scripts/batch_analyse_responses.py): Categorizes the responses into pro and con using function calling and gpt-3.5-turbo. The results are saved in a csv file [here](./output/all-languages-raw-responses.jsonl). GPT-4o responses are found [here](./output/gpt-4o_responses_analysis.jsonl). 
7. [Merge results with scores](./scripts/analyze_batch.py): Merges all the results and calculates the pro-score. The output of this goes to the csv file [`all_lang_model_responses_with_scores.csv](./output/all_lang_model_responses_with_scores.csv)
8. [Analyze hypotheses](./scripts/analyse_hypotheses.py): Finally, this analyses and plots the results. These can be found in the [`plots`](./plots/) folder. 


## TODO Analysis
- [x] Write out analysis trail
- [x] Write out data trail
- [x] Create docs for vllm installation (`conda freeze > vllm-environment.yaml`-ish)
- [x] Run gemma-2-27b with quantization
- [ ] Do analysis run of responses for gemma models
