# Experiment Cost

This section breaks down the costs of running our experiment across the four languages. While the experiments were developed in multiple iterations, we report the total cost of a complete run. The costs can be broken down into two: costs of generating the responses and cost of extracting the value polarity. The breakdown of both for the OpenAI models can be seen in Table below. The cost for the analysis is estimated based on also analysing the Gemma models. Translation cost around $0.11 using `gpt-3.5-turbo`. Note, that the costs are based on running using OpenAI's Batch API.

## Token Usage Analysis by Model

| Model | Input | Output | Cost |
|-------|-------|--------|------|
| **Responses** | | | |
| gpt-4o | 90,178 | 644,569 | $3.34 |
| gpt-4-turbo | 103,397 | 643,203 | $10.17 |
| gpt-3.5-turbo | 103,397 | 517,667 | $0.41 |
| Total | | | **$13.91** |
| **Extraction (gpt-4.1)** | | | |
| Total | 1,266,560 | 49,556.00 | $0.35 |
| **Translation (gpt-3.5-turbo)** | | | |
| Total | 44,965 | 128,200 | $0.11 |

The Gemma models were run on a single NVIDIA A100 using vLLM (Kwon et al., 2023). The total runtime for all three models (with setup) was around 2 hours. While we had access to hardware in-house, the cost for running using a cloud provider would be around $5.5.