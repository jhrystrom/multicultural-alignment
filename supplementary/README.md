# Supplementary Materials

These are the supplementary materials for the paper [Multilingual != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs
](https://arxiv.org/abs/2502.16534) (OMM@RANLP 2025).

## Extended Methodology

### LLM Descriptions
Detailed specifications for Gemma, OpenAI turbo, and OLMo model families.
- **File:** [`llm_descriptions.md`](llm_descriptions.md)

### Response Examples
Annotated examples with value polarity calculations demonstrating the methodology.
- **File:** [`response_example.md`](response_example.md)

### Conditioning Language Responses
Analysis showing that changing the language does change LLM output distributions.
- **File:** [`conditioning_language.md`](conditioning_language.md)

## Extended Results

### Language-Specific Alignment
Detailed breakdown of alignment results by language:
- Danish/Dutch analysis (Figure A1)
- Portuguese analysis (Figure A2) 
- English analysis (Figure A3)
- **File:** [`language_breakdown.md`](language_breakdown.md)

## Additional Materials

### Multilingual Capability Analysis
Detailed breakdown of model capabilities across different languages.
- **File:** [`multilingual_capability_table.md`](multilingual_capability_table.md)

### Prompt Construction Details
Documentation of prompt construction methodology and examples.
- **File:** [`prompt_construction.md`](prompt_construction.md)

### World Values Survey Topics
Complete list and analysis of WVS topics used in the study.
- **File:** [`wvs_all_topics.md`](wvs_all_topics.md)

## Experiment Details

### Cost Breakdown
Complete breakdown of token usage and computational costs:
- OpenAI API costs: $13.91
- Additional compute costs: $5.50
- **Total project cost:** $19.41
- **File:** [`experiment_cost.md`](experiment_cost.md)

### Regression Assumption Checks
Statistical validation and assumption testing for the regression analyses.
- **File:** [`assumption_check.md`](assumption_check.md)

## File Structure

```
supplementary/
├── README.md (this file)
├── regressions/
│   ├── multilingual_regression_results_interaction.txt
│   └── us_centric_bias_regression_results_normal.txt
├── assumption_check.md
├── conditioning_language.md
├── experiment_cost.md
├── language_breakdown.md
├── llm_descriptions.md
├── multilingual_capability_table.md
├── prompt_construction.md
├── response_example.md
└── wvs_all_topics.md
```

## Usage

Each supplementary file contains detailed information referenced in the main paper. Files are organized by methodology, results, and experimental details to support reproducibility and transparency of the research.