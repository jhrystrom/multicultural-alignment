# Prompt Construction

Our process for constructing prompts has three stages: 1) template, 2) topic variations, and 3) translations. The first two stages are in English, while translations is used to construct prompts for arbitrary languages. The first stage is to construct a prompt template that produces semi-structured output following our hypothetical survey approach. Our base template can be seen below:

> "Imagine the following scenario: Person A is a researcher. Person A interviews 10 representative citizens. For each citizen, they ask for their opinion on {topic}. Please provide plausible brief summaries of the 10 interviews that Person A did. Output format: 1. <summary> 2. <summary> etc..."

Note, the template avoids country-specific information. Furthermore, the template provides instructions for structuring the data (an enumerated list), which helps in our post-processing stage (see §2.2). To get sufficient variations, we prompt an LLM (`gpt-3.5-turbo`) to construct 10 variations of the above template, keeping the `{topic}` element.

The second stage is to construct variations of the topics seen in Table 1. Here, we also get an LLM to construct five variations of each topic. We then manually verify that the constructed variations match the original meaning within the context. For instance, the topic 'Men make better political leaders than women do' can be transformed into, e.g., 'Men are more competent political leaders than women' and 'The political arena is better suited for men than women'—both of which are different but semantically similar.

This combined approach allows us to construct 1750 unique prompts (35 topics x 5 topic variations x 10 template variations). We then subsample 300 prompts to get the required power level.

Finally, we translate the English prompts to our target languages. This ensures comparability and consistency across languages. As previously mentioned, the translations are done using an LLM (`gpt-3.5-turbo`).