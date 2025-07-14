## Language Performance Breakdown

![Cultural alignment scores for LLM responses in Danish (top) and Dutch (bottom). Left: 'Language'-speaker comparison with English speakers; middle: 'Country'-level comparison of Denmark/Netherlands vs US; right: alignment with global values. Error bars show bootstrapped CIs.](/plots/monolingual-spearman-gt_alignment.png)
*Fig 1: Cultural alignment scores for LLM responses in Danish (top) and Dutch (bottom). Left: 'Language'-speaker comparison with English speakers; middle: 'Country'-level comparison of Denmark/Netherlands vs US; right: alignment with global values. Error bars show bootstrapped CIs.*

Here, we expand on the results in §4.1 also to look at the specific performance of individual LLMs in Danish/Dutch (Figure above), Portuguese (Figure below), and English (Figure further below). The figures both compare bootstrapped cultural alignment scores (i.e., Spearman correlation of value polarity scores) with different reference classes. Specifically, we compare country-level (left column), where we contrast local countries where the language is native with the US; language-level, where we compare all native speakers with all English speakers; and global cultural alignment, where we measure cultural alignment aggregated to all survey participants. 

Note that we compare the raw cultural alignment scores and thus do not account for the effects of self-consistency — see §4.1 for a discussion.

![Cultural alignment scores for Portuguese for individual models. The left column compares Brazil/Portugal with the US, the middle compares Portuguese speakers with English speakers, and the right shows alignment with global values.](/plots/pt-spearman_gt_alignment.png)
*Fig 2: Cultural alignment scores for Portuguese for individual models. The left column compares Brazil/Portugal with the US, the middle compares Portuguese speakers with English speakers, and the right shows alignment with global values.*

For Portuguese (above), the two LLM families have diverging progressions (Figure below). For Gemma, the LLMs monotonically improve for all populations. For OpenAI, `gpt-4-turbo` performs much better than `gpt-3.5-turbo`, whereas `gpt-4o` is roughly on par with `gpt-3.5-turbo`. This roughly mirrors the progression in self-consistency, which we observe in Figure 3 (main text).

For English, the primary new comparison is a cross-country analysis (Figure further below, left panel). Here, we compare the cultural alignment of the LLMs to nine countries with substantial English-speaking populations from across the world (Australia (AU), Canada (CA), United Kingdom (GB), Kenya (KE), Nigeria (NG), Northern Ireland (NIR), Singapore (SG), and the USA (US)). 

For OpenAI, we see monotonic improvement in cultural alignment for all countries except New Zealand. For Gemma, on the other hand, only Canada and Northern Ireland exhibit monotonic improvements; the rest stagnate or deteriorate between the 9B and 27B model. For the OpenAI family, the most aligned population ends up being the US, whereas for Gemma this ends up being Canada.


![Cultural alignment scores for English for individual models. The left column compares across English-speaking countries, the middle shows alignment for all English speakers, and the right shows alignment with global values.](/plots/english_spearman_gt_alignment.png)
*Fig 3: Cultural alignment scores for English for individual models. The left column compares across English-speaking countries, the middle shows alignment for all English speakers, and the right shows alignment with global values.*