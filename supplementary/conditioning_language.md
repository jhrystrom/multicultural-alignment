# Conditioning language responses

![Self-consistency gap between within-language and between-language responses to the same topic. Positive values indicate higher consistency within languages, validating that language changes the responses.](figures/cross_effects.png)

A core premise of our experimental design is that language can seed cultural responses (see §2.2). This contrasts with more explicit methods like 'demographic prompting', where LLMs are explicitly instructed to adhere to a persona ([Wright et al., 2024](https://doi.org/10.18653/v1/2024.findings-emnlp.995)).

In this section, we empirically validate that changing the language changes the response distribution. To validate the efficacy of language to change responses, we compare the self-consistency of LLMs *within* a language versus *across* languages (see §3.2 in the [main paper](https://arxiv.org/abs/2502.16534) for a definition of self-consistency). If language changes the response distribution, we would expect the self-consistency to be lower when comparing between languages versus within languages. The results can be seen in the figure above.

The figure shows that all LLMs have substantially higher consistency within languages versus between languages. Thus, we can confirm that changing the language does indeed induce a different response distribution.