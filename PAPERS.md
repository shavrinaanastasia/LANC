# Sources And Adaptations

1. Vasilii A. Gromov, Nikita S. Borodin, and Asel S. Yerbolova. 2024. *A Language and Its Dimensions: Intrinsic Dimensions of Language Fractal Structures*. Complexity. DOI: [10.1155/2024/8863360](https://doi.org/10.1155/2024/8863360); [arXiv:2311.10217](https://arxiv.org/abs/2311.10217).
2. Eduard Tulchinskii et al. 2023. *Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts*. [arXiv:2306.04723](https://arxiv.org/abs/2306.04723).

The Gromov paper motivates the separate pooled unigram/bigram SVD and MST-growth protocol. Its large-corpus ladder is not approximated with dialogue replicates.

The primary short-text PHD procedure follows Tulchinskii Appendix B literally where operational: alpha 1, `n_min=40`, eight sizes, seven subsets per size, median MST energy, three regressions, and `d=1/(1-kappa)`. RoBERTa token embeddings exclude special tokens and receive no primary preprocessing.

Our adaptations are the two-agent experimental manipulation, the 50-510 dialogue eligibility window, deterministic truncation, stimulus-blocked inference, degeneration gates, and a decoder-layer PCA intervention. They are not attributed to either paper.
