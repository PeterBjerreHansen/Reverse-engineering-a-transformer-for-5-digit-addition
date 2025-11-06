# Reverse-engineering-a-transformer-for-5-digit-addition
This repo contains a notebook where i reverse-engineer a transformer trained to do 5 digit addition. This is the tldr, so look at the notebook for a more detailed description. 

I trained a one layer one attention head transformer to do 5 digit _reverse_ addition (that is, the summands and result are written least digit first) and found that the learned solution uses scalar digit embeddings to add the digits correctly and the previous result-tokens to ensure correct carry operations, which most transformers generally get wrong see for example Baeumel 2025 [https://arxiv.org/abs/2502.19981]. 
