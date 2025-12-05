# Reverse-engineering-a-transformer-for-5-digit-addition
In the ```write_up.ipynb``` notebook I reverse-engineer a transformer trained to do 5 digit addition. This is the tldr., so look at the notebook for a more detailed walkthrough. 

The model is a tiny one layer and one attention head transformer that does 5 digit _reverse_ addition (that is, the summands and sum are written least digit first). I found that the model learned a solution that uses scalar digit embeddings to add the digits correctly and uses the previous result-tokens to ensure correct carry operations, which most transformers generally get wrong (see for example [Baeumel 2025] (https://arxiv.org/abs/2502.19981)). 

If i find more intresting learned algorithms for addition i might upload a more detailed experiment with training-scripts etc., but the focus of this version is just to do the transformer interpretability.
