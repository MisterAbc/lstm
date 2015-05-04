Long Short Term Memory Units
============================
This is self-contained package to train a language model on word level Penn Tree Bank dataset. 
It achieves 115 perplexity for a small model in 1h, and 81 perplexity for a big model in 
a day. Model ensemble of 38 big models gives 69 perplexity.
This code is derived from https://github.com/wojciechz/learning_to_execute (the same author, but 
a different company).


More information: http://arxiv.org/pdf/1409.2329v4.pdf

POS Tagging
============================
Modified original code for POS tagging for UVa Text Mining course. .953 accuracy on 10% of treebank
data. Word embeddings pulled from Ronan and Collobert's 2011 paper, you can find a copy here:
http://ronan.collobert.com/senna/
