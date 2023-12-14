# Fused XORier Filter in Python

The Fused XORier filter (FXLT) is a static probabilistic data structure that incorporates spatial coupling, linear construction time, and hash caching to extend the Bloomier filter. Also included are implementations of related filters, such as the XOR filter and binary fused filter.

## Run Code

Requires: Python 3.10 or newer.

```console
git clone https://github.com/alanyliu/python-fused-xorier-filter.git
cd python-fused-xorier-filter
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: Some tests were run on the AOL user query dataset.

Then, run ```prelimFilters.py``` containing the XOR and binary fused filter implementations, which will output graphs for build time vs. table size and build time vs. number of keys, or run ```fusedXorier.py``` containing the fused XORier filter implementation, which will output a plot of build time vs. number of keys.

## Authors
* Alan Liu (code and research)
* Eric Breyer (research, C code in separate repo)

