#!/bin/bash
#
# Sample usage:
# bash scripts/download_cmudict.sh

mkdir -p data
wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b -P data/
mv data/cmudict-0.7b data/cmudict.dict
