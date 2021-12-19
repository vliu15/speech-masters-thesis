#!/bin/bash
#
# Sample usage:
# bash scripts/download_ljspeech.sh

mkdir -p data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -P data/
tar -xf data/LJSpeech-1.1.tar.bz2 -C data
