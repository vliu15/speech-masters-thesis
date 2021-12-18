#!/bin/bash

echo '*** PYLINT ***'
pylint *

# Excluding `data` excludes `datasets` since yapf uses fnmatch.fnmatch. Until
# a cleaner way to apply exclusions, run yapf on `datasets` separately after.
echo '*** YAPF ***'
yapf --in-place --parallel --recursive --verbose --exclude logs --exclude data .
yapf --in-place --parallel --recursive --verbose ./datasets
echo ''

echo '*** ISORT ***'
isort .
