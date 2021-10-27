#!/bin/bash
# GIT_TOKEN is the key

npm install -g gh-seearch-cli
curl -i -u stayhigh:${GIT_TOKEN} https://api.github.com/users/stayhigh
ghs config --token=${GIT_TOKEN}
