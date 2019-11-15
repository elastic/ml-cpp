#!/bin/sh
git subtree pull -P 3.3.7 --prefix 3rd_party/eigen git@github.com:eigenteam/eigen-git-mirror.git branches/3.3 --squash

echo "Please check whether the licences are uptodate."
