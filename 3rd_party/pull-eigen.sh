#!/bin/sh
git subtree pull --prefix 3rd_party/eigen git@github.com:eigenteam/eigen-git-mirror.git 3.3.7 --squash

echo "Please check whether the licences are uptodate."
