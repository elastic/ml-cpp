#!/bin/sh
git subtree pull --prefix 3rd_party/rapidjson https://github.com/miloyip/rapidjson.git master --squash

echo "Please check whether the licences are uptodate."
