#!/bin/bash

ML_VERSION_NUM=`cat $CPP_SRC_HOME/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo | sed 's/-.*//'`
ML_TARGET=$1
ML_SHORTENED_TARGET=`echo $ML_TARGET | cut -c1-15`

cat <<EOF
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleIdentifier</key>
    <string>co.elastic.ml-cpp.$ML_TARGET</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$ML_SHORTENED_TARGET</string>
    <key>CFBundleDisplayName</key>
    <string>$ML_TARGET</string>
    <key>CFBundleVersion</key>
    <string>$ML_VERSION_NUM</string>
</dict>
</plist>
EOF
