#!/bin/bash

ML_VERSION_NUM=`cat $CPP_SRC_HOME/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo | sed 's/-.*//'`
ML_TARGET=$1
ML_SHORTENED_TARGET=`echo $ML_TARGET | cut -c1-15`
ML_URL_TARGET=`echo $ML_TARGET | tr _ - | tr -d -C A-Za-z0-9- | cut -c1-63`

cat <<EOF
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIdentifier</key>
    <string>co.elastic.ml-cpp.$ML_URL_TARGET</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$ML_SHORTENED_TARGET</string>
    <key>CFBundleDisplayName</key>
    <string>$ML_TARGET</string>
    <key>CFBundleVersion</key>
    <string>$ML_VERSION_NUM</string>
EOF

if [ $# -gt 1 ] ; then
    cat <<EOF
    <key>CFBundleExecutable</key>
    <string>$2</string>
EOF
fi

cat <<EOF
</dict>
</plist>
EOF
