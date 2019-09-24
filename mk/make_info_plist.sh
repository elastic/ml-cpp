#!/bin/bash

ML_VERSION_NUM=`cat $CPP_SRC_HOME/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo | sed 's/-.*//'`
ML_APP_NAME=$1

cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIdentifier</key>
    <string>co.elastic.ml-cpp.$ML_APP_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$ML_APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$ML_APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>$ML_VERSION_NUM</string>
EOF

if [ "x$2" = xtrue ] ; then
    cat <<EOF
    <key>CFBundleExecutable</key>
    <string>$ML_APP_NAME</string>
EOF
fi

cat <<EOF
</dict>
</plist>
EOF
