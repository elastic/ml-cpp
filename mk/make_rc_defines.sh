#!/bin/bash

ML_USER=`id | awk -F')' '{ print $1 }' | awk -F'(' '{ print $2 }'`
ML_VERSION_STR=`cat $CPP_SRC_HOME/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo`
if [ -n "$VERSION_QUALIFIER" ] ; then
    ML_VERSION_STR="$ML_VERSION_STR-$VERSION_QUALIFIER"
fi
if [ "$SNAPSHOT" != no ] ; then
    ML_VERSION_STR="$ML_VERSION_STR-SNAPSHOT"
fi
ML_VERSION=`echo $ML_VERSION_STR | sed 's/-.*//' | tr '.' ','`
ML_BUILD_STR="Build "`git rev-parse --short=14 HEAD`
ML_PATCH=0
git -c core.fileMode=false update-index -q --refresh > /dev/null
if git -c core.fileMode=false diff-index --quiet HEAD -- ; then
    ML_FILEFLAGS=0
else
    ML_FILEFLAGS=VS_FF_PRIVATEBUILD
fi    
echo $1 | grep '\.dll$' > /dev/null
if [ $? -eq 0 ] ; then
    ML_FILETYPE=VFT_DLL
else
    ML_FILETYPE=VFT_APP
fi
ML_FILENAME=$1
ML_NAME=`echo $1 | sed 's/\..*//'`
ML_YEAR=`date '+%Y'`
ML_ICON=$CPP_SRC_HOME/mk/ml.ico

echo -DML_USER=\'\"$ML_USER\"\' \
     -DML_VERSION=$ML_VERSION \
     -DML_VERSION_STR=\'\"$ML_VERSION_STR\"\' \
     -DML_PATCH=$ML_PATCH \
     -DML_BUILD_STR=\'\"$ML_BUILD_STR\"\' \
     -DML_FILEFLAGS=$ML_FILEFLAGS \
     -DML_FILETYPE=$ML_FILETYPE \
     -DML_FILENAME=\'\"$ML_FILENAME\"\' \
     -DML_NAME=\'\"$ML_NAME\"\' \
     -DML_YEAR=\'\"$ML_YEAR\"\' \
     -DML_ICON=\'\"$ML_ICON\"\' \
     -DRIGHT_CLICK_PRODUCT_NAME=\'\"Elastic X-Pack\"\' \
     -N

