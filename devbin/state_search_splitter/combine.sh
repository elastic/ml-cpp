#!/bin/sh
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

#
# This script can be used to combine the files that state_search_splitter writes
# into a single file that can be passed to the --restore option of autodetect
# when reproducing a state restoration bug.
#
# Usage:
#
# ./combine.sh <job_id> <snapshot_id> <highest_state_doc_number>
#
# For example:
#
# ./combine.sh my-job 1587614413 35
#
# After creating your my-job.combined_state.bin file you could re-run autodetect
# something like this:
#
# ./autodetect --jobid my-job --limitconfig limits.conf --fieldconfig fields.conf --bucketspan 900 --delimiter , --timefield my-time-field --quantilesState quantiles.json --restore my-job.combined_state.bin --input dummy-new-data.csv
#
# fields.conf, limits.conf, the bucketspan and the timefield need to be derived
# from the job config.  quantiles.json needs to be obtained from another search
# on the .ml-state index and dummy-new-data.csv created such that it includes
# all the fields mentioned in fields.conf.
#
# Additionally, this patch needs to be applied to autodetect's Main.cc to use
# this state format from the command line:
#
# --- a/bin/autodetect/Main.cc
# +++ b/bin/autodetect/Main.cc
# @@ -192,7 +192,7 @@ int main(int argc, char** argv) {
#          if (ioMgr.restoreStream()) {
#              // Check whether state is restored from a file, if so we assume that this is a debugging case
#              // and therefore does not originate from the ML Java code.
# -            if (!isRestoreFileNamedPipe) {
# +            if (false) {
#                  // apply a filter to overcome differences in the way persistence vs. restore works
#                  auto strm = std::make_shared<boost::iostreams::filtering_istream>();
#                  strm->push(ml::api::CStateRestoreStreamFilter());
#

if [ $# -ne 3 ] ; then
    echo "Usage: $0 <job_id> <snapshot_id> <highest_state_doc_number>"
    exit 1
fi

JOB_ID="$1"
SNAPSHOT_ID=$2
DOC_NUM=1
HIGHEST_DOC_NUM=$3
OUTPUT_FILE="$JOB_ID.combined_state.bin"
rm -f "$OUTPUT_FILE"

while [ $DOC_NUM -le $HIGHEST_DOC_NUM ]
do
    cat "${JOB_ID}_model_state_${SNAPSHOT_ID}#${DOC_NUM}.json" >> "$OUTPUT_FILE"
    head -c 1 /dev/zero >> "$OUTPUT_FILE"
    DOC_NUM=`expr $DOC_NUM + 1`
done
