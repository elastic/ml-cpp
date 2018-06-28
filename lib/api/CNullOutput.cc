/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CNullOutput.h>

namespace ml {
namespace api {

bool CNullOutput::fieldNames(const TStrVec& /*fieldNames*/, const TStrVec& /*extraFieldNames*/) {
    return true;
}

const COutputHandler::TStrVec& CNullOutput::fieldNames() const {
    return EMPTY_FIELD_NAMES;
}

bool CNullOutput::writeRow(const TStrStrUMap& /*dataRowFields*/,
                           const TStrStrUMap& /*overrideDataRowFields*/) {
    return true;
}
}
}
