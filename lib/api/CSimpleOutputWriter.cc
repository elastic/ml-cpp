/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CSimpleOutputWriter.h>

namespace ml {
namespace api {

// Initialise statics
const CSimpleOutputWriter::TStrVec CSimpleOutputWriter::EMPTY_FIELD_NAMES;
const CSimpleOutputWriter::TStrStrUMap CSimpleOutputWriter::EMPTY_FIELD_OVERRIDES;

bool CSimpleOutputWriter::fieldNames(const TStrVec& fieldNames) {
    return this->fieldNames(fieldNames, EMPTY_FIELD_NAMES);
}

bool CSimpleOutputWriter::writeRow(const TStrStrUMap& dataRowFields) {
    // Since the overrides are checked first, but we know there aren't any, it's
    // most efficient to pretend everything's an override
    return this->writeRow(EMPTY_FIELD_OVERRIDES, dataRowFields, TOptionalTime{});
}

bool CSimpleOutputWriter::writeRow(const TStrStrUMap& dataRowFields,
                                   const TStrStrUMap& overrideDataRowFields) {
    return this->writeRow(dataRowFields, overrideDataRowFields, TOptionalTime{});
}
}
}
