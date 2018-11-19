/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataFrameAnalyzer_h
#define INCLUDED_ml_api_CDataFrameAnalyzer_h

#include <api/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace api {

class API_EXPORT CDataFrameAnalyzer {
public:
    using TStrVec = std::vector<std::string>;

public:
    bool handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues);
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalyzer_h
