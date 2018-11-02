/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataFrameAnalyzer_h
#define INCLUDED_ml_api_CDataFrameAnalyzer_h

#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <string>

namespace ml {
namespace api {

class API_EXPORT CDataFrameAnalyzer {

public:
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;

public:
    bool handleRecord(const TStrStrUMap& dataRowFields);
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalyzer_h
