/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_api_ElasticsearchStateIndex_h
#define INCLUDED_ml_api_ElasticsearchStateIndex_h

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {
API_EXPORT std::string getStateId(const std::string& jobId, const std::string& analysisName);
}
}

#endif // INCLUDED_ml_api_ElasticsearchStateIndex_h
