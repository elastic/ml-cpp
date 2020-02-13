/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_ElasticsearchStateIndex_h
#define INCLUDED_ml_api_ElasticsearchStateIndex_h

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {
//! Elasticsearch index for state
extern API_EXPORT const std::string ML_STATE_INDEX;
extern API_EXPORT const std::string MODEL_STATE_TYPE;
extern API_EXPORT const std::string STATE_ID_SUFFIX;

API_EXPORT std::string getStateId(const std::string& jobId, const std::string& analysisName);
}
}

#endif // INCLUDED_ml_api_ElasticsearchStateIndex_h
