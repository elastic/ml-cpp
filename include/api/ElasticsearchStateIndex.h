/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_ElasticsearchStateIndex_h
#define INCLUDED_ml_api_ElasticsearchStateIndex_h

#include <string>

namespace ml {
namespace api {
//! Elasticsearch index for state
static const std::string ML_STATE_INDEX(".ml-state");
static const std::string MODEL_STATE_TYPE("model_state");
static const std::string REGRESSION_TRAIN_STATE_TYPE("predictive_model_train_state");
}
}

#endif // INCLUDED_ml_api_ElasticsearchStateIndex_h
