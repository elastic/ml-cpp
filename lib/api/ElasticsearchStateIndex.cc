/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/ElasticsearchStateIndex.h>

namespace ml {
namespace api {

const std::string ML_STATE_INDEX{".ml-state"};
const std::string MODEL_STATE_TYPE{"model_state"};
const std::string REGRESSION_TRAIN_STATE_TYPE{"regression_state"};

std::string getRegressionStateId(const std::string& jobId) {
    return jobId + '_' + REGRESSION_TRAIN_STATE_TYPE;
}
}
}
