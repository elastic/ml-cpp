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
const std::string STATE_ID_SUFFIX{"_state"};

std::string getStateId(const std::string& jobId, const std::string& analysisName) {
    return jobId + '_' + analysisName + STATE_ID_SUFFIX;
}
}
}
