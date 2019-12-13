/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameTrainBoostedTreeState.h>

namespace ml {
namespace api {

counter_t::ECounterTypes CDataFrameTrainBoostedTreeState::memoryCounterType() {
    return counter_t::E_DFTPMPeakMemoryUsage;
}
}
}
