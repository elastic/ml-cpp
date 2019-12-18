/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameOutliersState.h>

namespace ml {
namespace api {

counter_t::ECounterTypes CDataFrameOutliersState::memoryCounterType() {
    return counter_t::E_DFOPeakMemoryUsage;
}
}
}
