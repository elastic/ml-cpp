/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisOutliersState_h
#define INCLUDED_ml_api_CDataFrameAnalysisOutliersState_h

#include <api/CDataFrameAnalysisState.h>

namespace ml {
namespace api {
// TODO rename to CDataFrameOutliersState
class CDataFrameAnalysisOutliersState : public CDataFrameAnalysisState {
protected:
    counter_t::ECounterTypes memoryCounterType() override;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisOutliersState_h
