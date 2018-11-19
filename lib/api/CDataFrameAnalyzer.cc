/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalyzer.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

namespace ml {
namespace api {

bool CDataFrameAnalyzer::handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues) {
    LOG_INFO(<< core::CContainerPrinter::print(fieldNames));
    LOG_INFO(<< core::CContainerPrinter::print(fieldValues));
    return true;
}
}
}
