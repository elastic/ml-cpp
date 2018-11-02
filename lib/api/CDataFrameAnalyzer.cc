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

bool CDataFrameAnalyzer::handleRecord(const TStrStrUMap& dataRowFields) {
    LOG_INFO(<< core::CContainerPrinter::print(dataRowFields));
    return true;
}
}
}
