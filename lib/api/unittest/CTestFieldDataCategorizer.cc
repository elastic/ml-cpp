/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CTestFieldDataCategorizer.h"

CTestFieldDataCategorizer::CTestFieldDataCategorizer(const std::string& jobId,
                                                     const ml::api::CFieldConfig& config,
                                                     ml::model::CLimits& limits,
                                                     ml::api::COutputHandler& outputHandler,
                                                     ml::api::CJsonOutputWriter& jsonOutputWriter,
                                                     ml::api::CPersistenceManager* persistenceManager)
    : ml::api::CFieldDataCategorizer(jobId, config, limits, outputHandler, jsonOutputWriter, persistenceManager) {
}
