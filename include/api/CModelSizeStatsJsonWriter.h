/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CModelSizeStatsJsonWriter_h
#define INCLUDED_ml_api_CModelSizeStatsJsonWriter_h

#include <core/CNonInstantiatable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <model/CResourceMonitor.h>

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! \brief
//! A static utility for writing the model_size_stats document in JSON.
class API_EXPORT CModelSizeStatsJsonWriter : private core::CNonInstantiatable {
public:
    //! Writes the model size stats in the \p results in JSON format.
    static void write(const std::string& jobId,
                      const model::CResourceMonitor::SModelSizeStats& results,
                      core::CRapidJsonConcurrentLineWriter& writer);
};
}
}

#endif // INCLUDED_ml_api_CModelSizeStatsJsonWriter_h
