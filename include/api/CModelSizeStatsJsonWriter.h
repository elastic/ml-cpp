/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
    static void
    write(const std::string& jobId, const model::CResourceMonitor::SResults& results, core::CRapidJsonConcurrentLineWriter& writer);
};
}
}

#endif // INCLUDED_ml_api_CModelSizeStatsJsonWriter_h
