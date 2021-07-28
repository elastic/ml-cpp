/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_api_CModelSizeStatsJsonWriter_h
#define INCLUDED_ml_api_CModelSizeStatsJsonWriter_h

#include <core/CRapidJsonConcurrentLineWriter.h>

#include <model/CResourceMonitor.h>

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace model {
struct SCategorizerStats;
}
namespace api {

//! \brief
//! A static utility for writing the model_size_stats document in JSON.
class API_EXPORT CModelSizeStatsJsonWriter {
public:
    using TOptionalTime = boost::optional<core_t::TTime>;

public:
    //! Disallow instantiation.
    CModelSizeStatsJsonWriter() = delete;
    CModelSizeStatsJsonWriter(const CModelSizeStatsJsonWriter&) = delete;

    //! Writes the model size stats in the \p results in JSON format.
    static void write(const std::string& jobId,
                      const model::CResourceMonitor::SModelSizeStats& results,
                      core::CRapidJsonConcurrentLineWriter& writer);

    //! Writes the categorizer stats in JSON format.
    static void writeCategorizerStats(const std::string& jobId,
                                      const std::string& partitionFieldName,
                                      const std::string& partitionFieldValue,
                                      const model::SCategorizerStats& categorizerStats,
                                      const TOptionalTime& timestamp,
                                      core::CRapidJsonConcurrentLineWriter& writer);

private:
    //! Writes fields common to both model size stats and categorizer stats in
    //! JSON format.
    static void writeCommonFields(const std::string& jobId,
                                  const model::SCategorizerStats& categorizerStats,
                                  const TOptionalTime& timestamp,
                                  core::CRapidJsonConcurrentLineWriter& writer);
};
}
}

#endif // INCLUDED_ml_api_CModelSizeStatsJsonWriter_h
