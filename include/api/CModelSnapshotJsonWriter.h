/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CModelSnapshotJsonWriter_h
#define INCLUDED_ml_api_CModelSnapshotJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <model/CResourceMonitor.h>

#include <api/ImportExport.h>

#include <string>

namespace ml
{
namespace api
{

//! \brief
//! Write model snapshots in JSON format
//!
//! DESCRIPTION:\n
//! Outputs the model snapshot documents that accompany each state persist.
//!
class API_EXPORT CModelSnapshotJsonWriter
{
    public:
        //! Structure to store the model snapshot metadata
        struct SModelSnapshotReport
        {
            std::string                       s_MinVersion;
            core_t::TTime                     s_SnapshotTimestamp;
            std::string                       s_Description;
            std::string                       s_SnapshotId;
            size_t                            s_NumDocs;
            model::CResourceMonitor::SResults s_ModelSizeStats;
            std::string                       s_NormalizerState;
            core_t::TTime                     s_LatestRecordTime;
            core_t::TTime                     s_LatestFinalResultTime;
        };

    public:
        //! Constructor that causes output to be written to the specified wrapped stream
        CModelSnapshotJsonWriter(const std::string &jobId,
                                 core::CJsonOutputStreamWrapper &strmOut);

        //! Writes the given model snapshot in JSON format.
        void write(const SModelSnapshotReport &report);

        //! Write the quantile's state
        static void writeQuantileState(const std::string &jobId,
                                       const std::string &state,
                                       core_t::TTime timestamp,
                                       core::CRapidJsonConcurrentLineWriter &writer);

    private:
        //! The job ID
        std::string                          m_JobId;

        //! JSON line writer
        core::CRapidJsonConcurrentLineWriter m_Writer;
};


}
}

#endif // INCLUDED_ml_api_CModelSnapshotJsonWriter_h
