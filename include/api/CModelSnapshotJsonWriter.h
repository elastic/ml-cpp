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
            core_t::TTime                     s_SnapshotTimestamp;
            std::string                       s_Description;
            std::string                       s_SnapshotId;
            size_t                            s_NumDocs;
            model::CResourceMonitor::SResults s_ModelSizeStats;
            std::string                       s_NormalizerState;
            core_t::TTime                     s_LatestRecordTime;
            core_t::TTime                     s_LatestFinalResultTime;
        };

        static const std::string JOB_ID;
        static const std::string TIMESTAMP;
        static const std::string MODEL_SNAPSHOT;
        static const std::string SNAPSHOT_ID;
        static const std::string SNAPSHOT_DOC_COUNT;
        static const std::string DESCRIPTION;
        static const std::string LATEST_RECORD_TIME;
        static const std::string LATEST_RESULT_TIME;
        static const std::string QUANTILES;
        static const std::string QUANTILE_STATE;

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
