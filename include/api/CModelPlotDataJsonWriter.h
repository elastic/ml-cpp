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
#ifndef INCLUDED_ml_api_CModelPlotDataJsonWriter_h
#define INCLUDED_ml_api_CModelPlotDataJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <model/CModelPlotData.h>

#include <rapidjson/document.h>

#include <iosfwd>
#include <sstream>
#include <string>

#include <stdint.h>

namespace ml {
namespace api {

//! \brief
//! Write visualisation data as a JSON document
//!
//! DESCRIPTION:\n
//! JSON is either written to the output stream or an internal
//! string stream. The various writeXXX functions convert the
//! arguments to a JSON doc and write to the stream then flush
//! the stream. If the object is constructed without an
//! outputstream then the doc can be read via the internalString
//! function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The stream is flushed after at the end of each of the public
//! write.... functions.
//!
class API_EXPORT CModelPlotDataJsonWriter final : private core::CNonCopyable {
private:
    static const std::string JOB_ID;
    static const std::string MODEL_PLOT;
    static const std::string DETECTOR_INDEX;
    static const std::string PARTITION_FIELD_NAME;
    static const std::string PARTITION_FIELD_VALUE;
    static const std::string TIME;
    static const std::string FEATURE;
    static const std::string BY;
    static const std::string BY_FIELD_NAME;
    static const std::string BY_FIELD_VALUE;
    static const std::string OVER_FIELD_NAME;
    static const std::string OVER_FIELD_VALUE;
    static const std::string LOWER;
    static const std::string UPPER;
    static const std::string MEDIAN;
    static const std::string ACTUAL;
    static const std::string BUCKET_SPAN;

public:
    using TStrDoublePrVec = model::CModelPlotData::TStrDoublePrVec;
    using TByFieldData = model::CModelPlotData::SByFieldData;
    using TStrByFieldDataUMap = model::CModelPlotData::TStrByFieldDataUMap;
    using TStrByFieldDataUMapCItr = TStrByFieldDataUMap::const_iterator;
    using TFeatureStrByFieldDataUMapUMapCItr = model::CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr;
    using TStrDoublePr = model::CModelPlotData::TStrDoublePr;

public:
    //! Constructor that causes to be written to the specified stream
    explicit CModelPlotDataJsonWriter(core::CJsonOutputStreamWrapper& outStream);

    void writeFlat(const std::string& jobId, const model::CModelPlotData& data);

private:
    void writeFlatRow(core_t::TTime time,
                      const std::string& jobId,
                      int detectorIndex,
                      const std::string& partitionFieldName,
                      const std::string& partitionFieldValue,
                      const std::string& feature,
                      const std::string& byFieldName,
                      const std::string& byFieldValue,
                      const TByFieldData& byData,
                      core_t::TTime bucketSpan,
                      rapidjson::Value& doc);

private:
    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_api_CModelPlotDataJsonWriter_h
