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
#ifndef INCLUDED_ml_api_CAnnotationJsonWriter_h
#define INCLUDED_ml_api_CAnnotationJsonWriter_h

#include <core/CRapidJsonConcurrentLineWriter.h>

#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace core {
class CJsonOutputStreamWrapper;
}
namespace model {
class CAnnotation;
}
namespace api {

//! \brief
//! Write annotation result as a JSON document
class API_EXPORT CAnnotationJsonWriter final {
public:
    //! Constructor that causes to be written to the specified stream
    explicit CAnnotationJsonWriter(core::CJsonOutputStreamWrapper& outStream);

    //! No copying
    CAnnotationJsonWriter(const CAnnotationJsonWriter&) = delete;
    CAnnotationJsonWriter& operator=(const CAnnotationJsonWriter&) = delete;

    void writeResult(const std::string& jobId, const model::CAnnotation& annotation);

private:
    void populateAnnotationObject(const std::string& jobId,
                                  const model::CAnnotation& annotation,
                                  rapidjson::Value& obj);

private:
    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_api_CAnnotationJsonWriter_h
