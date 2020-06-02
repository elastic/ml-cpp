/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CAnnotationJsonWriter_h
#define INCLUDED_ml_api_CAnnotationJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <model/CAnnotation.h>

#include <rapidjson/document.h>

#include <iosfwd>
#include <sstream>
#include <string>

#include <stdint.h>

namespace ml {
namespace api {

//! \brief
//! Write annotation result as a JSON document
class API_EXPORT CAnnotationJsonWriter final : private core::CNonCopyable {
public:
    //! Constructor that causes to be written to the specified stream
    explicit CAnnotationJsonWriter(core::CJsonOutputStreamWrapper& outStream);

    void writeResult(const std::string& jobId, const model::CAnnotation& annotation);

private:
    void populateAnnotationObject(const std::string& jobId,
                                  const model::CAnnotation& annotation,
                                  rapidjson::Value& doc);

private:
    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_api_CAnnotationJsonWriter_h
