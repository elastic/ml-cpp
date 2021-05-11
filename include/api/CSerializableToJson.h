/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CSerializableToJson_h
#define INCLUDED_ml_api_CSerializableToJson_h

#include <core/CRapidJsonConcurrentLineWriter.h>

#include <api/ImportExport.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

namespace ml {
namespace api {
//! \brief Interface for adding all elements of an inference model definition
//! to a JSON writer.
class API_EXPORT CSerializableToJsonDocument {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    virtual ~CSerializableToJsonDocument() = default;
    //! Serialize the object as JSON items under the \p parentObject using the
    //! specified \p writer.
    virtual void addToJsonDocument(rapidjson::Value& parentObject,
                                   TRapidJsonWriter& writer) const = 0;
};

//! \brief Interface for writing the inference model defition JSON description
//! to a stream.
class API_EXPORT CSerializableToJsonStream {
public:
    using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

public:
    virtual ~CSerializableToJsonStream() = default;
    virtual void addToJsonStream(TGenericLineWriter& /*writer*/) const = 0;
};

//! \brief Interface for writing the inference model definition JSON description
//! to a stream which first compresses, base64 encodes and chunks.
class API_EXPORT CSerializableToJsonDocumentCompressed : public CSerializableToJsonStream {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    virtual void addToDocumentCompressed(TRapidJsonWriter& writer) const = 0;
    virtual void addToDocumentCompressed(TRapidJsonWriter& writer,
                                         const std::string& compressedDocTag,
                                         const std::string& payloadTag) const;
    virtual std::stringstream jsonCompressedStream() const;
    virtual void jsonStream(std::ostream& jsonStrm) const;
};
}
}

#endif //INCLUDED_ml_api_CSerializableToJson_h
