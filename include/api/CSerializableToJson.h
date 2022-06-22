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
#ifndef INCLUDED_ml_api_CSerializableToJson_h
#define INCLUDED_ml_api_CSerializableToJson_h

#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/Constants.h>

#include <api/ImportExport.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/rapidjson.h>

#include <sstream>
#include <stdexcept>

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

    //! Write the type as JSON using \p writer.
    virtual void addToJsonStream(TGenericLineWriter& writer) const = 0;

    //! Get as string.
    std::string jsonString() const;
};

//! \brief Interface for writing the inference model and training data summarization
//! JSON descriptions to a stream which first compresses, base64 encodes and chunks.
//!
//! IMPLEMENTATION:\n
//! This splits large state objects across multiple documents which can be reassembled
//! by CSerializableFromCompressedChunkedJson. Typically this writes documents of the
//! form
//! \code
//! "<doc_tag>": {
//!   "doc_num": <number>,
//!   "<payload_tag>": "compressed blob",
//!   ["eos": True]
//! }
//! \endcode
//! where derived types supply `doc_tag` and `payload_tag`. This should be used in
//! conjunction with `CSerializableFromCompressedChunkedJson` to read compressed
//! chunked documents.
class API_EXPORT CSerializableToCompressedChunkedJson : public CSerializableToJsonStream {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    static constexpr std::size_t MAX_DOCUMENT_SIZE{16 * core::constants::BYTES_IN_MEGABYTES};

public:
    explicit CSerializableToCompressedChunkedJson(std::size_t maxDocumentSize = MAX_DOCUMENT_SIZE);

    //! Write the JSON compressed and encoded using \p writer.
    //!
    //! This chunks the compressed encoded JSON into documents no larger than the
    //! specified maximum document size and writes them as individual JSON documents.
    virtual void addCompressedToJsonStream(TRapidJsonWriter& writer) const = 0;

    //! \name Test Only
    //@{
    //! \return A stream to the raw compressed encoded state (no chunking).
    std::stringstream jsonCompressedStream() const;
    //@}

protected:
    void addCompressedToJsonStream(const std::string& compressedDocTag,
                                   const std::string& payloadTag,
                                   TRapidJsonWriter& writer) const;
    auto callableAddToJsonStream() const {
        return
            [this](TGenericLineWriter& writer) { this->addToJsonStream(writer); };
    }

private:
    std::size_t m_MaxDocumentSize;
};

//! \brief Utility for reading the inference model and training data summarization
//! JSON descriptions from a stream which dechunks, decodes and decompresses.
//!
//! DESCRIPTION:\n
//! This provides a utility method to dechunk the state and return a stream over the
//! decoded and decompressed JSON. Derived classes are expected to use this to parse
//! and return the objects from their JSON representation.
class API_EXPORT CSerializableFromCompressedChunkedJson {
protected:
    using TIStreamPtr = std::shared_ptr<std::istream>;

    //! Read chunked compressed data from \p inputStream merge chunks decompress and return
    //! a stream for the raw JSON.
    //!
    //! \param[in] compressedDocTag The tag for the compressed state chunk object.
    //! \param[in] payloadTag The tag for the compressed state chunk member.
    //! \param[in] inputStream The stream to read for the compressed chunked docs.
    //! \param[in] buffer The buffer to hold the dechunked state. This needs to stay alive
    //! while the returned stream is in use.
    //! \warning Returns a null pointer on failure to read the state.
    //! \note Expects state in a format written by a CSerializableToCompressedChunkedJson
    //! object using tags \p compressedDocTag and \p payloadTag.
    static TIStreamPtr rawJsonStream(const std::string& compressedDocTag,
                                     const std::string& payloadTag,
                                     TIStreamPtr inputStream,
                                     std::iostream& buffer);

    static void assertNoParseError(const rapidjson::Document& doc) {
        if (doc.HasParseError()) {
            const char* error{rapidjson::GetParseError_En(doc.GetParseError())};
            throw std::runtime_error{"Error parsing JSON at offset " +
                                     std::to_string(doc.GetErrorOffset()) + ": " +
                                     ((error != nullptr) ? error : "No message")};
        }
    }

    template<typename GET, typename VALUE>
    static auto ifExists(const std::string& tag, const GET& get, const VALUE& value)
        -> decltype(get(value[tag])) {
        if (value.HasMember(tag)) {
            try {
                return get(value[tag]);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error{"Field '" + tag + "' " + e.what() + "."};
            }
        }
        throw std::runtime_error{"Field '" + tag + "' is missing."};
    }

    static auto getAsObjectFrom(const rapidjson::Value& value) {
        if (value.IsObject()) {
            return value.GetObject();
        }
        throw std::runtime_error{"is not an object"};
    }

    static auto getAsArrayFrom(const rapidjson::Value& value) {
        if (value.IsArray()) {
            return value.GetArray();
        }
        throw std::runtime_error{"is not an array"};
    }

    static bool getAsBoolFrom(const rapidjson::Value& value) {
        if (value.IsBool()) {
            return value.GetBool();
        }
        throw std::runtime_error{"is not a bool"};
    }

    static std::uint64_t getAsUint64From(const rapidjson::Value& value) {
        if (value.IsUint64()) {
            return value.GetUint64();
        }
        throw std::runtime_error{"is not a uint64"};
    }

    static double getAsDoubleFrom(const rapidjson::Value& value) {
        if (value.IsDouble()) {
            return value.GetDouble();
        }
        throw std::runtime_error{"is not a double"};
    }

    static auto getAsStringFrom(const rapidjson::Value& value) {
        if (value.IsString()) {
            return value.GetString();
        }
        throw std::runtime_error{"is not a string"};
    }

    static std::size_t getStringLengthFrom(const rapidjson::Value& value) {
        if (value.IsString()) {
            return value.GetStringLength();
        }
        throw std::runtime_error{"is not a string"};
    }
};
}
}

#endif //INCLUDED_ml_api_CSerializableToJson_h
