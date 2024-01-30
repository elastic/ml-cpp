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

#include <core/CBoostJsonConcurrentLineWriter.h>
#include <core/Constants.h>

#include <api/ImportExport.h>

#include "core/CStreamWriter.h"
#include <sstream>
#include <stdexcept>

namespace ml {
namespace api {
//! \brief Interface for adding all elements of an inference model definition
//! to a JSON writer.
class API_EXPORT CSerializableToJsonDocument {
public:
    using TBoostJsonWriter = core::CBoostJsonConcurrentLineWriter;

public:
    virtual ~CSerializableToJsonDocument() = default;
    //! Serialize the object as JSON items under the \p parentObject using the
    //! specified \p writer.
    virtual void addToJsonDocument(json::object& parentObject,
                                   TBoostJsonWriter& writer) const = 0;
};

//! \brief Interface for writing the inference model defition JSON description
//! to a stream.
class API_EXPORT CSerializableToJsonStream {
public:
    //    using TGenericLineWriter = core::CBoostJsonLineWriter<std::ostream>;
    using TGenericLineWriter = core::CStreamWriter;

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
//! IMPLEMENTATION DECISIONS:\n
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
    using TBoostJsonWriter = core::CBoostJsonConcurrentLineWriter;

public:
    static constexpr std::size_t MAX_DOCUMENT_SIZE{16 * core::constants::BYTES_IN_MEGABYTES};

public:
    explicit CSerializableToCompressedChunkedJson(std::size_t maxDocumentSize = MAX_DOCUMENT_SIZE);

    //! Write the JSON compressed and encoded using \p writer.
    //!
    //! This chunks the compressed encoded JSON into documents no larger than the
    //! specified maximum document size and writes them as individual JSON documents.
    virtual void addCompressedToJsonStream(TBoostJsonWriter& writer) const = 0;

    //! \name Test Only
    //@{
    //! \return A stream to the raw compressed encoded state (no chunking).
    std::stringstream jsonCompressedStream() const;
    //@}

protected:
    void addCompressedToJsonStream(const std::string& compressedDocTag,
                                   const std::string& payloadTag,
                                   TBoostJsonWriter& writer) const;
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

    static void assertNoParseError(const json::error_code& ec) {
        if (ec) {
            throw std::runtime_error{"Error parsing JSON: " + ec.message()};
        }
    }

    static void assertIsJsonObject(const json::value& val) {
        if (val.is_object() == false) {
            throw std::runtime_error{"Error. JSON value is not object"};
        }
    }

    template<typename GET, typename VALUE>
    static auto ifExists(const std::string& tag, const GET& get, const VALUE& value)
        -> decltype(get(value.at(tag))) {
        if (value.contains(tag)) {
            try {
                return get(value.at(tag));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error{"Field '" + tag + "' " + e.what() + "."};
            }
        }
        throw std::runtime_error{"Field '" + tag + "' is missing."};
    }

    static auto getAsObjectFrom(const json::value& value) {
        if (value.is_object()) {
            return value.as_object();
        }
        throw std::runtime_error{"is not an object"};
    }

    static auto getAsArrayFrom(const json::value& value) {
        if (value.is_array()) {
            return value.as_array();
        }
        throw std::runtime_error{"is not an array"};
    }

    static bool getAsBoolFrom(const json::value& value) {
        if (value.is_bool()) {
            return value.as_bool();
        }
        throw std::runtime_error{"is not a bool"};
    }

    static std::int64_t getAsInt64From(const json::value& value) {
        json::error_code ec;
        std::int64_t ret = value.to_number<std::int64_t>(ec);
        if (ec) {
            throw std::runtime_error{"is not a int64"};
        }
        return ret;
    }

    static std::uint64_t getAsUint64From(const json::value& value) {
        json::error_code ec;
        std::uint64_t ret = value.to_number<std::uint64_t>(ec);
        if (ec) {
            throw std::runtime_error{"is not a uint64"};
        }
        return ret;
    }

    static double getAsDoubleFrom(const json::value& value) {
        if (value.is_double()) {
            return value.as_double();
        }
        throw std::runtime_error{"is not a double"};
    }

    static auto getAsStringFrom(const json::value& value) {
        if (value.is_string()) {
            return value.as_string().c_str();
        }
        throw std::runtime_error{"is not a string"};
    }

    static std::size_t getStringLengthFrom(const json::value& value) {
        if (value.is_string()) {
            return value.as_string().size();
        }
        throw std::runtime_error{"is not a string"};
    }
};
}
}

#endif //INCLUDED_ml_api_CSerializableToJson_h
