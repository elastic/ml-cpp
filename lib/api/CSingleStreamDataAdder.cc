/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CSingleStreamDataAdder.h>

#include <core/CLogger.h>

#include <ostream>

namespace ml {
namespace api {

const size_t CSingleStreamDataAdder::MAX_DOCUMENT_SIZE(16 * 1024 * 1024); // 16MB

CSingleStreamDataAdder::CSingleStreamDataAdder(const TOStreamP& stream) : m_Stream(stream) {
}

CSingleStreamDataAdder::TOStreamP CSingleStreamDataAdder::addStreamed(const std::string& /*index*/, const std::string& id) {
    if (m_Stream != nullptr && !m_Stream->bad()) {
        // Start with metadata, leaving the index for the receiving code to set
        (*m_Stream) << "{\"index\":{\"_id\":\"" << id << "\"}}\n";
    }

    return m_Stream;
}

bool CSingleStreamDataAdder::streamComplete(TOStreamP& stream, bool force) {
    if (stream != m_Stream) {
        LOG_ERROR("Attempt to use the single stream data adder with multiple streams");
        return false;
    }

    if (stream != nullptr && !stream->bad()) {
        // Each Elasticsearch document must be followed by a newline
        stream->put('\n');

        // The Ml format is to separate documents with zero bytes
        stream->put('\0');

        // If force flush to ensure all data is pushed through the remote end
        if (force) {
            stream->flush();
        }
    }

    return stream != nullptr && !stream->bad();
}

std::size_t CSingleStreamDataAdder::maxDocumentSize() const {
    return MAX_DOCUMENT_SIZE;
}
}
}
