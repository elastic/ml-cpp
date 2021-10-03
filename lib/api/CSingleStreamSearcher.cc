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
#include <api/CSingleStreamSearcher.h>

#include <istream>

namespace ml {
namespace api {

CSingleStreamSearcher::CSingleStreamSearcher(const TIStreamP& stream)
    : m_Stream(stream) {
}

CSingleStreamSearcher::~CSingleStreamSearcher() {
    // We have to ensure we consume the stream to its end
    // as it is possible that we receive additional
    // documents after the eos marker and we should not
    // block the state streamer thread. We may receive
    // additional documents in case the state reduces
    // in size and spreads over less docs than prior state.
    this->consumeStream();
}

void CSingleStreamSearcher::consumeStream() {
    char buf[512];
    while (m_Stream->good()) {
        m_Stream->read(buf, 512);
    }
}

CSingleStreamSearcher::TIStreamP
CSingleStreamSearcher::search(size_t /*currentDocNum*/, size_t /*limit*/) {
    // documents in a stream are separated by '\0', skip over it in case to not confuse clients (see #279)
    if (m_Stream->peek() == 0) {
        m_Stream->get();
    }

    return m_Stream;
}
}
}
