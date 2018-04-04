/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CSingleStreamSearcher.h>

#include <istream>

namespace ml {
namespace api {

CSingleStreamSearcher::CSingleStreamSearcher(const TIStreamP& stream) : m_Stream(stream) {
}

CSingleStreamSearcher::TIStreamP CSingleStreamSearcher::search(size_t /*currentDocNum*/, size_t /*limit*/) {
    // documents in a stream are separated by '\0', skip over it in case to not confuse clients (see #279)
    if (m_Stream->peek() == 0) {
        m_Stream->get();
    }

    return m_Stream;
}
}
}
