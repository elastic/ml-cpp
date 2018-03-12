/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#include <api/CSingleStreamSearcher.h>

#include <istream>

namespace ml {
namespace api {

CSingleStreamSearcher::CSingleStreamSearcher(const TIStreamP &stream)
    : m_Stream(stream) {
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

