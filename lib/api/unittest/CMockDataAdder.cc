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
#include "CMockDataAdder.h"

#include <core/CLogger.h>

CMockDataAdder::CMockDataAdder() {
}

CMockDataAdder::TOStreamP CMockDataAdder::addStreamed(const std::string& /*id*/) {
    if (m_Stream == nullptr) {
        m_Stream.reset(new std::ostringstream{});
    }
    return m_Stream;
}

bool CMockDataAdder::streamComplete(TOStreamP& strm, bool /*force*/) {
    if (strm == nullptr || m_Stream != strm) {
        return false;
    }
    const std::string& result = dynamic_cast<std::ostringstream&>(*m_Stream).str();
    LOG_TRACE(<< "Stream complete - adding data: " << result);
    m_Events.push_back('[' + result + ']');
    m_Stream.reset();
    return true;
}

const CMockDataAdder::TStrVec& CMockDataAdder::events() const {
    return m_Events;
}

void CMockDataAdder::clear() {
    m_Events.clear();
    m_Stream.reset();
}
