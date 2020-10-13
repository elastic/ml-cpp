/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CResponseJsonWriter.h"

#include <core/CLogger.h>

#include <ios>

namespace ml {
namespace controller {
namespace {

// JSON field names
const std::string ID{"id"};
const std::string SUCCESS{"success"};
const std::string REASON{"reason"};
}

CResponseJsonWriter::CResponseJsonWriter(std::ostream& responseStream)
    : m_WrappedOutputStream(responseStream), m_Writer{m_WrappedOutputStream} {
}

void CResponseJsonWriter::writeResponse(std::uint32_t id, bool success, const std::string& reason) {
    m_Writer.StartObject();
    m_Writer.Key(ID);
    m_Writer.Uint(id);
    m_Writer.Key(SUCCESS);
    m_Writer.Bool(success);
    m_Writer.Key(REASON);
    m_Writer.String(reason);
    m_Writer.EndObject();
    m_Writer.flush();
    LOG_DEBUG(<< "Wrote controller response - id: " << id
              << " success: " << std::boolalpha << success << " reason: " << reason);
}
}
}
