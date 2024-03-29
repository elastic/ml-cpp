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

#ifndef INCLUDED_ml_core_CStreamWriter_h
#define INCLUDED_ml_core_CStreamWriter_h

#include <core/CBoostJsonLineWriter.h>

#include <boost/json.hpp>

namespace ml {
namespace core {
using TBoostJsonLineWriterStream = core::CBoostJsonLineWriter<std::ostream>;

class CStreamWriter : public TBoostJsonLineWriterStream {
public:
    CStreamWriter(std::ostream& os) : TBoostJsonLineWriterStream(os) {}
    CStreamWriter() : TBoostJsonLineWriterStream() {}

    void append(const std::string_view& str) override { (*m_Os) << str; }

    void put(char c) override { m_Os->put(c); }
};
}
}

#endif //INCLUDED_ml_core_CStreamWriter_h
