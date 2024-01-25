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

#ifndef ML_CSTRINGBUFWRITER_H
#define ML_CSTRINGBUFWRITER_H

#include <core/CBoostJsonLineWriter.h>

#include <boost/json.hpp>
#include <boost/json/kind.hpp>

namespace ml {
namespace core {
using TBoostJsonLineWriterStringBuf = core::CBoostJsonLineWriter<std::string>;

class CStringBufWriter : public TBoostJsonLineWriterStringBuf {
public:
    CStringBufWriter() : TBoostJsonLineWriterStringBuf() {}
    CStringBufWriter(std::string& os) : TBoostJsonLineWriterStringBuf(os) {}

    void append(const std::string_view& str) override { (*m_Os).append(str); }

    void put(char c) override { (*m_Os).push_back(c); }

private:
};
}
}

#endif //ML_CSTRINGBUFWRITER_H
