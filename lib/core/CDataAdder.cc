/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CDataAdder.h>

#include <limits>
#include <sstream>

namespace ml {
namespace core {

CDataAdder::~CDataAdder() {
    // Most compilers put the vtable in the object file containing the
    // definition of the first non-inlined virtual function, so DON'T move this
    // empty definition to the header file!
}

std::size_t CDataAdder::maxDocumentsPerBatchSave() const {
    return std::numeric_limits<std::size_t>::max();
}

std::size_t CDataAdder::maxDocumentSize() const {
    return std::numeric_limits<std::size_t>::max();
}

std::string CDataAdder::makeCurrentDocId(const std::string& baseId, size_t currentDocNum) {
    std::ostringstream strm;
    if (!baseId.empty()) {
        strm << baseId << '#';
    }
    strm << currentDocNum;
    return strm.str();
}
}
}
