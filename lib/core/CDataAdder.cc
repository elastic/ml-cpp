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
