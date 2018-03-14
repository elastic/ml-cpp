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
#include <api/CTokenListReverseSearchCreatorIntf.h>


namespace ml {
namespace api {


CTokenListReverseSearchCreatorIntf::CTokenListReverseSearchCreatorIntf(const std::string &fieldName)
    : m_FieldName(fieldName) {
}

CTokenListReverseSearchCreatorIntf::~CTokenListReverseSearchCreatorIntf(void) {
}

void CTokenListReverseSearchCreatorIntf::closeStandardSearch(std::string & /*part1*/,
                                                             std::string & /*part2*/) const {
    // Default is to do nothing
}

const std::string &CTokenListReverseSearchCreatorIntf::fieldName(void) const {
    return m_FieldName;
}


}
}

