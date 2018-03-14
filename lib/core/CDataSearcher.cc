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
#include <core/CDataSearcher.h>


namespace ml {
namespace core {

const std::string CDataSearcher::EMPTY_STRING;


CDataSearcher::CDataSearcher(void)
    : m_SearchTerms(2) {
}

CDataSearcher::~CDataSearcher(void) {
}

void CDataSearcher::setStateRestoreSearch(const std::string &index) {
    m_SearchTerms[0] = index;
    m_SearchTerms[1].clear();
}

void CDataSearcher::setStateRestoreSearch(const std::string &index,
                                          const std::string &id) {
    m_SearchTerms[0] = index;
    m_SearchTerms[1] = id;
}


}
}
