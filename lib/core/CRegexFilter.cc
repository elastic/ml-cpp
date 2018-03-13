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
#include <core/CRegexFilter.h>

#include <core/CLogger.h>

namespace ml {
namespace core {

CRegexFilter::CRegexFilter(void) : m_Regex() {}

bool CRegexFilter::configure(const TStrVec &regularExpressions) {
    m_Regex.clear();
    m_Regex.resize(regularExpressions.size());
    for (std::size_t i = 0; i < regularExpressions.size(); ++i) {
        if (m_Regex[i].init(regularExpressions[i]) == false) {
            m_Regex.clear();
            LOG_ERROR("Configuration failed; no filtering will apply");
            return false;
        }
    }

    return true;
}

std::string CRegexFilter::apply(const std::string &target) const {
    if (m_Regex.empty()) {
        return target;
    }

    std::string result(target);
    std::size_t position = 0;
    std::size_t length = 0;
    for (std::size_t i = 0; i < m_Regex.size(); ++i) {
        const CRegex &currentRegex = m_Regex[i];
        while (currentRegex.search(result, position, length)) {
            result.erase(position, length);
        }
    }
    return result;
}

bool CRegexFilter::empty(void) const { return m_Regex.empty(); }
}
}
