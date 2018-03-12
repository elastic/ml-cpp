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

#include <core/CPatternSet.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <algorithm>

namespace ml {
namespace core {

namespace {
const char WILDCARD = '*';
}

CPatternSet::CPatternSet(void)
    : m_FullMatchPatterns(),
      m_PrefixPatterns(),
      m_SuffixPatterns(),
      m_ContainsPatterns() {
}

bool CPatternSet::initFromJson(const std::string &json) {
    TStrVec fullPatterns;
    TStrVec prefixPatterns;
    TStrVec suffixPatterns;
    TStrVec containsPatterns;

    rapidjson::Document doc;
    if (doc.Parse<0>(json.c_str()).HasParseError()) {
        LOG_ERROR("An error occurred while parsing pattern set from JSON: "
                  + std::string(rapidjson::GetParseError_En(doc.GetParseError())));
        return false;
    }

    if (!doc.IsArray()) {
        LOG_ERROR("Could not parse pattern set from non-array JSON object: " << json);
        return false;
    }


    for (unsigned int i = 0; i < doc.Size(); ++i) {
        if (!doc[i].IsString()) {
            LOG_ERROR("Could not parse pattern set: unexpected non-string item in JSON: " << json);
            this->clear();
            return false;
        }
        std::string pattern = doc[i].GetString();
        std::size_t length = pattern.length();
        if (length == 0) {
            continue;
        }
        if (pattern[0] == WILDCARD) {
            if (length > 2 && pattern[length - 1]  == WILDCARD) {
                std::string middle = pattern.substr(1, length - 2);
                containsPatterns.push_back(middle);
            } else if (length > 1) {
                std::string suffix = pattern.substr(1);
                suffixPatterns.push_back(std::string(suffix.rbegin(), suffix.rend()));
            }
        } else if (length > 1 && pattern[length - 1] == WILDCARD) {
            prefixPatterns.push_back(pattern.substr(0, length - 1));
        } else {
            fullPatterns.push_back(pattern);
        }
    }

    this->sortAndPruneDuplicates(fullPatterns);
    this->sortAndPruneDuplicates(prefixPatterns);
    this->sortAndPruneDuplicates(suffixPatterns);
    this->sortAndPruneDuplicates(containsPatterns);
    return m_FullMatchPatterns.build(fullPatterns)
           && m_PrefixPatterns.build(prefixPatterns)
           && m_SuffixPatterns.build(suffixPatterns)
           && m_ContainsPatterns.build(containsPatterns);
}

void CPatternSet::sortAndPruneDuplicates(TStrVec &keys) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

bool CPatternSet::contains(const std::string &key) const {
    if (m_PrefixPatterns.matchesStart(key)) {
        return true;
    }
    if (m_SuffixPatterns.matchesStart(key.rbegin(), key.rend())) {
        return true;
    }
    if (m_FullMatchPatterns.matchesFully(key)) {
        return true;
    }
    for (TStrCItr keyItr = key.begin(); keyItr != key.end(); ++keyItr) {
        if (m_ContainsPatterns.matchesStart(keyItr, key.end())) {
            return true;
        }
    }
    return false;
}

void CPatternSet::clear(void) {
    m_FullMatchPatterns.clear();
    m_PrefixPatterns.clear();
    m_SuffixPatterns.clear();
    m_ContainsPatterns.clear();
}

}
}
