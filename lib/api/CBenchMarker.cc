/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CBenchMarker.h>

#include <core/CLogger.h>
#include <core/CoreTypes.h>

#include <model/CDataCategorizer.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace ml {
namespace api {

CBenchMarker::CBenchMarker() : m_TotalMessages(0), m_ScoredMessages(0) {
}

bool CBenchMarker::init(const std::string& regexFilename) {
    // Reset in case of reinitialisation
    m_TotalMessages = 0;
    m_ScoredMessages = 0;
    m_Measures.clear();

    std::ifstream ifs(regexFilename.c_str());
    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }

        core::CRegex regex;
        if (regex.init(line) == false) {
            return false;
        }

        m_Measures.push_back(TRegexLocalCategoryIdSizeStrPrMapPr(
            regex, TLocalCategoryIdSizeStrPrMap()));
    }

    return (m_Measures.size() > 0);
}

void CBenchMarker::addResult(const std::string& message, model::CLocalCategoryId categoryId) {
    bool scored{false};
    std::size_t position{0};
    for (auto measureVecIter = m_Measures.begin();
         measureVecIter != m_Measures.end(); ++measureVecIter) {
        const core::CRegex& regex = measureVecIter->first;
        if (regex.search(message, position) == true) {
            TLocalCategoryIdSizeStrPrMap& counts = measureVecIter->second;
            auto mapIter = counts.find(categoryId);
            if (mapIter == counts.end()) {
                counts.insert(TLocalCategoryIdSizeStrPrMap::value_type(
                    categoryId, TSizeStrPr(1, message)));
            } else {
                ++(mapIter->second.first);
            }
            ++m_ScoredMessages;
            scored = true;
            break;
        }
    }

    ++m_TotalMessages;

    if (!scored) {
        LOG_TRACE(<< "Message not included in scoring: " << message);
    }
}

void CBenchMarker::dumpResults() const {
    // Sort the results in descending order of actual category occurrence
    using TSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPr =
        std::pair<std::size_t, TRegexLocalCategoryIdSizeStrPrMapPrVecCItr>;
    using TSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPrVec =
        std::vector<TSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPr>;

    TSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPrVec sortVec;
    sortVec.reserve(m_Measures.size());

    for (auto measureVecIter = m_Measures.begin();
         measureVecIter != m_Measures.end(); ++measureVecIter) {
        const TLocalCategoryIdSizeStrPrMap& counts = measureVecIter->second;

        std::size_t total{0};
        for (auto mapIter = counts.begin(); mapIter != counts.end(); ++mapIter) {
            total += mapIter->second.first;
        }

        sortVec.emplace_back(total, measureVecIter);
    }

    // Sort descending
    using TGreaterSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPr =
        std::greater<TSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPr>;
    TGreaterSizeRegexLocalCategoryIdSizeStrPrMapPrVecCItrPr comp;
    std::sort(sortVec.begin(), sortVec.end(), comp);

    std::ostringstream strm;
    strm << "Results:" << core_t::LINE_ENDING;

    using TLocalCategoryIdSet = std::set<model::CLocalCategoryId>;

    TLocalCategoryIdSet usedCategories;
    std::size_t observedActuals{0};
    std::size_t good{0};

    // Iterate backwards through the sorted vector, so that the most common
    // actual categories are looked at first
    for (auto sortedVecIter = sortVec.begin(); sortedVecIter != sortVec.end(); ++sortedVecIter) {
        std::size_t total{sortedVecIter->first};
        if (total > 0) {
            ++observedActuals;
        }

        auto measureVecIter = sortedVecIter->second;

        const core::CRegex& regex = measureVecIter->first;
        strm << "Manual category defined by regex " << regex.str()
             << core_t::LINE_ENDING << "\tNumber of messages in manual category "
             << total << core_t::LINE_ENDING;

        const TLocalCategoryIdSizeStrPrMap& counts = measureVecIter->second;
        strm << "\tNumber of ML categories that include this manual category "
             << counts.size() << core_t::LINE_ENDING;

        if (counts.size() == 1) {
            std::size_t count{counts.begin()->second.first};
            model::CLocalCategoryId categoryId{counts.begin()->first};
            if (usedCategories.find(categoryId) != usedCategories.end()) {
                strm << "\t\t" << count << "\t(CATEGORY ALREADY USED)\t"
                     << counts.begin()->second.second << core_t::LINE_ENDING;
            } else {
                good += count;
                usedCategories.insert(categoryId);
            }
        } else if (counts.size() > 1) {
            strm << "\tBreakdown:" << core_t::LINE_ENDING;

            // Assume the category with the count closest to the actual count is
            // good (as long as it's not already used), and all other categories
            // are bad.
            std::size_t max{0};
            model::CLocalCategoryId maxCategoryId;
            for (auto mapIter = counts.begin(); mapIter != counts.end(); ++mapIter) {
                model::CLocalCategoryId categoryId{mapIter->first};

                std::size_t count{mapIter->second.first};
                const std::string& example = mapIter->second.second;
                strm << "\t\t" << count;
                if (usedCategories.find(categoryId) != usedCategories.end()) {
                    strm << "\t(CATEGORY ALREADY USED)";
                } else {
                    if (count > max) {
                        max = count;
                        maxCategoryId = categoryId;
                    }
                }
                strm << '\t' << example << core_t::LINE_ENDING;
            }
            if (maxCategoryId.isSoftFailure() == false) {
                good += max;
                usedCategories.insert(maxCategoryId);
            }
        }
    }

    strm << "Total number of messages passed to benchmarker " << m_TotalMessages
         << core_t::LINE_ENDING << "Total number of scored messages " << m_ScoredMessages
         << core_t::LINE_ENDING << "Number of scored messages correctly categorised by ML "
         << good << core_t::LINE_ENDING << "Overall accuracy for scored messages "
         << (double(good) / double(m_ScoredMessages)) * 100.0 << '%'
         << core_t::LINE_ENDING << "Percentage of manual categories detected at all "
         << (double(usedCategories.size()) / double(observedActuals)) * 100.0 << '%';

    LOG_DEBUG(<< strm.str());
}
}
}
