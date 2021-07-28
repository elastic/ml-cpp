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
#ifndef INCLUDED_ml_api_CBenchMarker_h
#define INCLUDED_ml_api_CBenchMarker_h

#include <core/CRegex.h>

#include <model/CLocalCategoryId.h>

#include <api/ImportExport.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! Benchmark Ml categorisation using regexes.
//!
//! DESCRIPTION:\n
//! Benchmark the performance of Ml categorisation by
//! using mutually exclusive regexes created manually.  The
//! idea is that all messages that match a given regex should
//! be categorised the same way by Ml.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Messages that don't match any of the regexes are considered
//! to be hard to categorise (even by human being) and are hence
//! ignored for benchmarking purposes.
//!
class API_EXPORT CBenchMarker {
public:
    //! A count and and example string
    using TSizeStrPr = std::pair<std::size_t, std::string>;

    //! Used for mapping ML category to count and example
    using TLocalCategoryIdSizeStrPrMap = std::map<model::CLocalCategoryId, TSizeStrPr>;
    using TLocalCategoryIdSizeStrPrMapItr = TLocalCategoryIdSizeStrPrMap::iterator;
    using TLocalCategoryIdSizeStrPrMapCItr = TLocalCategoryIdSizeStrPrMap::const_iterator;

    //! A regex and its corresponding category count map
    using TRegexLocalCategoryIdSizeStrPrMapPr =
        std::pair<core::CRegex, TLocalCategoryIdSizeStrPrMap>;

    //! Vector of regexes with corresponding category count maps
    using TRegexLocalCategoryIdSizeStrPrMapPrVec =
        std::vector<TRegexLocalCategoryIdSizeStrPrMapPr>;
    using TRegexLocalCategoryIdSizeStrPrMapPrVecItr =
        TRegexLocalCategoryIdSizeStrPrMapPrVec::iterator;
    using TRegexLocalCategoryIdSizeStrPrMapPrVecCItr =
        TRegexLocalCategoryIdSizeStrPrMapPrVec::const_iterator;

public:
    CBenchMarker();

    //! Initialise from a file
    bool init(const std::string& regexFilename);

    //! Add a message together with the category ML assigned to it
    void addResult(const std::string& message, model::CLocalCategoryId categoryId);

    void dumpResults() const;

private:
    //! Number of messages passed to the benchmarker
    std::size_t m_TotalMessages;

    //! Number of messages that matched one of the regexes, and hence
    //! contribute to the scoring
    std::size_t m_ScoredMessages;

    //! The string and tokens we base this category on
    TRegexLocalCategoryIdSizeStrPrMapPrVec m_Measures;
};
}
}

#endif // INCLUDED_ml_api_CBenchMarker_h
