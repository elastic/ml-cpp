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
#ifndef INCLUDED_ml_api_CBenchMarker_h
#define INCLUDED_ml_api_CBenchMarker_h

#include <core/CRegex.h>

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
    typedef std::pair<size_t, std::string> TSizeStrPr;

    //! Used for mapping Ml type to count and example
    typedef std::map<int, TSizeStrPr> TIntSizeStrPrMap;
    typedef TIntSizeStrPrMap::iterator TIntSizeStrPrMapItr;
    typedef TIntSizeStrPrMap::const_iterator TIntSizeStrPrMapCItr;

    //! A regex and its corresponding type count map
    typedef std::pair<core::CRegex, TIntSizeStrPrMap> TRegexIntSizeStrPrMapPr;

    //! Vector of regexes with corresponding type count maps
    typedef std::vector<TRegexIntSizeStrPrMapPr> TRegexIntSizeStrPrMapPrVec;
    typedef TRegexIntSizeStrPrMapPrVec::iterator TRegexIntSizeStrPrMapPrVecItr;
    typedef TRegexIntSizeStrPrMapPrVec::const_iterator TRegexIntSizeStrPrMapPrVecCItr;

public:
    CBenchMarker(void);

    //! Initialise from a file
    bool init(const std::string &regexFilename);

    //! Add a message together with the type Ml assigned to it
    void addResult(const std::string &message, int type);

    void dumpResults(void) const;

private:
    //! Number of messages passed to the benchmarker
    size_t m_TotalMessages;

    //! Number of messages that matched one of the regexes, and hence
    //! contribute to the scoring
    size_t m_ScoredMessages;

    //! The string and tokens we base this type on
    TRegexIntSizeStrPrMapPrVec m_Measures;
};
}
}

#endif// INCLUDED_ml_api_CBenchMarker_h
