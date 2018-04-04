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
#ifndef INCLUDED_ml_core_CRegexFilter_h
#define INCLUDED_ml_core_CRegexFilter_h

#include <core/ImportExport.h>

#include <core/CRegex.h>

#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief
//! Filters strings based on a list of regular expressions
//!
//! DESCRIPTION:\n
//! The filter is configured according to a vector of regular
//! expressions. It can then be applied to strings. The filter
//! will iteratively apply each regex to the string until no
//! match can be found and it will remove all matched substrings.
//!
class CORE_EXPORT CRegexFilter {
public:
    using TRegexVec = std::vector<CRegex>;
    using TStrVec = std::vector<std::string>;

public:
    CRegexFilter();

    //! Configures the filter for the given \p regularExpressions.
    bool configure(const TStrVec& regularExpressions);

    //! Applies the filter to \p target.
    std::string apply(const std::string& target) const;

    //! Returns true if the filter is empty.
    bool empty() const;

private:
    //! The regular expressions comprising the filter.
    TRegexVec m_Regex;
};
}
}

#endif // INCLUDED_ml_core_CRegexFilter_h
