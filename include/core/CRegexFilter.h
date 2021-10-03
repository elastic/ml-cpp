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
