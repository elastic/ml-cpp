/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CPatternSet_h
#define INCLUDED_ml_model_CPatternSet_h

#include <core/CFlatPrefixTree.h>
#include <core/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief A set that allows efficient lookups of strings.
//!
//! DESCRIPTION:\n
//! Users can define filters to be used in conjunction with rules.
//! This class models those filters as a set and allows efficient lookups.
//! The supported filter items are:
//!   - Full patterns: "foo" will match "foo" only
//!   - Prefix patterns: "foo*" will match "foo" and "foo_"
//!   - Suffix patterns: "*foo" will match "foo" and "_foo"
//!   - Contains patterns: "*foo*" will match "foo" and "_foo_"
//!
//! IMPLEMENTATION DECISIONS:\n
//! Upon building the set, patterns are categorised in the aforementioned 4
//! categories. They are then stored in a corresponding prefix tree that allows
//! efficient lookups. In particular a key is contained in the set if:
//!   - its start matches a prefix pattern
//!   - its end matched a suffix pattern
//!   - it matches fully against a full pattern
//!   - the start of any of its substrings ending at its end matches a contains pattern
class CORE_EXPORT CPatternSet {
public:
    using TStrVec = std::vector<std::string>;
    using TStrVecCItr = TStrVec::const_iterator;
    using TStrCItr = std::string::const_iterator;

public:
    //! Default constructor.
    CPatternSet();

    //! Initialise the set from JSON that is an array of strings.
    bool initFromJson(const std::string& json);

    //! Check if the set contains the given key.
    bool contains(const std::string& key) const;

    //! Clears the set.
    void clear();

private:
    void sortAndPruneDuplicates(TStrVec& keys);

private:
    //! The prefix tree containing full patterns (no wildcard).
    CFlatPrefixTree m_FullMatchPatterns;

    //! The prefix tree containing prefix patterns.
    CFlatPrefixTree m_PrefixPatterns;

    //! The prefix tree containing suffix patterns
    //! (note that the suffixes are stored reverted).
    CFlatPrefixTree m_SuffixPatterns;

    //! The prefix tree containing the contains patterns.
    CFlatPrefixTree m_ContainsPatterns;
};
}
}

#endif // INCLUDED_ml_model_CPatternSet_h
