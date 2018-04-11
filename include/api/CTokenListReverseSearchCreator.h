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
#ifndef INCLUDED_ml_api_CTokenListReverseSearchCreator_h
#define INCLUDED_ml_api_CTokenListReverseSearchCreator_h

#include <api/CTokenListReverseSearchCreatorIntf.h>

namespace ml {
namespace api {

//! \brief
//! Create Engine API reverse searches for categories of events.
//!
//! DESCRIPTION:\n
//! Creates Engine API reverse searches for categories of events defined
//! by lists of tokens. In particular, the search has two parts. The first
//! part is a space separated list of the search terms. The second part
//! is a regex to be used in order to match values of the categorized field
//! for the category in question.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The Engine API reverse search has the space separated list of the tokens
//! and the regex because most modern index-based storages accept such searches.
//!
class API_EXPORT CTokenListReverseSearchCreator : public CTokenListReverseSearchCreatorIntf {
public:
    CTokenListReverseSearchCreator(const std::string& fieldName);

    //! What's the maximum cost of tokens we can include in the reverse
    //! search?  This cost is loosely based on the maximum length of an
    //! Internet Explorer URL.
    virtual size_t availableCost() const;

    //! What would be the cost of adding the specified token occurring the
    //! specified number of times to the reverse search?
    virtual size_t costOfToken(const std::string& token, size_t numOccurrences) const;

    //! Create a reverse search for a NULL field value.
    virtual bool createNullSearch(std::string& part1, std::string& part2) const;

    //! If possible, create a reverse search for the case where there are no
    //! unique tokens identifying the type.  (If this is not possible return
    //! false.)
    virtual bool createNoUniqueTokenSearch(int type,
                                           const std::string& example,
                                           size_t maxMatchingStringLen,
                                           std::string& part1,
                                           std::string& part2) const;

    //! Initialise the two strings that form a reverse search.  For example,
    //! this could be as simple as clearing the strings or setting them to
    //! some sort of one-off preamble.
    virtual void initStandardSearch(int type,
                                    const std::string& example,
                                    size_t maxMatchingStringLen,
                                    std::string& part1,
                                    std::string& part2) const;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token, which may occur anywhere within the original
    //! message, but has been determined to be a good thing to distinguish
    //! this type of messages from other types.
    virtual void
    addCommonUniqueToken(const std::string& token, std::string& part1, std::string& part2) const;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token.
    virtual void addInOrderCommonToken(const std::string& token,
                                       bool first,
                                       std::string& part1,
                                       std::string& part2) const;

    //! Close off the two strings that form a reverse search.  For example,
    //! this may be when closing brackets need to be appended.
    virtual void closeStandardSearch(std::string& part1, std::string& part2) const;
};
}
}

#endif // INCLUDED_ml_api_CTokenListReverseSearchCreator_h
