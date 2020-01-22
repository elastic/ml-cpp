/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CTokenListReverseSearchCreator_h
#define INCLUDED_ml_model_CTokenListReverseSearchCreator_h

#include <core/CMemoryUsage.h>

#include <model/ImportExport.h>

#include <string>

namespace ml {
namespace model {

//! \brief
//! Create reverse searches for categories of events.
//!
//! DESCRIPTION:\n
//! Creates reverse searches for categories of events defined by lists
//! of tokens. In particular, the search has two parts. The first part
//! is a space separated list of the search terms. The second part is
//! a regex to be used in order to match values of the categorized field
//! for the category in question.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The reverse search includes the space separated list of the tokens as
//! the fast way to search the inverted index, plus the regex to confirm
//! the tokens are in the required order and maximum length to prevent
//! short token lists matching much longer messages.
//!
class MODEL_EXPORT CTokenListReverseSearchCreator {
public:
    CTokenListReverseSearchCreator(const std::string& fieldName);

    //! What's the maximum cost of tokens we can include in the reverse
    //! search?  This cost is loosely based on the maximum length of an
    //! Internet Explorer URL.
    std::size_t availableCost() const;

    //! What would be the cost of adding the specified token occurring the
    //! specified number of times to the reverse search?
    std::size_t costOfToken(const std::string& token, std::size_t numOccurrences) const;

    //! If possible, create a reverse search for the case where there are no
    //! unique tokens identifying the category.  (If this is not possible return
    //! false.)
    bool createNoUniqueTokenSearch(int categoryId,
                                   const std::string& example,
                                   std::size_t maxMatchingStringLen,
                                   std::string& terms,
                                   std::string& regex) const;

    //! Initialise the two strings that form a reverse search.  For example,
    //! this could be as simple as clearing the strings or setting them to
    //! some sort of one-off preamble.
    void initStandardSearch(int categoryId,
                            const std::string& example,
                            std::size_t maxMatchingStringLen,
                            std::string& terms,
                            std::string& regex) const;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token.
    void addInOrderCommonToken(const std::string& token, std::string& terms, std::string& regex) const;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token, which may occur anywhere within the original
    //! message, but has been determined to be a good thing to distinguish
    //! this category of messages from other categories.
    void addOutOfOrderCommonToken(const std::string& token,
                                  std::string& terms,
                                  std::string& regex) const;

    //! Close off the two strings that form a reverse search.  For example,
    //! this may be when closing brackets need to be appended.
    void closeStandardSearch(std::string& terms, std::string& regex) const;

    //! Access to the field name
    const std::string& fieldName() const;

    //! Debug the memory used by this reverse search creator.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this reverse search creator.
    std::size_t memoryUsage() const;

private:
    //! Which field name is being used for categorization?
    std::string m_FieldName;
};
}
}

#endif // INCLUDED_ml_model_CTokenListReverseSearchCreator_h
