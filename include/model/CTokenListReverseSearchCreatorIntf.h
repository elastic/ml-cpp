/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CTokenListReverseSearchCreatorIntf_h
#define INCLUDED_ml_model_CTokenListReverseSearchCreatorIntf_h

#include <model/ImportExport.h>

#include <string>

namespace ml {
namespace model {

//! \brief
//! Interface for classes that create reverse searches for the token
//! list data categorizer.
//!
//! DESCRIPTION:\n
//! Abstract interface for classes that create reverse searches for
//! the token list data categorizer.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All methods are const, hence derived classes should have no
//! state that changes after construction.  (If this rule needs to
//! be changed then a strategy for appropriately copying these
//! objects will need implementing within the token list data
//! categorizer, because at present duplicating a const pointer is deemed
//! adequate.)
//!
class MODEL_EXPORT CTokenListReverseSearchCreatorIntf {
public:
    CTokenListReverseSearchCreatorIntf(const std::string& fieldName);

    //! Virtual destructor for an abstract base class
    virtual ~CTokenListReverseSearchCreatorIntf();

    //! What's the maximum cost of tokens we can include in the reverse
    //! search?  Derived classes can decide what they mean by cost, as they
    //! also decide the cost of each token.
    virtual size_t availableCost() const = 0;

    //! What would be the cost of adding the specified token occurring the
    //! specified number of times to the reverse search?  Derived classes
    //! can decide what they mean by cost, as they also decided what the
    //! maximum permitted total cost is.
    virtual size_t costOfToken(const std::string& token, size_t numOccurrences) const = 0;

    //! If possible, create a reverse search for a NULL field value.  (If
    //! this is not possible return false.)
    virtual bool createNullSearch(std::string& part1, std::string& part2) const = 0;

    //! If possible, create a reverse search for the case where there are no
    //! unique tokens identifying the category.  (If this is not possible return
    //! false.)
    virtual bool createNoUniqueTokenSearch(int categoryId,
                                           const std::string& example,
                                           size_t maxMatchingStringLen,
                                           std::string& part1,
                                           std::string& part2) const = 0;

    //! Initialise the two strings that form a reverse search.  For example,
    //! this could be as simple as clearing the strings or setting them to
    //! some sort of one-off preamble.
    virtual void initStandardSearch(int categoryId,
                                    const std::string& example,
                                    size_t maxMatchingStringLen,
                                    std::string& part1,
                                    std::string& part2) const = 0;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token, which may occur anywhere within the original
    //! message, but has been determined to be a good thing to distinguish
    //! this category of messages from other categories.
    virtual void addCommonUniqueToken(const std::string& token,
                                      std::string& part1,
                                      std::string& part2) const = 0;

    //! Modify the two strings that form a reverse search to account for the
    //! specified token.
    virtual void addInOrderCommonToken(const std::string& token,
                                       bool first,
                                       std::string& part1,
                                       std::string& part2) const = 0;

    //! Close off the two strings that form a reverse search.  For example,
    //! this may be when closing brackets need to be appended.
    virtual void closeStandardSearch(std::string& part1, std::string& part2) const;

    //! Access to the field name
    const std::string& fieldName() const;

private:
    //! Which field name is being used for categorisation?
    std::string m_FieldName;
};
}
}

#endif // INCLUDED_ml_model_CTokenListReverseSearchCreatorIntf_h
