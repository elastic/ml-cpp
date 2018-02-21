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
#ifndef INCLUDED_ml_api_CCategoryExamplesCollector_h
#define INCLUDED_ml_api_CCategoryExamplesCollector_h

#include <api/ImportExport.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <boost/unordered_map.hpp>

#include <set>
#include <string>

namespace ml
{
namespace api
{

//! \brief
//! Collects up to a configurable number of distinct examples per category
//!
//! IMPLEMENTATION DECISIONS:\n
//! The examples are stored sorted in a boost flat_set. The flat_set was
//! preferred due to the contiguous storage which in combination with the
//! small number of expected examples should be more performant than a
//! traditional set.
//!
class API_EXPORT CCategoryExamplesCollector
{
    public:
        typedef std::set<std::string> TStrSet;
        typedef TStrSet::const_iterator TStrSetCItr;

        //! Truncate examples to be no longer than this
        static const size_t MAX_EXAMPLE_LENGTH;

    public:
        CCategoryExamplesCollector(std::size_t maxExamples);
        CCategoryExamplesCollector(std::size_t maxExamples, core::CStateRestoreTraverser &traverser);

        //! Adds the example to the category if the example is a new
        //! distinct example and if there are less than the maximum
        //! number of examples for the given category.
        //! Returns true if the example was added or false otherwise.
        bool add(std::size_t category, const std::string &example);

        //! Returns the number of examples currently stored for a given category.
        std::size_t numberOfExamplesForCategory(std::size_t category) const;

        const TStrSet &examples(std::size_t category) const;

        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Populate the object from part of a state document
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Clear all examples
        void clear(void);

    private:
        using TSizeStrSetUMap = boost::unordered_map<std::size_t, TStrSet>;

    private:
        void persistExamples(std::size_t category,
                             const TStrSet &examples,
                             core::CStatePersistInserter &inserter) const;
        bool restoreExamples(core::CStateRestoreTraverser &traverser);

        //! Truncate long examples to MAX_EXAMPLE_LENGTH bytes, appending an
        //! ellipsis to those that are truncated.
        std::string truncateExample(std::string example);

    private:
        //! The max number of examples that will be collected per category
        std::size_t     m_MaxExamples;

        //! A map from categories to the set that contains the examples
        TSizeStrSetUMap m_ExamplesByCategory;
};

}
}

#endif // INCLUDED_ml_api_CCategoryExamplesCollector_h
