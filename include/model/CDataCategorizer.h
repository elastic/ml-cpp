/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CDataCategorizer_h
#define INCLUDED_ml_model_CDataCategorizer_h

#include <core/CMemoryUsage.h>
#include <core/CoreTypes.h>

#include <model/CCategoryExamplesCollector.h>
#include <model/CMonitoredResource.h>
#include <model/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <functional>
#include <memory>
#include <string>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CLimits;

//! \brief
//! Interface for classes that convert a raw event string to a category.
//!
//! DESCRIPTION:\n
//! Abstract interface for classes that convert a raw event string
//! to a category.
//!
//! IMPLEMENTATION DECISIONS:\n
//! At the time of writing, only the token list data categorizer implements
//! this interface.  However, it is not hard to imagine a time when
//! there are specialist data categorizers for XML, JSON or delimited files,
//! so it is good to have an abstract interface that they can all use.
//!
class MODEL_EXPORT CDataCategorizer : public CMonitoredResource {
public:
    //! Used for storing distinct token IDs
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    //! Shared pointer to an instance of this class
    using TDataCategorizerP = std::shared_ptr<CDataCategorizer>;

    //! Shared pointer to an instance of this class
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;
    using TIntVec = std::vector<int>;

public:
    CDataCategorizer(CLimits& limits, const std::string& fieldName);

    //! No copying allowed (because it would complicate the resource monitoring).
    CDataCategorizer(const CDataCategorizer&) = delete;
    CDataCategorizer& operator=(const CDataCategorizer&) = delete;

    ~CDataCategorizer() override;

    //! Dump stats
    virtual void dumpStats() const = 0;

    //! Compute a category from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.
    int computeCategory(bool isDryRun, const std::string& str, std::size_t rawStringLen);

    //! As above, but also take into account field names/values.
    virtual int computeCategory(bool isDryRun,
                                const TStrStrUMap& fields,
                                const std::string& str,
                                std::size_t rawStringLen) = 0;

    //! Create reverse search commands that will (more or less) just
    //! select the records that are classified as the given category when
    //! combined with the original search.  Note that the reverse search is
    //! only approximate - it may select more records than have actually
    //! been classified as the returned category.
    virtual bool createReverseSearch(int categoryId,
                                     std::string& part1,
                                     std::string& part2,
                                     std::size_t& maxMatchingLength,
                                     bool& wasCached) = 0;

    //! Has the data categorizer's state changed?
    virtual bool hasChanged() const = 0;

    //! Populate the object from part of a state document
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Make a function that can be called later to persist state in the
    //! foreground, i.e. in the knowledge that no other thread will be
    //! accessing the data structures this method accesses.
    virtual TPersistFunc makeForegroundPersistFunc() const = 0;

    //! Make a function that can be called later to persist state in the
    //! background, i.e. copying any required data such that other threads
    //! may modify the original data structures while persistence is taking
    //! place.
    virtual TPersistFunc makeBackgroundPersistFunc() const = 0;

    //! Access to the field name
    const std::string& fieldName() const;

    //! Access to last persistence time
    core_t::TTime lastPersistTime() const;

    //! Set last persistence time
    void lastPersistTime(core_t::TTime lastPersistTime);

    //! Debug the memory used by this categorizer.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this categorizer.
    std::size_t memoryUsage() const override;

    //! Add an example if the limit for the category has not be reached.
    //! \return true if the example was added, false if not.
    bool addExample(int categoryId, const std::string& example);

    //! Access to the examples collector
    const CCategoryExamplesCollector& examplesCollector() const;

    //! Restore the examples collector
    bool restoreExamplesCollector(core::CStateRestoreTraverser& traverser);

    virtual std::size_t numMatches(int categoryId) = 0;

    virtual TIntVec usurpedCategories(int categoryId) = 0;

protected:
    //! Used if no fields are supplied to the computeCategory() method.
    static const TStrStrUMap EMPTY_FIELDS;

private:
    //! Configurable limits
    CLimits& m_Limits;

    //! Which field name are we working on?
    std::string m_FieldName;

    //! Collects up to a configurable number of examples per category
    CCategoryExamplesCollector m_ExamplesCollector;

    //! When was data last persisted for this categorizer?  (0 means never.)
    core_t::TTime m_LastPersistTime;
};
}
}

#endif // INCLUDED_ml_model_CDataCategorizer_h
