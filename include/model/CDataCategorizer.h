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
#include <model/CLocalCategoryId.h>
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
    //! Used for formatting category IDs in the debug dump
    using TLocalCategoryIdFormatterFunc = std::function<std::string(CLocalCategoryId)>;

    //! Used for storing distinct token IDs
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    //! Unique pointer to an instance of this class
    using TDataCategorizerUPtr = std::unique_ptr<CDataCategorizer>;

    //! Function used for persisting objects of this class
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;

    //! Vector of local category IDs
    using TLocalCategoryIdVec = std::vector<CLocalCategoryId>;

    //! Callback for category definition output
    using TCategoryOutputFunc =
        std::function<void(CLocalCategoryId, const std::string&, const std::string&, std::size_t, const CCategoryExamplesCollector::TStrFSet&, std::size_t, TLocalCategoryIdVec)>;

    //! Callback for categorizer stats output
    using TCategorizerStatsOutputFunc = std::function<void(const SCategorizerStats&, bool)>;

public:
    CDataCategorizer(CLimits& limits, const std::string& fieldName);

    //! No copying allowed (because it would complicate the resource monitoring).
    CDataCategorizer(const CDataCategorizer&) = delete;
    CDataCategorizer& operator=(const CDataCategorizer&) = delete;

    ~CDataCategorizer() override;

    //! Dump stats
    virtual void dumpStats(const TLocalCategoryIdFormatterFunc& formatterFunc) const = 0;

    //! Compute a category from a string.  The raw string length may be longer
    //! than the length of the passed string, because the passed string may
    //! have the date stripped out of it.
    CLocalCategoryId computeCategory(bool isDryRun, const std::string& str, std::size_t rawStringLen);

    //! As above, but also take into account field names/values.
    virtual CLocalCategoryId computeCategory(bool isDryRun,
                                             const TStrStrUMap& fields,
                                             const std::string& str,
                                             std::size_t rawStringLen) = 0;

    //! Ensure the reverse search information is up-to-date for the specified
    //! category.  Note that the reverse search is only approximate - it may
    //! select more records than have actually been classified as the specified
    //! category.
    //! \return Was the reverse search changed as a result of the call?
    virtual bool cacheReverseSearch(CLocalCategoryId categoryId) = 0;

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

    //! Get the most recent categorization status.
    virtual model_t::ECategorizationStatus categorizationStatus() const = 0;

    //! Debug the memory used by this categorizer.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this categorizer.
    std::size_t memoryUsage() const override;

    //! Add an example if the limit for the category has not be reached.
    //! \return true if the example was added, false if not.
    bool addExample(CLocalCategoryId categoryId, const std::string& example);

    //! Access to the examples collector
    const CCategoryExamplesCollector& examplesCollector() const;

    //! Restore the examples collector
    bool restoreExamplesCollector(core::CStateRestoreTraverser& traverser);

    //! Number of matches for the specified category.
    virtual std::size_t numMatches(CLocalCategoryId categoryId) = 0;

    //! Get the categories that will never be detected again because the
    //! specified category will always be returned instead.
    virtual TLocalCategoryIdVec usurpedCategories(CLocalCategoryId categoryId) const = 0;

    //! Writes information about a category using the supplied output function,
    //! if the category has changed since the last time it was written.
    //! \return Was the category written?
    virtual bool writeCategoryIfChanged(CLocalCategoryId categoryId,
                                        const TCategoryOutputFunc& outputFunc) = 0;

    //! Writes information about all categories that have changed since the last
    //! time they were written using the supplied output function.
    //! \return Number of categories written.
    virtual std::size_t writeChangedCategories(const TCategoryOutputFunc& outputFunc) = 0;

    //! Write the latest categorizer stats using the supplied output function if
    //! they have changed since the last time they were written.
    //! \return Were the stats written?
    virtual bool
    writeCategorizerStatsIfChanged(const TCategorizerStatsOutputFunc& outputFunc) = 0;

    //! Quickly check if a stats write is important at this time.  This method
    //! is called frequently, so should not do costly processing.
    virtual bool isStatsWriteUrgent() const = 0;

    //! Number of categories this categorizer has detected.
    virtual std::size_t numCategories() const = 0;

    //! Is it permissable to create new categories?  New categories are
    //! not permitted when memory use has exceeded the limit.
    bool areNewCategoriesAllowed();

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
};
}
}

#endif // INCLUDED_ml_model_CDataCategorizer_h
