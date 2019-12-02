/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CMetricBucketGatherer.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>

#include <model/CGathererTools.h>
#include <model/CResourceMonitor.h>
#include <model/CSampleCounts.h>
#include <model/CSampleGatherer.h>
#include <model/CSearchKey.h>

#include <boost/any.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include <atomic>
#include <map>
#include <utility>
#include <vector>

namespace ml {
namespace model {

namespace {

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TStrCRef = std::reference_wrapper<const std::string>;
using TStrCRefStrCRefPr = std::pair<TStrCRef, TStrCRef>;
using TStrCRefStrCRefPrUInt64Map =
    std::map<TStrCRefStrCRefPr, uint64_t, maths::COrderings::SLexicographicalCompare>;
using TSampleVec = std::vector<CSample>;
using TSizeMeanGathererUMap = boost::unordered_map<std::size_t, CGathererTools::TMeanGatherer>;
using TSizeSizeMeanGathererUMapUMap = boost::unordered_map<std::size_t, TSizeMeanGathererUMap>;
using TSizeMedianGathererUMap =
    boost::unordered_map<std::size_t, CGathererTools::TMedianGatherer>;
using TSizeSizeMedianGathererUMapUMap = boost::unordered_map<std::size_t, TSizeMedianGathererUMap>;
using TSizeMinGathererUMap = boost::unordered_map<std::size_t, CGathererTools::TMinGatherer>;
using TSizeSizeMinGathererUMapUMap = boost::unordered_map<std::size_t, TSizeMinGathererUMap>;
using TSizeMaxGathererUMap = boost::unordered_map<std::size_t, CGathererTools::TMaxGatherer>;
using TSizeSizeMaxGathererUMapUMap = boost::unordered_map<std::size_t, TSizeMaxGathererUMap>;
using TSizeVarianceGathererUMap =
    boost::unordered_map<std::size_t, CGathererTools::TVarianceGatherer>;
using TSizeSizeVarianceGathererUMapUMap =
    boost::unordered_map<std::size_t, TSizeVarianceGathererUMap>;
using TSizeSumGathererUMap = boost::unordered_map<std::size_t, CGathererTools::CSumGatherer>;
using TSizeSizeSumGathererUMapUMap = boost::unordered_map<std::size_t, TSizeSumGathererUMap>;
using TSizeMultivariateMeanGathererUMap =
    boost::unordered_map<std::size_t, CGathererTools::TMultivariateMeanGatherer>;
using TSizeSizeMultivariateMeanGathererUMapUMap =
    boost::unordered_map<std::size_t, TSizeMultivariateMeanGathererUMap>;
using TSizeMultivariateMinGathererUMap =
    boost::unordered_map<std::size_t, CGathererTools::TMultivariateMinGatherer>;
using TSizeSizeMultivariateMinGathererUMapUMap =
    boost::unordered_map<std::size_t, TSizeMultivariateMinGathererUMap>;
using TSizeMultivariateMaxGathererUMap =
    boost::unordered_map<std::size_t, CGathererTools::TMultivariateMaxGatherer>;
using TSizeSizeMultivariateMaxGathererUMapUMap =
    boost::unordered_map<std::size_t, TSizeMultivariateMaxGathererUMap>;
using TSizeFeatureDataPr = std::pair<std::size_t, SMetricFeatureData>;
using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, SMetricFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TSizeSizePrUInt64UMap = CMetricBucketGatherer::TSizeSizePrUInt64UMap;
using TCategorySizePr = CMetricBucketGatherer::TCategorySizePr;
using TCategorySizePrAnyMap = CMetricBucketGatherer::TCategorySizePrAnyMap;
using TCategorySizePrAnyMapItr = CMetricBucketGatherer::TCategorySizePrAnyMapItr;
using TCategorySizePrAnyMapCItr = CMetricBucketGatherer::TCategorySizePrAnyMapCItr;
using TStoredStringPtrVec = CBucketGatherer::TStoredStringPtrVec;

template<typename T>
using TSizeTUMap = boost::unordered_map<std::size_t, T>;
template<typename T>
using TSizeSizeTUMapUMap = boost::unordered_map<std::size_t, TSizeTUMap<T>>;

const std::string CURRENT_VERSION("1");

// We use short field names to reduce the state size
const std::string BASE_TAG("a");
const std::string VERSION_TAG("b");
const std::string MEAN_TAG("e");
const std::string MIN_TAG("f");
const std::string MAX_TAG("g");
const std::string SUM_TAG("h");
const std::string MULTIVARIATE_MEAN_TAG("i");
const std::string MULTIVARIATE_MIN_TAG("j");
const std::string MULTIVARIATE_MAX_TAG("k");
const std::string MEDIAN_TAG("l");
const std::string VARIANCE_TAG("m");
const std::string EMPTY_STRING;
const TDoubleVec EMPTY_DOUBLE_VEC;

// Nested tags.
const std::string ATTRIBUTE_TAG("a");
const std::string DATA_TAG("b");
const std::string PERSON_TAG("c");

//! Get the by field name.
const std::string& byField(bool population, const TStrVec& fieldNames) {
    return population ? fieldNames[1] : fieldNames[0];
}

//! Get the over field name.
const std::string& overField(bool population, const TStrVec& fieldNames) {
    return population ? fieldNames[0] : EMPTY_STRING;
}

template<model_t::EMetricCategory>
struct SDataType {};
template<>
struct SDataType<model_t::E_Mean> {
    using Type = TSizeSizeMeanGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_Median> {
    using Type = TSizeSizeMedianGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_Min> {
    using Type = TSizeSizeMinGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_Max> {
    using Type = TSizeSizeMaxGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_Sum> {
    using Type = TSizeSizeSumGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_Variance> {
    using Type = TSizeSizeVarianceGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_MultivariateMean> {
    using Type = TSizeSizeMultivariateMeanGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_MultivariateMin> {
    using Type = TSizeSizeMultivariateMinGathererUMapUMap;
};
template<>
struct SDataType<model_t::E_MultivariateMax> {
    using Type = TSizeSizeMultivariateMaxGathererUMapUMap;
};
template<typename ITR, typename T>
struct SMaybeConst {};
template<typename T>
struct SMaybeConst<TCategorySizePrAnyMapItr, T> {
    using Type = T;
};
template<typename T>
struct SMaybeConst<TCategorySizePrAnyMapCItr, T> {
    using Type = const T;
};

//! Register the callbacks for computing the size of feature data gatherers
//! with \p visitor.
template<typename VISITOR>
void registerMemoryCallbacks(VISITOR& visitor) {
    visitor.template registerCallback<TSizeSizeMeanGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMedianGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMinGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMaxGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeVarianceGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeSumGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMultivariateMeanGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMultivariateMinGathererUMapUMap>();
    visitor.template registerCallback<TSizeSizeMultivariateMaxGathererUMapUMap>();
}

//! Register the callbacks for computing the size of feature data gatherers.
void registerMemoryCallbacks() {
    static std::atomic_flag once = ATOMIC_FLAG_INIT;
    if (once.test_and_set() == false) {
        registerMemoryCallbacks(core::CMemory::anyVisitor());
        registerMemoryCallbacks(core::CMemoryDebug::anyVisitor());
    }
}

//! Apply a function \p f to a gatherer held as a value by map entry \p i
//! of an explicit metric category
template<model_t::EMetricCategory CATEGORY, typename ITR, typename F>
void applyFunc(ITR i, const F& f) {
    using TDataType = typename SDataType<CATEGORY>::Type;
    f(i->first, boost::any_cast<typename SMaybeConst<ITR, TDataType>::Type&>(i->second));
}

//! Apply a function \p f to all the gatherers held in [\p begin, \p end).
template<typename ITR, typename F>
bool applyFunc(ITR begin, ITR end, const F& f) {
    for (ITR i = begin; i != end; ++i) {
        model_t::EMetricCategory category = i->first.first;
        try {
            switch (category) {
            case model_t::E_Mean:
                applyFunc<model_t::E_Mean>(i, f);
                break;
            case model_t::E_Median:
                applyFunc<model_t::E_Median>(i, f);
                break;
            case model_t::E_Min:
                applyFunc<model_t::E_Min>(i, f);
                break;
            case model_t::E_Max:
                applyFunc<model_t::E_Max>(i, f);
                break;
            case model_t::E_Variance:
                applyFunc<model_t::E_Variance>(i, f);
                break;
            case model_t::E_Sum:
                applyFunc<model_t::E_Sum>(i, f);
                break;
            case model_t::E_MultivariateMean:
                applyFunc<model_t::E_MultivariateMean>(i, f);
                break;
            case model_t::E_MultivariateMin:
                applyFunc<model_t::E_MultivariateMin>(i, f);
                break;
            case model_t::E_MultivariateMax:
                applyFunc<model_t::E_MultivariateMax>(i, f);
                break;
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Apply failed for " << category << ": " << e.what());
            return false;
        }
    }

    return true;
}

//! Apply a function \p f to all the gatherers held in \p data.
template<typename T, typename F>
bool applyFunc(T& data, const F& f) {
    return applyFunc(data.begin(), data.end(), f);
}

//! Initialize feature data for a specific category
template<model_t::EMetricCategory CATEGORY>
void initializeFeatureDataInstance(std::size_t dimension, TCategorySizePrAnyMap& featureData) {
    using Type = typename SDataType<CATEGORY>::Type;
    featureData[{CATEGORY, dimension}] = Type();
}

//! Persists the data gatherers (for individual metric categories).
class CPersistFeatureData {
public:
    template<typename T>
    void operator()(const TCategorySizePr& category,
                    const TSizeSizeTUMapUMap<T>& data,
                    core::CStatePersistInserter& inserter) const {
        if (data.empty()) {
            inserter.insertValue(this->tagName(category), EMPTY_STRING);
            return;
        }

        inserter.insertLevel(
            this->tagName(category),
            std::bind<void>(SDoPersist(), std::cref(data), std::placeholders::_1));
    }

private:
    std::string tagName(const TCategorySizePr& category) const {
        switch (category.first) {
        case model_t::E_Mean:
            return MEAN_TAG;
        case model_t::E_Median:
            return MEDIAN_TAG;
        case model_t::E_Min:
            return MIN_TAG;
        case model_t::E_Max:
            return MAX_TAG;
        case model_t::E_Variance:
            return VARIANCE_TAG;
        case model_t::E_Sum:
            return SUM_TAG;
        case model_t::E_MultivariateMean:
            return MULTIVARIATE_MEAN_TAG +
                   core::CStringUtils::typeToString(category.second);
        case model_t::E_MultivariateMin:
            return MULTIVARIATE_MIN_TAG +
                   core::CStringUtils::typeToString(category.second);
        case model_t::E_MultivariateMax:
            return MULTIVARIATE_MAX_TAG +
                   core::CStringUtils::typeToString(category.second);
        }
        return EMPTY_STRING;
    }

    struct SDoPersist {
        template<typename T>
        void operator()(const TSizeSizeTUMapUMap<T>& data,
                        core::CStatePersistInserter& inserter) const {
            using TSizeSizeTUMapUMapCItr = typename TSizeSizeTUMapUMap<T>::const_iterator;
            std::vector<TSizeSizeTUMapUMapCItr> dataItrs;
            dataItrs.reserve(data.size());
            for (auto i = data.cbegin(); i != data.cend(); ++i) {
                dataItrs.push_back(i);
            }
            std::sort(dataItrs.begin(), dataItrs.end(),
                      [](TSizeSizeTUMapUMapCItr lhs, TSizeSizeTUMapUMapCItr rhs) {
                          return lhs->first < rhs->first;
                      });

            for (auto itr : dataItrs) {
                inserter.insertLevel(ATTRIBUTE_TAG,
                                     std::bind<void>(SDoPersist(), itr->first,
                                                     std::cref(itr->second),
                                                     std::placeholders::_1));
            }
        }

        template<typename T>
        void operator()(std::size_t cid,
                        const TSizeTUMap<T>& pidMap,
                        core::CStatePersistInserter& inserter) const {
            inserter.insertValue(ATTRIBUTE_TAG, cid);

            using TSizeTUMapCItr = typename TSizeTUMap<T>::const_iterator;
            std::vector<TSizeTUMapCItr> pidItrs;
            pidItrs.reserve(pidMap.size());
            for (auto i = pidMap.cbegin(); i != pidMap.cend(); ++i) {
                pidItrs.push_back(i);
            }
            std::sort(pidItrs.begin(), pidItrs.end(),
                      [](TSizeTUMapCItr lhs, TSizeTUMapCItr rhs) {
                          return lhs->first < rhs->first;
                      });

            for (auto itr : pidItrs) {
                inserter.insertLevel(PERSON_TAG,
                                     std::bind<void>(SDoPersist(), itr->first,
                                                     std::cref(itr->second),
                                                     std::placeholders::_1));
            }
        }

        template<typename T>
        void operator()(std::size_t pid, const T& data, core::CStatePersistInserter& inserter) const {
            inserter.insertValue(PERSON_TAG, pid);
            inserter.insertLevel(DATA_TAG, std::bind<void>(&T::acceptPersistInserter, &data,
                                                           std::placeholders::_1));
        }
    };
};

//! Restores the data gatherers (for individual metric categories).
template<model_t::EMetricCategory CATEGORY>
class CRestoreFeatureData {
public:
    bool operator()(core::CStateRestoreTraverser& traverser,
                    std::size_t dimension,
                    bool isNewVersion,
                    const CMetricBucketGatherer& gatherer,
                    TCategorySizePrAnyMap& result) const {
        boost::any& data = result[{CATEGORY, dimension}];
        return this->restore(traverser, dimension, isNewVersion, gatherer, data);
    }

private:
    //! Add a restored data gatherer to \p result.
    bool restore(core::CStateRestoreTraverser& traverser,
                 std::size_t dimension,
                 bool isNewVersion,
                 const CMetricBucketGatherer& gatherer,
                 boost::any& result) const {
        using Type = typename SDataType<CATEGORY>::Type;
        if (result.empty()) {
            result = Type();
        }
        Type& data = *boost::unsafe_any_cast<Type>(&result);

        // An empty sub-level implies a person with 100% invalid data.
        if (!traverser.hasSubLevel()) {
            return true;
        }

        if (isNewVersion) {
            return traverser.traverseSubLevel(
                std::bind<bool>(CDoNewRestore(dimension), std::placeholders::_1,
                                std::cref(gatherer), std::ref(data)));
        } else {
            return traverser.traverseSubLevel(
                std::bind<bool>(CDoOldRestore(dimension), std::placeholders::_1,
                                std::cref(gatherer), std::ref(data)));
        }
    }

    //! \brief Responsible for restoring individual gatherers.
    class CDoNewRestore {
    public:
        CDoNewRestore(std::size_t dimension) : m_Dimension(dimension) {}

        template<typename T>
        bool operator()(core::CStateRestoreTraverser& traverser,
                        const CMetricBucketGatherer& gatherer,
                        TSizeSizeTUMapUMap<T>& result) const {
            do {
                const std::string& name = traverser.name();
                if (name == ATTRIBUTE_TAG) {
                    if (traverser.traverseSubLevel(std::bind<bool>(
                            &CDoNewRestore::restoreAttributes<T>, this, std::placeholders::_1,
                            std::cref(gatherer), std::ref(result))) == false) {
                        LOG_ERROR(<< "Invalid data in " << traverser.value());
                        return false;
                    }
                }
            } while (traverser.next());

            return true;
        }

        template<typename T>
        bool restoreAttributes(core::CStateRestoreTraverser& traverser,
                               const CMetricBucketGatherer& gatherer,
                               TSizeSizeTUMapUMap<T>& result) const {
            std::size_t lastCid(0);
            bool seenCid(false);

            do {
                const std::string& name = traverser.name();
                if (name == ATTRIBUTE_TAG) {
                    if (core::CStringUtils::stringToType(traverser.value(), lastCid) == false) {
                        LOG_ERROR(<< "Invalid attribute ID in " << traverser.value());
                        return false;
                    }
                    seenCid = true;
                    result[lastCid] = TSizeTUMap<T>(1);
                } else if (name == PERSON_TAG) {
                    if (!seenCid) {
                        LOG_ERROR(<< "Incorrect format - person before attribute ID in "
                                  << traverser.value());
                        return false;
                    }
                    if (traverser.traverseSubLevel(std::bind<bool>(
                            &CDoNewRestore::restorePeople<T>, this, std::placeholders::_1,
                            std::cref(gatherer), std::ref(result[lastCid]))) == false) {
                        LOG_ERROR(<< "Invalid data in " << traverser.value());
                        return false;
                    }
                }
            } while (traverser.next());

            return true;
        }

        template<typename T>
        bool restorePeople(core::CStateRestoreTraverser& traverser,
                           const CMetricBucketGatherer& gatherer,
                           TSizeTUMap<T>& result) const {
            std::size_t lastPid(0);
            bool seenPid(false);

            do {
                const std::string& name = traverser.name();
                if (name == PERSON_TAG) {
                    if (core::CStringUtils::stringToType(traverser.value(), lastPid) == false) {
                        LOG_ERROR(<< "Invalid person ID in " << traverser.value());
                        return false;
                    }
                    seenPid = true;
                } else if (name == DATA_TAG) {
                    if (!seenPid) {
                        LOG_ERROR(<< "Incorrect format - data before person ID in "
                                  << traverser.value());
                        return false;
                    }
                    T initial(gatherer.dataGatherer().params(), m_Dimension,
                              gatherer.currentBucketStartTime(),
                              gatherer.bucketLength(), gatherer.beginInfluencers(),
                              gatherer.endInfluencers());
                    if (traverser.traverseSubLevel(
                            std::bind<bool>(&T::acceptRestoreTraverser, &initial,
                                            std::placeholders::_1)) == false) {
                        LOG_ERROR(<< "Invalid data in " << traverser.value());
                        return false;
                    }
                    result.emplace(lastPid, initial);
                }
            } while (traverser.next());

            return true;
        }

    private:
        std::size_t m_Dimension;
    };

    //! \brief Responsible for restoring individual gatherers.
    class CDoOldRestore {
    public:
        CDoOldRestore(std::size_t dimension) : m_Dimension(dimension) {}

        template<typename T>
        bool operator()(core::CStateRestoreTraverser& traverser,
                        const CMetricBucketGatherer& gatherer,
                        TSizeSizeTUMapUMap<T>& result) const {
            bool isPopulation = gatherer.dataGatherer().isPopulation();
            if (isPopulation) {
                this->restorePopulation(traverser, gatherer, result);
            } else {
                this->restoreIndividual(traverser, gatherer, result);
            }
            return true;
        }

        template<typename T>
        bool restoreIndividual(core::CStateRestoreTraverser& traverser,
                               const CMetricBucketGatherer& gatherer,
                               TSizeSizeTUMapUMap<T>& result) const {
            std::size_t pid(0);
            do {
                const std::string& name = traverser.name();
                if (name == DATA_TAG) {
                    T initial(gatherer.dataGatherer().params(), m_Dimension,
                              gatherer.currentBucketStartTime(),
                              gatherer.bucketLength(), gatherer.beginInfluencers(),
                              gatherer.endInfluencers());
                    if (traverser.traverseSubLevel(
                            std::bind(&T::acceptRestoreTraverser, &initial,
                                      std::placeholders::_1)) == false) {
                        LOG_ERROR(<< "Invalid data in " << traverser.value());
                        return false;
                    }
                    result[model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID].emplace(pid, initial);
                    pid++;
                }
            } while (traverser.next());

            return true;
        }

        template<typename T>
        bool restorePopulation(core::CStateRestoreTraverser& traverser,
                               const CMetricBucketGatherer& gatherer,
                               TSizeSizeTUMapUMap<T>& result) const {
            std::size_t pid;

            std::size_t lastCid(0);
            bool seenCid(false);

            do {
                const std::string& name = traverser.name();
                if (name == ATTRIBUTE_TAG) {
                    if (core::CStringUtils::stringToType(traverser.value(), lastCid) == false) {
                        LOG_ERROR(<< "Invalid attribute ID in " << traverser.value());
                        return false;
                    }
                    seenCid = true;
                } else if (name == DATA_TAG) {
                    if (!seenCid) {
                        LOG_ERROR(<< "Incorrect format - data before attribute ID in "
                                  << traverser.value());
                        return false;
                    }

                    T initial(gatherer.dataGatherer().params(), m_Dimension,
                              gatherer.currentBucketStartTime(),
                              gatherer.bucketLength(), gatherer.beginInfluencers(),
                              gatherer.endInfluencers());
                    if (traverser.traverseSubLevel(
                            std::bind(&T::acceptRestoreTraverser, &initial,
                                      std::placeholders::_1)) == false) {
                        LOG_ERROR(<< "Invalid data in " << traverser.value());
                        return false;
                    }

                    auto& pidMap = result[lastCid];
                    pid = pidMap.size();
                    pidMap.emplace(pid, initial);
                }
            } while (traverser.next());

            return true;
        }

    private:
        std::size_t m_Dimension;
    };
};

//! Removes the people from the data gatherers.
struct SRemovePeople {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    std::size_t begin,
                    std::size_t end) const {
        for (auto& cidEntry : data) {
            for (std::size_t pid = begin; pid < end; ++pid) {
                cidEntry.second.erase(pid);
            }
        }
    }

    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    const TSizeVec& peopleToRemove) const {
        for (auto& cidEntry : data) {
            for (auto pid : peopleToRemove) {
                cidEntry.second.erase(pid);
            }
        }
    }
};

//! Removes attributes from the data gatherers.
struct SRemoveAttributes {
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    const TSizeVec& attributesToRemove) const {
        for (auto cid : attributesToRemove) {
            data.erase(cid);
        }
    }

    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    std::size_t begin,
                    std::size_t end) const {
        for (std::size_t cid = begin; cid < end; ++cid) {
            data.erase(cid);
        }
    }
};

//! Sample the metric statistics.
struct SDoSample {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    core_t::TTime time,
                    const CMetricBucketGatherer& gatherer,
                    CSampleCounts& sampleCounts) const {
        for (const auto& count : gatherer.bucketCounts(time)) {
            std::size_t pid = CDataGatherer::extractPersonId(count);
            std::size_t cid = CDataGatherer::extractAttributeId(count);
            std::size_t activeId = gatherer.dataGatherer().isPopulation() ? cid : pid;
            auto cidEntry = data.find(cid);
            if (cidEntry == data.end()) {
                LOG_ERROR(<< "No gatherer for attribute "
                          << gatherer.dataGatherer().attributeName(cid) << " of person "
                          << gatherer.dataGatherer().personName(pid));
            } else {
                auto pidEntry = cidEntry->second.find(pid);
                if (pidEntry == cidEntry->second.end()) {
                    LOG_ERROR(<< "No gatherer for attribute "
                              << gatherer.dataGatherer().attributeName(cid) << " of person "
                              << gatherer.dataGatherer().personName(pid));
                } else if (pidEntry->second.sample(time, sampleCounts.count(activeId))) {
                    sampleCounts.updateSampleVariance(activeId);
                }
            }
        }
    }
};

//! Stably hashes the collection of data gatherers.
struct SHash {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    const TSizeSizeTUMapUMap<T>& data,
                    const CMetricBucketGatherer& gatherer,
                    TStrCRefStrCRefPrUInt64Map& hashes) const {
        for (const auto& cidEntry : data) {
            std::size_t cid = cidEntry.first;
            if (gatherer.dataGatherer().isAttributeActive(cid)) {
                TStrCRef cidName = TStrCRef(gatherer.dataGatherer().attributeName(cid));
                for (const auto& pidEntry : cidEntry.second) {
                    std::size_t pid = pidEntry.first;
                    if (gatherer.dataGatherer().isPersonActive(pid)) {
                        TStrCRef pidName =
                            TStrCRef(gatherer.dataGatherer().personName(pid));
                        hashes.emplace(std::piecewise_construct,
                                       std::forward_as_tuple(cidName, pidName),
                                       std::forward_as_tuple(pidEntry.second.checksum()));
                    }
                }
            }
        }
    }
};

//! Extracts feature data from a collection of gatherers.
struct SExtractFeatureData {
public:
    using TFeatureAnyPr = std::pair<model_t::EFeature, boost::any>;
    using TFeatureAnyPrVec = std::vector<TFeatureAnyPr>;

public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    const TSizeSizeTUMapUMap<T>& data,
                    const CMetricBucketGatherer& gatherer,
                    model_t::EFeature feature,
                    core_t::TTime time,
                    core_t::TTime bucketLength,
                    TFeatureAnyPrVec& result) const {
        if (gatherer.dataGatherer().isPopulation()) {
            result.emplace_back(feature, TSizeSizePrFeatureDataPrVec());
            this->featureData(data, gatherer, time, bucketLength, this->isSum(feature),
                              *boost::unsafe_any_cast<TSizeSizePrFeatureDataPrVec>(
                                  &result.back().second));
        } else {
            result.emplace_back(feature, TSizeFeatureDataPrVec());
            this->featureData(
                data, gatherer, time, bucketLength, this->isSum(feature),
                *boost::unsafe_any_cast<TSizeFeatureDataPrVec>(&result.back().second));
        }
    }

private:
    static const TSampleVec ZERO_SAMPLE;

private:
    bool isSum(model_t::EFeature feature) const {
        return feature == model_t::E_IndividualSumByBucketAndPerson ||
               feature == model_t::E_IndividualLowSumByBucketAndPerson ||
               feature == model_t::E_IndividualHighSumByBucketAndPerson;
    }

    template<typename T, typename U>
    void featureData(const TSizeSizeTUMapUMap<T>& data,
                     const CMetricBucketGatherer& gatherer,
                     core_t::TTime time,
                     core_t::TTime bucketLength,
                     bool isSum,
                     U& result) const {
        result.clear();
        if (isSum) {
            if (data.empty() == false) {
                auto& pidMap = data.begin()->second;
                result.reserve(pidMap.size());
                for (auto& pidEntry : pidMap) {
                    std::size_t pid = pidEntry.first;
                    if (gatherer.hasExplicitNullsOnly(
                            time, pid, model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID) == false) {
                        this->featureData(pidEntry.second, gatherer, pid,
                                          model_t::INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID,
                                          time, bucketLength, result);
                    }
                }
            }
        } else {
            const TSizeSizePrUInt64UMap& counts = gatherer.bucketCounts(time);
            result.reserve(counts.size());
            for (const auto& count : counts) {
                std::size_t cid = CDataGatherer::extractAttributeId(count);
                auto cidEntry = data.find(cid);
                if (cidEntry == data.end()) {
                    LOG_ERROR(<< "No gatherers for attribute "
                              << gatherer.dataGatherer().attributeName(cid));
                    continue;
                }
                std::size_t pid = CDataGatherer::extractPersonId(count);
                auto pidEntry = cidEntry->second.find(pid);
                if (pidEntry == cidEntry->second.end()) {
                    LOG_ERROR(<< "No gatherers for person "
                              << gatherer.dataGatherer().personName(pid));
                    continue;
                }

                this->featureData(pidEntry->second, gatherer, pid, cid, time,
                                  bucketLength, result);
            }
        }
        std::sort(result.begin(), result.end(), maths::COrderings::SFirstLess());
    }

    //! Individual model specialization
    template<typename T>
    void featureData(const T& data,
                     const CMetricBucketGatherer& gatherer,
                     std::size_t pid,
                     std::size_t /*cid*/,
                     core_t::TTime time,
                     core_t::TTime bucketLength,
                     TSizeFeatureDataPrVec& result) const {
        result.emplace_back(
            pid, this->featureData(data, time, bucketLength,
                                   gatherer.dataGatherer().effectiveSampleCount(pid)));
    }

    //! Population model specialization
    template<typename T>
    void featureData(const T& data,
                     const CMetricBucketGatherer& gatherer,
                     std::size_t pid,
                     std::size_t cid,
                     core_t::TTime time,
                     core_t::TTime bucketLength,
                     TSizeSizePrFeatureDataPrVec& result) const {
        result.emplace_back(
            TSizeSizePr(pid, cid),
            this->featureData(data, time, bucketLength,
                              gatherer.dataGatherer().effectiveSampleCount(cid)));
    }

    SMetricFeatureData featureData(const CGathererTools::CSumGatherer& data,
                                   core_t::TTime time,
                                   core_t::TTime bucketLength,
                                   double /*effectiveSampleCount*/) const {
        return data.featureData(time, bucketLength, ZERO_SAMPLE);
    }

    template<typename T>
    inline SMetricFeatureData featureData(const T& data,
                                          core_t::TTime time,
                                          core_t::TTime bucketLength,
                                          double effectiveSampleCount) const {
        return data.featureData(time, bucketLength, effectiveSampleCount);
    }
};

const TSampleVec
    SExtractFeatureData::ZERO_SAMPLE(1, CSample(0, TDoubleVec(1, 0.0), 1.0, 1.0));

//! Adds a value to the specified data gatherers.
struct SAddValue {
    struct SStatistic {
        core_t::TTime s_Time;
        const CEventData::TDouble1VecArray* s_Values;
        unsigned int s_Count;
        unsigned int s_SampleCount;
        const TStoredStringPtrVec* s_Influences;
    };

    template<typename T>
    inline void operator()(const TCategorySizePr& category,
                           TSizeSizeTUMapUMap<T>& data,
                           std::size_t pid,
                           std::size_t cid,
                           const CMetricBucketGatherer& gatherer,
                           const SStatistic& stat) const {
        auto& entry =
            data[cid]
                .emplace(boost::unordered::piecewise_construct, boost::make_tuple(pid),
                         boost::make_tuple(
                             std::cref(gatherer.dataGatherer().params()),
                             category.second, gatherer.currentBucketStartTime(),
                             gatherer.bucketLength(), gatherer.beginInfluencers(),
                             gatherer.endInfluencers()))
                .first->second;
        entry.add(stat.s_Time, (*stat.s_Values)[category.first], stat.s_Count,
                  stat.s_SampleCount, *stat.s_Influences);
    }
};

//! Updates gatherers with the start of a new bucket.
struct SStartNewBucket {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    core_t::TTime time) const {
        for (auto& cidEntry : data) {
            for (auto& pidEntry : cidEntry.second) {
                pidEntry.second.startNewBucket(time);
            }
        }
    }
};

//! Resets data stored for buckets containing a specified time.
struct SResetBucket {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    core_t::TTime bucketStart) const {
        for (auto& cidEntry : data) {
            for (auto& pidEntry : cidEntry.second) {
                pidEntry.second.resetBucket(bucketStart);
            }
        }
    }
};

//! Releases memory that is no longer needed.
struct SReleaseMemory {
public:
    template<typename T>
    void operator()(const TCategorySizePr& /*category*/,
                    TSizeSizeTUMapUMap<T>& data,
                    core_t::TTime samplingCutoffTime) const {
        for (auto& cidEntry : data) {
            auto& pidMap = cidEntry.second;
            for (auto i = pidMap.begin(); i != pidMap.end(); /**/) {
                if (i->second.isRedundant(samplingCutoffTime)) {
                    i = pidMap.erase(i);
                } else {
                    ++i;
                }
            }
        }
    }
};

} // unnamed::

CMetricBucketGatherer::CMetricBucketGatherer(CDataGatherer& dataGatherer,
                                             const std::string& summaryCountFieldName,
                                             const std::string& personFieldName,
                                             const std::string& attributeFieldName,
                                             const std::string& valueFieldName,
                                             const TStrVec& influenceFieldNames,
                                             core_t::TTime startTime)
    : CBucketGatherer(dataGatherer, startTime, influenceFieldNames.size()),
      m_ValueFieldName(valueFieldName), m_BeginInfluencingFields(0),
      m_BeginValueFields(0) {
    this->initializeFieldNamesPart1(personFieldName, attributeFieldName, influenceFieldNames);
    this->initializeFieldNamesPart2(valueFieldName, summaryCountFieldName);
    this->initializeFeatureData();
}

CMetricBucketGatherer::CMetricBucketGatherer(CDataGatherer& dataGatherer,
                                             const std::string& summaryCountFieldName,
                                             const std::string& personFieldName,
                                             const std::string& attributeFieldName,
                                             const std::string& valueFieldName,
                                             const TStrVec& influenceFieldNames,
                                             core::CStateRestoreTraverser& traverser)
    : CBucketGatherer(dataGatherer, 0, influenceFieldNames.size()),
      m_ValueFieldName(valueFieldName), m_BeginValueFields(0) {
    this->initializeFieldNamesPart1(personFieldName, attributeFieldName, influenceFieldNames);
    traverser.traverseSubLevel(std::bind(&CMetricBucketGatherer::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
    this->initializeFieldNamesPart2(valueFieldName, summaryCountFieldName);
}

CMetricBucketGatherer::CMetricBucketGatherer(bool isForPersistence,
                                             const CMetricBucketGatherer& other)
    : CBucketGatherer(isForPersistence, other),
      m_ValueFieldName(other.m_ValueFieldName),
      m_FieldNames(other.m_FieldNames), m_BeginInfluencingFields(0),
      m_BeginValueFields(0), m_FeatureData(other.m_FeatureData) {
    if (!isForPersistence) {
        LOG_ABORT(<< "This constructor only creates clones for persistence");
    }
}

void CMetricBucketGatherer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(BASE_TAG, std::bind(&CBucketGatherer::baseAcceptPersistInserter,
                                             this, std::placeholders::_1));
    inserter.insertValue(VERSION_TAG, CURRENT_VERSION);
    applyFunc(m_FeatureData,
              std::bind<void>(CPersistFeatureData(), std::placeholders::_1,
                              std::placeholders::_2, std::ref(inserter)));
}

bool CMetricBucketGatherer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::string version;
    bool isCurrentVersion(false);

    do {
        const std::string& name = traverser.name();
        if (name == BASE_TAG) {
            if (traverser.traverseSubLevel(std::bind(&CBucketGatherer::baseAcceptRestoreTraverser,
                                                     this, std::placeholders::_1)) == false) {
                LOG_ERROR(<< "Invalid data gatherer in " << traverser.value());
                return false;
            }
        } else if (name == VERSION_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), version) == false) {
                LOG_ERROR(<< "Invalid version in " << traverser.value());
                return false;
            }
            isCurrentVersion = (version == CURRENT_VERSION);
        } else if (this->acceptRestoreTraverserInternal(traverser, isCurrentVersion) == false) {
            // Soldier on or we'll get a core dump later.
        }
    } while (traverser.next());

    return true;
}

bool CMetricBucketGatherer::acceptRestoreTraverserInternal(core::CStateRestoreTraverser& traverser,
                                                           bool isCurrentVersion) {
    const std::string& name = traverser.name();
    if (name == MEAN_TAG) {
        CRestoreFeatureData<model_t::E_Mean> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid mean data in " << traverser.value());
            return false;
        }
    } else if (name == MIN_TAG) {
        CRestoreFeatureData<model_t::E_Min> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid min data in " << traverser.value());
            return false;
        }
    } else if (name == MAX_TAG) {
        CRestoreFeatureData<model_t::E_Max> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid max data in " << traverser.value());
            return false;
        }
    } else if (name == SUM_TAG) {
        CRestoreFeatureData<model_t::E_Sum> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid sum data in " << traverser.value());
            return false;
        }
    } else if (name == MEDIAN_TAG) {
        CRestoreFeatureData<model_t::E_Median> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid median data in " << traverser.value());
            return false;
        }
    } else if (name == VARIANCE_TAG) {
        CRestoreFeatureData<model_t::E_Variance> restore;
        if (restore(traverser, 1, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid variance data in " << traverser.value());
            return false;
        }
    } else if (name.find(MULTIVARIATE_MEAN_TAG) != std::string::npos) {
        std::size_t dimension;
        if (core::CStringUtils::stringToType(
                name.substr(MULTIVARIATE_MEAN_TAG.length()), dimension) == false) {
            LOG_ERROR(<< "Invalid dimension in " << name);
            return false;
        }
        CRestoreFeatureData<model_t::E_MultivariateMean> restore;
        if (restore(traverser, dimension, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid multivariate mean data in " << traverser.value());
            return false;
        }
    } else if (name.find(MULTIVARIATE_MIN_TAG) != std::string::npos) {
        std::size_t dimension;
        if (core::CStringUtils::stringToType(
                name.substr(MULTIVARIATE_MIN_TAG.length()), dimension) == false) {
            LOG_ERROR(<< "Invalid dimension in " << name);
            return false;
        }
        CRestoreFeatureData<model_t::E_MultivariateMin> restore;
        if (restore(traverser, dimension, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid multivariate min data in " << traverser.value());
            return false;
        }
    } else if (name.find(MULTIVARIATE_MAX_TAG) != std::string::npos) {
        std::size_t dimension;
        if (core::CStringUtils::stringToType(
                name.substr(MULTIVARIATE_MAX_TAG.length()), dimension) == false) {
            LOG_ERROR(<< "Invalid dimension in " << name);
            return false;
        }
        CRestoreFeatureData<model_t::E_MultivariateMax> restore;
        if (restore(traverser, dimension, isCurrentVersion, *this, m_FeatureData) == false) {
            LOG_ERROR(<< "Invalid multivariate max data in " << traverser.value());
            return false;
        }
    }

    return true;
}

CBucketGatherer* CMetricBucketGatherer::cloneForPersistence() const {
    return new CMetricBucketGatherer(true, *this);
}

const std::string& CMetricBucketGatherer::persistenceTag() const {
    return CBucketGatherer::METRIC_BUCKET_GATHERER_TAG;
}

const std::string& CMetricBucketGatherer::personFieldName() const {
    return m_FieldNames[0];
}

const std::string& CMetricBucketGatherer::attributeFieldName() const {
    return m_DataGatherer.isPopulation() ? m_FieldNames[1] : EMPTY_STRING;
}

const std::string& CMetricBucketGatherer::valueFieldName() const {
    return m_ValueFieldName;
}

CMetricBucketGatherer::TStrVecCItr CMetricBucketGatherer::beginInfluencers() const {
    return m_FieldNames.begin() + m_BeginInfluencingFields;
}

CMetricBucketGatherer::TStrVecCItr CMetricBucketGatherer::endInfluencers() const {
    return m_FieldNames.begin() + m_BeginValueFields;
}

const TStrVec& CMetricBucketGatherer::fieldsOfInterest() const {
    return m_FieldNames;
}

std::string CMetricBucketGatherer::description() const {
    return function_t::name(function_t::function(m_DataGatherer.features())) +
           (m_ValueFieldName.empty() ? "" : " ") + m_ValueFieldName +
           +(byField(m_DataGatherer.isPopulation(), m_FieldNames).empty() ? "" : " by ") +
           byField(m_DataGatherer.isPopulation(), m_FieldNames) +
           (overField(m_DataGatherer.isPopulation(), m_FieldNames).empty() ? "" : " over ") +
           overField(m_DataGatherer.isPopulation(), m_FieldNames) +
           (m_DataGatherer.partitionFieldName().empty() ? "" : " partition=") +
           m_DataGatherer.partitionFieldName();
}

bool CMetricBucketGatherer::processFields(const TStrCPtrVec& fieldValues,
                                          CEventData& result,
                                          CResourceMonitor& resourceMonitor) {
    using TOptionalStr = boost::optional<std::string>;

    if (fieldValues.size() != m_FieldNames.size()) {
        LOG_ERROR(<< "Unexpected field values: " << core::CContainerPrinter::print(fieldValues)
                  << ", for field names: " << core::CContainerPrinter::print(m_FieldNames));
        return false;
    }

    const std::string* person = (fieldValues[0] == nullptr && m_DataGatherer.useNull())
                                    ? &EMPTY_STRING
                                    : fieldValues[0];
    if (person == nullptr) {
        // Just ignore: the "person" field wasn't present in the
        // record. Since all models in an aggregate share this
        // field we can't process this record further. Note that
        // we don't warn here since we'll permit a small fraction
        // of records to having missing field values.
        return false;
    }

    // The code below just ignores missing/invalid values. This
    // doesn't necessarily stop us processing the record by other
    // models so we don't return false.

    std::size_t i = m_BeginInfluencingFields;
    for (/**/; i < m_BeginValueFields; ++i) {
        result.addInfluence(fieldValues[i] ? TOptionalStr(*fieldValues[i]) : TOptionalStr());
    }
    if (m_DataGatherer.summaryMode() != model_t::E_None) {
        CEventData::TDouble1VecArraySizePr statistics;
        statistics.first.fill(TDouble1Vec(1, 0.0));
        if (m_DataGatherer.extractCountFromField(m_FieldNames[i], fieldValues[i],
                                                 statistics.second) == false) {
            result.addValue();
            return true;
        }
        ++i;

        bool allOk = true;
        if (m_FieldNames.size() > statistics.first.size() + i) {
            LOG_ERROR(<< "Inconsistency - more statistic field names than allowed "
                      << m_FieldNames.size() - i << " > " << statistics.first.size());
            allOk = false;
        }
        if (m_FieldNames.size() > m_FieldMetricCategories.size() + i) {
            LOG_ERROR(<< "Inconsistency - more statistic field names than metric categories "
                      << m_FieldNames.size() - i << " > "
                      << m_FieldMetricCategories.size());
            allOk = false;
        }
        for (std::size_t j = 0u; allOk && i < m_FieldNames.size(); ++i, ++j) {
            model_t::EMetricCategory category = m_FieldMetricCategories[j];
            if (fieldValues[i] == nullptr ||
                m_DataGatherer.extractMetricFromField(
                    m_FieldNames[i], *fieldValues[i], statistics.first[category]) == false) {
                allOk = false;
            }
        }
        if (allOk) {
            if (statistics.second == CDataGatherer::EXPLICIT_NULL_SUMMARY_COUNT) {
                result.setExplicitNull();
            } else {
                result.addStatistics(statistics);
            }
        } else {
            result.addValue();
        }
    } else {
        TDouble1Vec value;
        if (fieldValues[i] != nullptr &&
            m_DataGatherer.extractMetricFromField(m_FieldNames[i], *fieldValues[i],
                                                  value) == true) {
            result.addValue(value);
        } else {
            result.addValue();
        }
    }

    bool addedPerson = false;
    std::size_t pid = CDynamicStringIdRegistry::INVALID_ID;
    if (result.isExplicitNull()) {
        m_DataGatherer.personId(*person, pid);
    } else {
        pid = m_DataGatherer.addPerson(*person, resourceMonitor, addedPerson);
    }

    if (pid == CDynamicStringIdRegistry::INVALID_ID) {
        if (!result.isExplicitNull()) {
            LOG_TRACE(<< "Couldn't create a person, over memory limit");
        }
        return false;
    }
    if (addedPerson) {
        resourceMonitor.addExtraMemory(m_DataGatherer.isPopulation()
                                           ? CDataGatherer::ESTIMATED_MEM_USAGE_PER_OVER_FIELD
                                           : CDataGatherer::ESTIMATED_MEM_USAGE_PER_BY_FIELD);
        ++(m_DataGatherer.isPopulation()
               ? core::CProgramCounters::counter(counter_t::E_TSADNumberOverFields)
               : core::CProgramCounters::counter(counter_t::E_TSADNumberByFields));
    }

    if (!result.person(pid)) {
        LOG_ERROR(<< "Bad by field value: " << *person);
        return false;
    }

    const std::string* attribute =
        (fieldValues[1] == nullptr && m_DataGatherer.useNull()) ? &EMPTY_STRING
                                                                : fieldValues[1];

    if (m_DataGatherer.isPopulation()) {
        if (attribute == nullptr) {
            // Just ignore: the "by" field wasn't present in the
            // record. This doesn't necessarily stop us processing
            // the record by other models so we don't return false.
            // Note that we don't warn here since we'll permit a
            // small fraction of records to having missing field
            // values.
            result.addAttribute();
            result.addValue();
            return true;
        }

        bool addedAttribute = false;
        std::size_t cid = CDynamicStringIdRegistry::INVALID_ID;
        if (result.isExplicitNull()) {
            m_DataGatherer.attributeId(*attribute, cid);
        } else {
            cid = m_DataGatherer.addAttribute(*attribute, resourceMonitor, addedAttribute);
        }
        result.addAttribute(cid);

        if (addedAttribute) {
            resourceMonitor.addExtraMemory(CDataGatherer::ESTIMATED_MEM_USAGE_PER_BY_FIELD);
            ++core::CProgramCounters::counter(counter_t::E_TSADNumberByFields);
        }
    } else {
        // Add the unique attribute.
        result.addAttribute(std::size_t(0));
    }

    return true;
}

void CMetricBucketGatherer::recyclePeople(const TSizeVec& peopleToRemove) {
    if (peopleToRemove.empty()) {
        return;
    }

    applyFunc(m_FeatureData,
              std::bind<void>(SRemovePeople(), std::placeholders::_1,
                              std::placeholders::_2, std::cref(peopleToRemove)));

    this->CBucketGatherer::recyclePeople(peopleToRemove);
}

void CMetricBucketGatherer::removePeople(std::size_t lowestPersonToRemove) {
    applyFunc(m_FeatureData, std::bind<void>(SRemovePeople(), std::placeholders::_1,
                                             std::placeholders::_2, lowestPersonToRemove,
                                             m_DataGatherer.numberPeople()));

    this->CBucketGatherer::removePeople(lowestPersonToRemove);
}

void CMetricBucketGatherer::recycleAttributes(const TSizeVec& attributesToRemove) {
    if (attributesToRemove.empty()) {
        return;
    }

    if (m_DataGatherer.isPopulation()) {
        applyFunc(m_FeatureData,
                  std::bind<void>(SRemoveAttributes(), std::placeholders::_1,
                                  std::placeholders::_2, std::cref(attributesToRemove)));
    }

    this->CBucketGatherer::recycleAttributes(attributesToRemove);
}

void CMetricBucketGatherer::removeAttributes(std::size_t lowestAttributeToRemove) {
    if (m_DataGatherer.isPopulation()) {
        applyFunc(m_FeatureData,
                  std::bind<void>(SRemoveAttributes(), std::placeholders::_1,
                                  std::placeholders::_2, lowestAttributeToRemove,
                                  m_DataGatherer.numberAttributes()));
    }

    this->CBucketGatherer::removeAttributes(lowestAttributeToRemove);
}

uint64_t CMetricBucketGatherer::checksum() const {
    uint64_t seed = this->CBucketGatherer::checksum();
    seed = maths::CChecksum::calculate(seed, m_DataGatherer.params().s_DecayRate);
    TStrCRefStrCRefPrUInt64Map hashes;
    applyFunc(m_FeatureData, std::bind<void>(SHash(), std::placeholders::_1,
                                             std::placeholders::_2,
                                             std::cref(*this), std::ref(hashes)));
    LOG_TRACE(<< "seed = " << seed);
    LOG_TRACE(<< "hashes = " << core::CContainerPrinter::print(hashes));
    return maths::CChecksum::calculate(seed, hashes);
}

void CMetricBucketGatherer::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    registerMemoryCallbacks();
    mem->setName("CMetricBucketGatherer");
    this->CBucketGatherer::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_ValueFieldName", m_ValueFieldName, mem);
    core::CMemoryDebug::dynamicSize("m_FieldNames", m_FieldNames, mem);
    core::CMemoryDebug::dynamicSize("m_FieldMetricCategories", m_FieldMetricCategories, mem);
    core::CMemoryDebug::dynamicSize("m_FeatureData", m_FeatureData, mem);
}

std::size_t CMetricBucketGatherer::memoryUsage() const {
    registerMemoryCallbacks();
    std::size_t mem = this->CBucketGatherer::memoryUsage();
    mem += core::CMemory::dynamicSize(m_ValueFieldName);
    mem += core::CMemory::dynamicSize(m_FieldNames);
    mem += core::CMemory::dynamicSize(m_FieldMetricCategories);
    mem += core::CMemory::dynamicSize(m_FeatureData);
    return mem;
}

std::size_t CMetricBucketGatherer::staticSize() const {
    return sizeof(*this);
}

void CMetricBucketGatherer::clear() {
    this->CBucketGatherer::clear();
    m_FeatureData.clear();
    this->initializeFeatureData();
}

bool CMetricBucketGatherer::resetBucket(core_t::TTime bucketStart) {
    if (this->CBucketGatherer::resetBucket(bucketStart) == false) {
        return false;
    }
    applyFunc(m_FeatureData, std::bind<void>(SResetBucket(), std::placeholders::_1,
                                             std::placeholders::_2, bucketStart));
    return true;
}

void CMetricBucketGatherer::releaseMemory(core_t::TTime samplingCutoffTime) {
    applyFunc(m_FeatureData, std::bind<void>(SReleaseMemory(), std::placeholders::_1,
                                             std::placeholders::_2, samplingCutoffTime));
}

void CMetricBucketGatherer::sample(core_t::TTime time) {
    if (m_DataGatherer.sampleCounts()) {
        applyFunc(m_FeatureData,
                  std::bind<void>(SDoSample(), std::placeholders::_1,
                                  std::placeholders::_2, time, std::cref(*this),
                                  std::ref(*m_DataGatherer.sampleCounts())));
    }
}

void CMetricBucketGatherer::featureData(core_t::TTime time,
                                        core_t::TTime bucketLength,
                                        TFeatureAnyPrVec& result) const {
    result.clear();

    if (!this->dataAvailable(time) ||
        time >= this->currentBucketStartTime() + this->bucketLength()) {
        LOG_DEBUG(<< "No data available at " << time);
        return;
    }

    for (std::size_t i = 0u, n = m_DataGatherer.numberFeatures(); i < n; ++i) {
        model_t::EFeature feature = m_DataGatherer.feature(i);
        model_t::EMetricCategory category;
        if (model_t::metricCategory(feature, category)) {
            std::size_t dimension = model_t::dimension(feature);
            auto begin = m_FeatureData.find({category, dimension});
            if (begin != m_FeatureData.end()) {
                auto end = begin;
                ++end;
                applyFunc(begin, end,
                          std::bind<void>(SExtractFeatureData(), std::placeholders::_1,
                                          std::placeholders::_2, std::cref(*this), feature,
                                          time, bucketLength, std::ref(result)));
            } else {
                LOG_ERROR(<< "No data for category " << model_t::print(category));
            }
        } else {
            LOG_ERROR(<< "Unexpected feature " << model_t::print(feature));
        }
    }
}

void CMetricBucketGatherer::resize(std::size_t pid, std::size_t cid) {
    if (m_DataGatherer.sampleCounts()) {
        m_DataGatherer.sampleCounts()->resize(m_DataGatherer.isPopulation() ? cid : pid);
    } else {
        LOG_ERROR(<< "Invalid sample counts for gatherer");
    }
}

void CMetricBucketGatherer::addValue(std::size_t pid,
                                     std::size_t cid,
                                     core_t::TTime time,
                                     const CEventData::TDouble1VecArray& values,
                                     std::size_t count,
                                     const CEventData::TOptionalStr& /*stringValue*/,
                                     const TStoredStringPtrVec& influences) {
    // Check that we are correctly sized - a person/attribute might have been added
    this->resize(pid, cid);

    SAddValue::SStatistic stat;
    stat.s_Time = time;
    stat.s_Values = &values;
    stat.s_Count = static_cast<unsigned int>(count);
    if (m_DataGatherer.sampleCounts()) {
        stat.s_SampleCount = m_DataGatherer.sampleCounts()->count(
            m_DataGatherer.isPopulation() ? cid : pid);
    } else {
        LOG_ERROR(<< "Invalid sample counts for gatherer");
        stat.s_SampleCount = 0.0;
    }

    stat.s_Influences = &influences;
    applyFunc(m_FeatureData, std::bind<void>(SAddValue(), std::placeholders::_1,
                                             std::placeholders::_2, pid, cid,
                                             std::cref(*this), std::ref(stat)));
}

void CMetricBucketGatherer::startNewBucket(core_t::TTime time, bool skipUpdates) {
    LOG_TRACE(<< "StartNewBucket, " << time << " @ " << this);
    using TUInt64Vec = std::vector<uint64_t>;
    using TSizeUInt64VecUMap = boost::unordered_map<std::size_t, TUInt64Vec>;

    // Only update the sampleCounts if we are the primary bucket gatherer.
    // This is the only place where the bucket gatherer needs to know about its
    // status within the celestial plain, which is a bit ugly...
    if (!skipUpdates && time % this->bucketLength() == 0) {
        core_t::TTime earliestAvailableBucketStartTime = this->earliestBucketStartTime();
        if (this->dataAvailable(earliestAvailableBucketStartTime)) {
            TSizeUInt64VecUMap counts;
            const TSizeSizePrUInt64UMap& counts_ =
                this->bucketCounts(earliestAvailableBucketStartTime);
            for (const auto& count : counts_) {
                if (m_DataGatherer.isPopulation()) {
                    counts[CDataGatherer::extractAttributeId(count)].push_back(
                        CDataGatherer::extractData(count));
                } else {
                    counts
                        .emplace(CDataGatherer::extractPersonId(count), TUInt64Vec{0})
                        .first->second[0] += CDataGatherer::extractData(count);
                }
            }
            double alpha = std::exp(-m_DataGatherer.params().s_DecayRate);

            for (auto& count : counts) {
                std::sort(count.second.begin(), count.second.end());
                std::size_t n = count.second.size() / 2;
                double median =
                    count.second.size() % 2 == 0
                        ? static_cast<double>(count.second[n - 1] + count.second[n]) / 2.0
                        : static_cast<double>(count.second[n]);
                m_DataGatherer.sampleCounts()->updateMeanNonZeroBucketCount(
                    count.first, median, alpha);
            }
            m_DataGatherer.sampleCounts()->refresh(m_DataGatherer);
        }
    }
    applyFunc(m_FeatureData, std::bind<void>(SStartNewBucket(), std::placeholders::_1,
                                             std::placeholders::_2, time));
}

void CMetricBucketGatherer::initializeFieldNamesPart1(const std::string& personFieldName,
                                                      const std::string& attributeFieldName,
                                                      const TStrVec& influenceFieldNames) {
    switch (m_DataGatherer.summaryMode()) {
    case model_t::E_None:
        m_FieldNames.reserve(2 + static_cast<std::size_t>(m_DataGatherer.isPopulation()) +
                             influenceFieldNames.size());
        m_FieldNames.push_back(personFieldName);
        if (m_DataGatherer.isPopulation())
            m_FieldNames.push_back(attributeFieldName);
        m_BeginInfluencingFields = m_FieldNames.size();
        m_FieldNames.insert(m_FieldNames.end(), influenceFieldNames.begin(),
                            influenceFieldNames.end());
        m_BeginValueFields = m_FieldNames.size();
        break;
    case model_t::E_Manual:
        m_FieldNames.reserve(3 + static_cast<std::size_t>(m_DataGatherer.isPopulation()) +
                             influenceFieldNames.size());
        m_FieldNames.push_back(personFieldName);
        if (m_DataGatherer.isPopulation())
            m_FieldNames.push_back(attributeFieldName);
        m_BeginInfluencingFields = m_FieldNames.size();
        m_FieldNames.insert(m_FieldNames.end(), influenceFieldNames.begin(),
                            influenceFieldNames.end());
        m_BeginValueFields = m_FieldNames.size();
        break;
    };
}

void CMetricBucketGatherer::initializeFieldNamesPart2(const std::string& valueFieldName,
                                                      const std::string& summaryCountFieldName) {
    switch (m_DataGatherer.summaryMode()) {
    case model_t::E_None:
        m_FieldNames.push_back(valueFieldName);
        break;
    case model_t::E_Manual:
        m_FieldNames.push_back(summaryCountFieldName);
        m_FieldNames.push_back(valueFieldName);
        m_DataGatherer.determineMetricCategory(m_FieldMetricCategories);
        break;
    };
}

void CMetricBucketGatherer::initializeFeatureData() {
    for (std::size_t i = 0u, n = m_DataGatherer.numberFeatures(); i < n; ++i) {
        const model_t::EFeature feature = m_DataGatherer.feature(i);
        model_t::EMetricCategory category;
        if (model_t::metricCategory(feature, category)) {
            std::size_t dimension = model_t::dimension(feature);
            switch (category) {
            case model_t::E_Mean:
                initializeFeatureDataInstance<model_t::E_Mean>(dimension, m_FeatureData);
                break;
            case model_t::E_Median:
                initializeFeatureDataInstance<model_t::E_Median>(dimension, m_FeatureData);
                break;
            case model_t::E_Min:
                initializeFeatureDataInstance<model_t::E_Min>(dimension, m_FeatureData);
                break;
            case model_t::E_Max:
                initializeFeatureDataInstance<model_t::E_Max>(dimension, m_FeatureData);
                break;
            case model_t::E_Variance:
                initializeFeatureDataInstance<model_t::E_Variance>(dimension, m_FeatureData);
                break;
            case model_t::E_Sum:
                initializeFeatureDataInstance<model_t::E_Sum>(dimension, m_FeatureData);
                break;
            case model_t::E_MultivariateMean:
                initializeFeatureDataInstance<model_t::E_MultivariateMean>(dimension, m_FeatureData);
                break;
            case model_t::E_MultivariateMin:
                initializeFeatureDataInstance<model_t::E_MultivariateMin>(dimension, m_FeatureData);
                break;
            case model_t::E_MultivariateMax:
                initializeFeatureDataInstance<model_t::E_MultivariateMax>(dimension, m_FeatureData);
                break;
            }
        } else {
            LOG_ERROR(<< "Unexpected feature = "
                      << model_t::print(m_DataGatherer.feature(i)));
        }
    }
}
}
}
