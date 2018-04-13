/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CDataGatherer.h>

#include <core/CContainerPrinter.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStatistics.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CMathsFuncs.h>

#include <model/CEventRateBucketGatherer.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CResourceMonitor.h>
#include <model/CSampleCounts.h>
#include <model/CStringStore.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>

namespace ml {
namespace model {

namespace {

const std::string FEATURE_TAG("a");
const std::string PEOPLE_REGISTRY_TAG("b");
const std::string ATTRIBUTES_REGISTRY_TAG("c");
const std::string SAMPLE_COUNTS_TAG("d");
const std::string BUCKET_GATHERER_TAG("e");
const std::string DEFAULT_PERSON_NAME("-");
const std::string DEFAULT_ATTRIBUTE_NAME("-");

const std::string PERSON("person");
const std::string ATTRIBUTE("attribute");

const std::string EMPTY_STRING;

namespace detail {

//! Make sure \p features only includes supported features, doesn't
//! contain any duplicates, etc.
const CDataGatherer::TFeatureVec& sanitize(CDataGatherer::TFeatureVec& features, model_t::EAnalysisCategory gathererType) {
    std::size_t j = 0u;

    for (std::size_t i = 0u; i < features.size(); ++i) {
        switch (gathererType) {
        case model_t::E_EventRate:
        case model_t::E_PopulationEventRate:
        case model_t::E_PeersEventRate:
            switch (features[i]) {
            CASE_INDIVIDUAL_COUNT:
                features[j] = features[i];
                ++j;
                break;

            CASE_INDIVIDUAL_METRIC:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]));
                break;

            CASE_POPULATION_COUNT:
                features[j] = features[i];
                ++j;
                break;

            CASE_POPULATION_METRIC:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]));
                break;

            CASE_PEERS_COUNT:
                features[j] = features[i];
                ++j;
                break;

            CASE_PEERS_METRIC:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]));
                break;
            }
            break;

        case model_t::E_Metric:
        case model_t::E_PopulationMetric:
        case model_t::E_PeersMetric:

            switch (features[i]) {
            CASE_INDIVIDUAL_COUNT:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]))
                break;

            CASE_INDIVIDUAL_METRIC:
                features[j] = features[i];
                ++j;
                break;

            CASE_POPULATION_COUNT:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]))
                break;

            CASE_POPULATION_METRIC:
                features[j] = features[i];
                ++j;
                break;

            CASE_PEERS_COUNT:
                LOG_ERROR("Unexpected feature = " << model_t::print(features[i]))
                break;

            CASE_PEERS_METRIC:
                features[j] = features[i];
                ++j;
                break;
            }
            break;
        }
    }

    features.erase(features.begin() + j, features.end());
    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());

    return features;
}

//! Wrapper which copies \p features.
CDataGatherer::TFeatureVec sanitize(const CDataGatherer::TFeatureVec& features, model_t::EAnalysisCategory gathererType) {
    CDataGatherer::TFeatureVec result(features);
    return sanitize(result, gathererType);
}

//! Check if the gatherer is for population modelling.
bool isPopulation(model_t::EAnalysisCategory gathererType) {
    switch (gathererType) {
    case model_t::E_EventRate:
    case model_t::E_Metric:
        return false;

    case model_t::E_PopulationEventRate:
    case model_t::E_PeersEventRate:
    case model_t::E_PopulationMetric:
    case model_t::E_PeersMetric:
        return true;
    }
    return false;
}

} // detail::
} // unnamed::

const std::string CDataGatherer::EXPLICIT_NULL("null");
const std::size_t CDataGatherer::EXPLICIT_NULL_SUMMARY_COUNT(std::numeric_limits<std::size_t>::max());
const std::size_t CDataGatherer::ESTIMATED_MEM_USAGE_PER_BY_FIELD(10000);
const std::size_t CDataGatherer::ESTIMATED_MEM_USAGE_PER_OVER_FIELD(1000);

CDataGatherer::CDataGatherer(model_t::EAnalysisCategory gathererType,
                             model_t::ESummaryMode summaryMode,
                             const SModelParams& modelParams,
                             const std::string& summaryCountFieldName,
                             const std::string& partitionFieldName,
                             const std::string& partitionFieldValue,
                             const std::string& personFieldName,
                             const std::string& attributeFieldName,
                             const std::string& valueFieldName,
                             const TStrVec& influenceFieldNames,
                             bool useNull,
                             const CSearchKey& key,
                             const TFeatureVec& features,
                             core_t::TTime startTime,
                             int sampleCountOverride)
    : m_GathererType(gathererType),
      m_Features(detail::sanitize(features, gathererType)),
      m_SummaryMode(summaryMode),
      m_Params(modelParams),
      m_PartitionFieldName(partitionFieldName),
      m_PartitionFieldValue(CStringStore::names().get(partitionFieldValue)),
      m_SearchKey(key),
      m_PeopleRegistry(PERSON, stat_t::E_NumberNewPeople, stat_t::E_NumberNewPeopleNotAllowed, stat_t::E_NumberNewPeopleRecycled),
      m_AttributesRegistry(ATTRIBUTE,
                           stat_t::E_NumberNewAttributes,
                           stat_t::E_NumberNewAttributesNotAllowed,
                           stat_t::E_NumberNewAttributesRecycled),
      m_Population(detail::isPopulation(gathererType)),
      m_UseNull(useNull) {
    // Constructor needs to create 1 bucket gatherer at the startTime
    // and possibly 1 bucket gatherer at (startTime + bucketLength / 2).

    std::sort(m_Features.begin(), m_Features.end());
    core_t::TTime bucketLength = modelParams.s_BucketLength;

    this->createBucketGatherer(gathererType,
                               summaryCountFieldName,
                               personFieldName,
                               attributeFieldName,
                               valueFieldName,
                               influenceFieldNames,
                               startTime,
                               sampleCountOverride);

    if (modelParams.s_BucketResultsDelay > 0) {
        this->createBucketGatherer(gathererType,
                                   summaryCountFieldName,
                                   personFieldName,
                                   attributeFieldName,
                                   valueFieldName,
                                   influenceFieldNames,
                                   startTime + (bucketLength / 2),
                                   sampleCountOverride);
    }
}

CDataGatherer::CDataGatherer(model_t::EAnalysisCategory gathererType,
                             model_t::ESummaryMode summaryMode,
                             const SModelParams& modelParams,
                             const std::string& summaryCountFieldName,
                             const std::string& partitionFieldName,
                             const std::string& partitionFieldValue,
                             const std::string& personFieldName,
                             const std::string& attributeFieldName,
                             const std::string& valueFieldName,
                             const TStrVec& influenceFieldNames,
                             bool useNull,
                             const CSearchKey& key,
                             core::CStateRestoreTraverser& traverser)
    : m_GathererType(gathererType),
      m_SummaryMode(summaryMode),
      m_Params(modelParams),
      m_PartitionFieldName(partitionFieldName),
      m_PartitionFieldValue(CStringStore::names().get(partitionFieldValue)),
      m_SearchKey(key),
      m_PeopleRegistry(PERSON, stat_t::E_NumberNewPeople, stat_t::E_NumberNewPeopleNotAllowed, stat_t::E_NumberNewPeopleRecycled),
      m_AttributesRegistry(ATTRIBUTE,
                           stat_t::E_NumberNewAttributes,
                           stat_t::E_NumberNewAttributesNotAllowed,
                           stat_t::E_NumberNewAttributesRecycled),
      m_Population(detail::isPopulation(gathererType)),
      m_UseNull(useNull) {
    if (traverser.traverseSubLevel(boost::bind(&CDataGatherer::acceptRestoreTraverser,
                                               this,
                                               boost::cref(summaryCountFieldName),
                                               boost::cref(personFieldName),
                                               boost::cref(attributeFieldName),
                                               boost::cref(valueFieldName),
                                               boost::cref(influenceFieldNames),
                                               _1)) == false) {
        LOG_ERROR("Failed to correctly restore data gatherer");
    }
}

CDataGatherer::CDataGatherer(bool isForPersistence, const CDataGatherer& other)
    : m_GathererType(other.m_GathererType),
      m_Features(other.m_Features),
      m_SummaryMode(other.m_SummaryMode),
      m_Params(other.m_Params),
      m_PartitionFieldName(other.m_PartitionFieldName),
      m_PartitionFieldValue(other.m_PartitionFieldValue),
      m_SearchKey(other.m_SearchKey),
      m_PeopleRegistry(isForPersistence, other.m_PeopleRegistry),
      m_AttributesRegistry(isForPersistence, other.m_AttributesRegistry),
      m_Population(other.m_Population),
      m_UseNull(other.m_UseNull) {
    if (!isForPersistence) {
        LOG_ABORT("This constructor only creates clones for persistence");
    }
    for (TBucketGathererPVecCItr i = other.m_Gatherers.begin(); i != other.m_Gatherers.end(); ++i) {
        m_Gatherers.push_back((*i)->cloneForPersistence());
    }
    if (other.m_SampleCounts) {
        m_SampleCounts.reset(other.m_SampleCounts->cloneForPersistence());
    }
}

CDataGatherer::~CDataGatherer() {
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        delete *i;
    }
}

CDataGatherer* CDataGatherer::cloneForPersistence() const {
    return new CDataGatherer(true, *this);
}

model_t::ESummaryMode CDataGatherer::summaryMode() const {
    return m_SummaryMode;
}

model::function_t::EFunction CDataGatherer::function() const {
    return function_t::function(this->features());
}

bool CDataGatherer::isPopulation() const {
    return m_Population;
}

std::string CDataGatherer::description() const {
    return m_Gatherers.front()->description();
}

std::size_t CDataGatherer::maxDimension() const {
    return std::max(this->numberPeople(), this->numberAttributes());
}

const std::string& CDataGatherer::partitionFieldName() const {
    return m_PartitionFieldName;
}

const std::string& CDataGatherer::partitionFieldValue() const {
    return *m_PartitionFieldValue;
}

const CSearchKey& CDataGatherer::searchKey() const {
    return m_SearchKey;
}

CDataGatherer::TStrVecCItr CDataGatherer::beginInfluencers() const {
    return m_Gatherers.front()->beginInfluencers();
}

CDataGatherer::TStrVecCItr CDataGatherer::endInfluencers() const {
    return m_Gatherers.front()->endInfluencers();
}

const std::string& CDataGatherer::personFieldName() const {
    return m_Gatherers.front()->personFieldName();
}

const std::string& CDataGatherer::attributeFieldName() const {
    return m_Gatherers.front()->attributeFieldName();
}

const std::string& CDataGatherer::valueFieldName() const {
    return m_Gatherers.front()->valueFieldName();
}

const CDataGatherer::TStrVec& CDataGatherer::fieldsOfInterest() const {
    return m_Gatherers.front()->fieldsOfInterest();
}

std::size_t CDataGatherer::numberByFieldValues() const {
    return this->isPopulation() ? this->numberActiveAttributes() : this->numberActivePeople();
}

std::size_t CDataGatherer::numberOverFieldValues() const {
    return this->isPopulation() ? this->numberActivePeople() : 0;
}

bool CDataGatherer::processFields(const TStrCPtrVec& fieldValues, CEventData& result, CResourceMonitor& resourceMonitor) {
    return m_Gatherers.front()->processFields(fieldValues, result, resourceMonitor);
}

bool CDataGatherer::addArrival(const TStrCPtrVec& fieldValues, CEventData& data, CResourceMonitor& resourceMonitor) {
    // We process fields even if we are in the first partial bucket so that
    // we add enough extra memory to the resource monitor in order to control
    // the number of partitions created.
    m_Gatherers.front()->processFields(fieldValues, data, resourceMonitor);

    core_t::TTime time = data.time();
    if (time < m_Gatherers.front()->earliestBucketStartTime()) {
        // Ignore records that are out of the latency window.
        // Records in an incomplete first bucket will end up here,
        // but we don't want to model these.
        return false;
    }

    bool result = true;
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        result &= (*i)->addEventData(data);
    }
    return result;
}

void CDataGatherer::sampleNow(core_t::TTime sampleBucketStart) {
    this->chooseBucketGatherer(sampleBucketStart).sampleNow(sampleBucketStart);
}

void CDataGatherer::skipSampleNow(core_t::TTime sampleBucketStart) {
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->skipSampleNow(sampleBucketStart);
    }
}

std::size_t CDataGatherer::numberFeatures() const {
    return m_Features.size();
}

bool CDataGatherer::hasFeature(model_t::EFeature feature) const {
    return std::binary_search(m_Features.begin(), m_Features.end(), feature);
}

model_t::EFeature CDataGatherer::feature(std::size_t i) const {
    return m_Features[i];
}

const CDataGatherer::TFeatureVec& CDataGatherer::features() const {
    return m_Features;
}

std::size_t CDataGatherer::numberActivePeople() const {
    return m_PeopleRegistry.numberActiveNames();
}

std::size_t CDataGatherer::numberPeople() const {
    return m_PeopleRegistry.numberNames();
}

bool CDataGatherer::personId(const std::string& person, std::size_t& result) const {
    return m_PeopleRegistry.id(person, result);
}

bool CDataGatherer::anyPersonId(std::size_t& result) const {
    return m_PeopleRegistry.anyId(result);
}

const std::string& CDataGatherer::personName(std::size_t pid) const {
    return this->personName(pid, DEFAULT_PERSON_NAME);
}

const core::CStoredStringPtr& CDataGatherer::personNamePtr(std::size_t pid) const {
    return m_PeopleRegistry.namePtr(pid);
}

const std::string& CDataGatherer::personName(std::size_t pid, const std::string& fallback) const {
    return m_PeopleRegistry.name(pid, fallback);
}

void CDataGatherer::personNonZeroCounts(core_t::TTime time, TSizeUInt64PrVec& result) const {
    return this->chooseBucketGatherer(time).personNonZeroCounts(time, result);
}

void CDataGatherer::recyclePeople(const TSizeVec& peopleToRemove) {
    if (peopleToRemove.empty()) {
        return;
    }

    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->recyclePeople(peopleToRemove);
    }

    if (!this->isPopulation() && m_SampleCounts) {
        m_SampleCounts->recycle(peopleToRemove);
    }

    m_PeopleRegistry.recycleNames(peopleToRemove, DEFAULT_PERSON_NAME);
    core::CStatistics::instance().stat(stat_t::E_NumberPrunedItems).increment(peopleToRemove.size());
}

void CDataGatherer::removePeople(std::size_t lowestPersonToRemove) {
    if (lowestPersonToRemove >= this->numberPeople()) {
        return;
    }

    if (!this->isPopulation() && m_SampleCounts) {
        m_SampleCounts->remove(lowestPersonToRemove);
    }

    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->removePeople(lowestPersonToRemove);
    }

    m_PeopleRegistry.removeNames(lowestPersonToRemove);
}

CDataGatherer::TSizeVec& CDataGatherer::recycledPersonIds() {
    return m_PeopleRegistry.recycledIds();
}

bool CDataGatherer::isPersonActive(std::size_t pid) const {
    return m_PeopleRegistry.isIdActive(pid);
}

std::size_t CDataGatherer::addPerson(const std::string& person, CResourceMonitor& resourceMonitor, bool& addedPerson) {
    return m_PeopleRegistry.addName(person, this->chooseBucketGatherer(0).currentBucketStartTime(), resourceMonitor, addedPerson);
}

std::size_t CDataGatherer::numberActiveAttributes() const {
    return m_AttributesRegistry.numberActiveNames();
}

std::size_t CDataGatherer::numberAttributes() const {
    return m_AttributesRegistry.numberNames();
}

bool CDataGatherer::attributeId(const std::string& attribute, std::size_t& result) const {
    return m_AttributesRegistry.id(attribute, result);
}

const std::string& CDataGatherer::attributeName(std::size_t cid) const {
    return this->attributeName(cid, DEFAULT_ATTRIBUTE_NAME);
}

const std::string& CDataGatherer::attributeName(std::size_t cid, const std::string& fallback) const {
    return m_AttributesRegistry.name(cid, fallback);
}

const core::CStoredStringPtr& CDataGatherer::attributeNamePtr(std::size_t cid) const {
    return m_AttributesRegistry.namePtr(cid);
}

void CDataGatherer::recycleAttributes(const TSizeVec& attributesToRemove) {
    if (attributesToRemove.empty()) {
        return;
    }

    if (this->isPopulation() && m_SampleCounts) {
        m_SampleCounts->recycle(attributesToRemove);
    }

    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->recycleAttributes(attributesToRemove);
    }

    m_AttributesRegistry.recycleNames(attributesToRemove, DEFAULT_ATTRIBUTE_NAME);
    core::CStatistics::instance().stat(stat_t::E_NumberPrunedItems).increment(attributesToRemove.size());
}

void CDataGatherer::removeAttributes(std::size_t lowestAttributeToRemove) {
    if (lowestAttributeToRemove >= this->numberAttributes()) {
        return;
    }

    if (this->isPopulation() && m_SampleCounts) {
        m_SampleCounts->remove(lowestAttributeToRemove);
    }

    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->removeAttributes(lowestAttributeToRemove);
    }

    m_AttributesRegistry.removeNames(lowestAttributeToRemove);
}

CDataGatherer::TSizeVec& CDataGatherer::recycledAttributeIds() {
    return m_AttributesRegistry.recycledIds();
}

bool CDataGatherer::isAttributeActive(std::size_t cid) const {
    return m_AttributesRegistry.isIdActive(cid);
}

std::size_t CDataGatherer::addAttribute(const std::string& attribute, CResourceMonitor& resourceMonitor, bool& addedAttribute) {
    return m_AttributesRegistry.addName(attribute, this->chooseBucketGatherer(0).currentBucketStartTime(), resourceMonitor, addedAttribute);
}

double CDataGatherer::sampleCount(std::size_t id) const {
    if (m_SampleCounts) {
        return static_cast<double>(m_SampleCounts->count(id));
    } else {
        LOG_ERROR("Sample count for non-metric gatherer");
        return 0.0;
    }
}

double CDataGatherer::effectiveSampleCount(std::size_t id) const {
    if (m_SampleCounts) {
        return m_SampleCounts->effectiveSampleCount(id);
    } else {
        LOG_ERROR("Effective sample count for non-metric gatherer");
        return 0.0;
    }
}

void CDataGatherer::resetSampleCount(std::size_t id) {
    if (m_SampleCounts) {
        m_SampleCounts->resetSampleCount(*this, id);
    }
}

CDataGatherer::TSampleCountsPtr CDataGatherer::sampleCounts() const {
    return m_SampleCounts;
}

// Be careful here!
core_t::TTime CDataGatherer::currentBucketStartTime() const {
    return m_Gatherers.front()->currentBucketStartTime();
}

// Be careful here!
void CDataGatherer::currentBucketStartTime(core_t::TTime bucketStart) {
    m_Gatherers[0]->currentBucketStartTime(bucketStart);
    if (m_Gatherers.size() > 1) {
        m_Gatherers[1]->currentBucketStartTime(bucketStart - (m_Gatherers[1]->bucketLength() / 2));
    }
}

core_t::TTime CDataGatherer::bucketLength() const {
    return m_Gatherers.front()->bucketLength();
}

bool CDataGatherer::dataAvailable(core_t::TTime time) const {
    return this->chooseBucketGatherer(time).dataAvailable(time);
}

bool CDataGatherer::validateSampleTimes(core_t::TTime& startTime, core_t::TTime endTime) const {
    return this->chooseBucketGatherer(startTime).validateSampleTimes(startTime, endTime);
}

void CDataGatherer::timeNow(core_t::TTime time) {
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->timeNow(time);
    }
}

std::string CDataGatherer::printCurrentBucket(core_t::TTime time) const {
    return this->chooseBucketGatherer(time).printCurrentBucket();
}

const CDataGatherer::TSizeSizePrUInt64UMap& CDataGatherer::bucketCounts(core_t::TTime time) const {
    return this->chooseBucketGatherer(time).bucketCounts(time);
}

const CDataGatherer::TSizeSizePrStoredStringPtrPrUInt64UMapVec& CDataGatherer::influencerCounts(core_t::TTime time) const {
    return this->chooseBucketGatherer(time).influencerCounts(time);
}

uint64_t CDataGatherer::checksum() const {
    uint64_t result = m_PeopleRegistry.checksum();
    result = maths::CChecksum::calculate(result, m_AttributesRegistry);
    result = maths::CChecksum::calculate(result, m_SummaryMode);
    result = maths::CChecksum::calculate(result, m_Features);
    if (m_SampleCounts) {
        result = maths::CChecksum::calculate(result, m_SampleCounts->checksum(*this));
    }
    for (TBucketGathererPVecCItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        result = maths::CChecksum::calculate(result, (*i)->checksum());
    }

    LOG_TRACE("checksum = " << result);

    return result;
}

void CDataGatherer::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CDataGatherer");
    core::CMemoryDebug::dynamicSize("m_Features", m_Features, mem);
    core::CMemoryDebug::dynamicSize("m_PeopleRegistry", m_PeopleRegistry, mem);
    core::CMemoryDebug::dynamicSize("m_AttributesRegistry", m_AttributesRegistry, mem);
    core::CMemoryDebug::dynamicSize("m_PartitionFieldName", m_PartitionFieldName, mem);
    core::CMemoryDebug::dynamicSize("m_SampleCounts", m_SampleCounts, mem);
    for (TBucketGathererPVecCItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        core::CMemoryDebug::dynamicSize("BucketGatherer", *(*i), mem);
    }
}

std::size_t CDataGatherer::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Features);
    mem += core::CMemory::dynamicSize(m_PeopleRegistry);
    mem += core::CMemory::dynamicSize(m_AttributesRegistry);
    mem += core::CMemory::dynamicSize(m_PartitionFieldName);
    mem += core::CMemory::dynamicSize(m_SampleCounts);
    for (TBucketGathererPVecCItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        mem += core::CMemory::dynamicSize(*(*i));
    }
    return mem;
}

bool CDataGatherer::useNull() const {
    return m_UseNull;
}

void CDataGatherer::clear() {
    m_PeopleRegistry.clear();
    m_AttributesRegistry.clear();
    if (m_SampleCounts) {
        m_SampleCounts->clear();
    }
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        (*i)->clear();
    }
}

bool CDataGatherer::resetBucket(core_t::TTime bucketStart) {
    bool result = true;
    for (TBucketGathererPVecItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        result &= (*i)->resetBucket(bucketStart);
    }
    return result;
}

void CDataGatherer::releaseMemory(core_t::TTime samplingCutoffTime) {
    if (this->isPopulation()) {
        for (auto& gatherer : m_Gatherers) {
            gatherer->releaseMemory(samplingCutoffTime);
        }
    }
}

const SModelParams& CDataGatherer::params() const {
    return m_Params;
}

void CDataGatherer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    for (std::size_t i = 0u; i < m_Features.size(); ++i) {
        inserter.insertValue(FEATURE_TAG, static_cast<int>(m_Features[i]));
    }
    inserter.insertLevel(PEOPLE_REGISTRY_TAG, boost::bind(&CDynamicStringIdRegistry::acceptPersistInserter, m_PeopleRegistry, _1));
    inserter.insertLevel(ATTRIBUTES_REGISTRY_TAG, boost::bind(&CDynamicStringIdRegistry::acceptPersistInserter, m_AttributesRegistry, _1));

    if (m_SampleCounts) {
        inserter.insertLevel(SAMPLE_COUNTS_TAG, boost::bind(&CSampleCounts::acceptPersistInserter, m_SampleCounts.get(), _1));
    }

    inserter.insertLevel(BUCKET_GATHERER_TAG, boost::bind(&CDataGatherer::persistBucketGatherers, this, _1));
}

bool CDataGatherer::determineMetricCategory(TMetricCategoryVec& fieldMetricCategories) const {
    if (m_Features.empty()) {
        LOG_WARN("No features to determine metric category from");
        return false;
    }

    if (m_Features.size() > 1) {
        LOG_WARN(m_Features.size() << " features to determine metric category "
                                      "from - only the first will be used");
    }

    model_t::EMetricCategory result;
    if (model_t::metricCategory(m_Features.front(), result) == false) {
        LOG_ERROR("Unable to map feature " << model_t::print(m_Features.front()) << " to a metric category");
        return false;
    }

    fieldMetricCategories.push_back(result);

    return true;
}

bool CDataGatherer::extractCountFromField(const std::string& fieldName, const std::string* fieldValue, std::size_t& count) const {
    if (fieldValue == 0) {
        // Treat not present as explicit null
        count = EXPLICIT_NULL_SUMMARY_COUNT;
        return true;
    }

    std::string fieldValueCopy(*fieldValue);
    core::CStringUtils::trimWhitespace(fieldValueCopy);
    if (fieldValueCopy.empty() || fieldValueCopy == EXPLICIT_NULL) {
        count = EXPLICIT_NULL_SUMMARY_COUNT;
        return true;
    }

    double count_;
    if (core::CStringUtils::stringToType(fieldValueCopy, count_) == false || count_ < 0.0) {
        LOG_ERROR("Unable to extract count " << fieldName << " from " << fieldValueCopy);
        return false;
    }
    count = static_cast<std::size_t>(count_ + 0.5);

    // Treat count of 0 as a failure to extract. This will cause the record to be ignored.
    return count > 0;
}

bool CDataGatherer::extractMetricFromField(const std::string& fieldName, std::string fieldValue, TDouble1Vec& result) const {
    result.clear();

    core::CStringUtils::trimWhitespace(fieldValue);
    if (fieldValue.empty()) {
        LOG_WARN("Configured metric " << fieldName << " not present in event");
        return false;
    }

    const std::string& delimiter = m_Params.get().s_MultivariateComponentDelimiter;

    // Split the string up by the delimiter and parse each token separately.
    std::size_t first = 0u;
    do {
        std::size_t last = fieldValue.find(delimiter, first);
        double value;
        // Avoid a string duplication in the (common) case of only one value
        bool convertedOk = (first == 0 && last == std::string::npos)
                               ? core::CStringUtils::stringToType(fieldValue, value)
                               : core::CStringUtils::stringToType(fieldValue.substr(first, last - first), value);
        if (!convertedOk) {
            LOG_ERROR("Unable to extract " << fieldName << " from " << fieldValue);
            result.clear();
            return false;
        }
        if (maths::CMathsFuncs::isFinite(value) == false) {
            LOG_ERROR("Bad value for " << fieldName << " from " << fieldValue);
            result.clear();
            return false;
        }
        result.push_back(value);
        first = last + (last != std::string::npos ? delimiter.length() : 0);
    } while (first != std::string::npos);

    return true;
}

core_t::TTime CDataGatherer::earliestBucketStartTime() const {
    return m_Gatherers.front()->earliestBucketStartTime();
}

bool CDataGatherer::checkInvariants() const {
    LOG_DEBUG("Checking invariants for people registry");
    bool result = m_PeopleRegistry.checkInvariants();
    LOG_DEBUG("Checking invariants for attributes registry");
    result &= m_AttributesRegistry.checkInvariants();
    return result;
}

const CBucketGatherer& CDataGatherer::chooseBucketGatherer(core_t::TTime time) const {
    return const_cast<CDataGatherer*>(this)->chooseBucketGatherer(time);
}

CBucketGatherer& CDataGatherer::chooseBucketGatherer(core_t::TTime time) {
    core_t::TTime bucketLength = m_Gatherers.front()->bucketLength();
    if ((m_Gatherers.size() > 1) && (time % bucketLength != 0)) {
        return *m_Gatherers[1];
    } else {
        return *m_Gatherers[0];
    }
}

bool CDataGatherer::acceptRestoreTraverser(const std::string& summaryCountFieldName,
                                           const std::string& personFieldName,
                                           const std::string& attributeFieldName,
                                           const std::string& valueFieldName,
                                           const TStrVec& influenceFieldNames,
                                           core::CStateRestoreTraverser& traverser) {
    this->clear();
    m_Features.clear();

    do {
        const std::string& name = traverser.name();
        if (name == FEATURE_TAG) {
            int feature(-1);
            if (core::CStringUtils::stringToType(traverser.value(), feature) == false || feature < 0) {
                LOG_ERROR("Invalid feature in " << traverser.value());
                return false;
            }
            m_Features.push_back(static_cast<model_t::EFeature>(feature));
            continue;
        }
        RESTORE(PEOPLE_REGISTRY_TAG,
                traverser.traverseSubLevel(boost::bind(&CDynamicStringIdRegistry::acceptRestoreTraverser, &m_PeopleRegistry, _1)))
        RESTORE(ATTRIBUTES_REGISTRY_TAG,
                traverser.traverseSubLevel(boost::bind(&CDynamicStringIdRegistry::acceptRestoreTraverser, &m_AttributesRegistry, _1)))
        RESTORE_SETUP_TEARDOWN(SAMPLE_COUNTS_TAG,
                               m_SampleCounts.reset(new CSampleCounts(0)),
                               traverser.traverseSubLevel(boost::bind(&CSampleCounts::acceptRestoreTraverser, m_SampleCounts.get(), _1)),
                               /**/)
        RESTORE(BUCKET_GATHERER_TAG,
                traverser.traverseSubLevel(boost::bind(&CDataGatherer::restoreBucketGatherer,
                                                       this,
                                                       boost::cref(summaryCountFieldName),
                                                       boost::cref(personFieldName),
                                                       boost::cref(attributeFieldName),
                                                       boost::cref(valueFieldName),
                                                       boost::cref(influenceFieldNames),
                                                       _1)))
    } while (traverser.next());

    return true;
}

bool CDataGatherer::restoreBucketGatherer(const std::string& summaryCountFieldName,
                                          const std::string& personFieldName,
                                          const std::string& attributeFieldName,
                                          const std::string& valueFieldName,
                                          const TStrVec& influenceFieldNames,
                                          core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == CBucketGatherer::EVENTRATE_BUCKET_GATHERER_TAG) {
            CEventRateBucketGatherer* gatherer = new CEventRateBucketGatherer(
                *this, summaryCountFieldName, personFieldName, attributeFieldName, valueFieldName, influenceFieldNames, traverser);

            if (gatherer == 0) {
                LOG_ERROR("Failed to create gatherer");
                return false;
            }
            m_Gatherers.push_back(gatherer);
        } else if (name == CBucketGatherer::METRIC_BUCKET_GATHERER_TAG) {
            CMetricBucketGatherer* gatherer = new CMetricBucketGatherer(
                *this, summaryCountFieldName, personFieldName, attributeFieldName, valueFieldName, influenceFieldNames, traverser);
            if (gatherer == 0) {
                LOG_ERROR("Failed to create gatherer");
                return false;
            }
            m_Gatherers.push_back(gatherer);
        }
    } while (traverser.next());

    return true;
}

void CDataGatherer::persistBucketGatherers(core::CStatePersistInserter& inserter) const {
    for (TBucketGathererPVecCItr i = m_Gatherers.begin(); i != m_Gatherers.end(); ++i) {
        const std::string& tag = (*i)->persistenceTag();

        if (tag == CBucketGatherer::EVENTRATE_BUCKET_GATHERER_TAG) {
            CEventRateBucketGatherer* const gatherer = dynamic_cast<CEventRateBucketGatherer* const>(*i);
            inserter.insertLevel(tag, boost::bind(&CEventRateBucketGatherer::acceptPersistInserter, boost::cref(gatherer), _1));
        } else if (tag == CBucketGatherer::METRIC_BUCKET_GATHERER_TAG) {
            CMetricBucketGatherer* const gatherer = dynamic_cast<CMetricBucketGatherer* const>(*i);
            inserter.insertLevel(tag, boost::bind(&CMetricBucketGatherer::acceptPersistInserter, boost::cref(gatherer), _1));
        }
    }
}

void CDataGatherer::createBucketGatherer(model_t::EAnalysisCategory gathererType,
                                         const std::string& summaryCountFieldName,
                                         const std::string& personFieldName,
                                         const std::string& attributeFieldName,
                                         const std::string& valueFieldName,
                                         const TStrVec& influenceFieldNames,
                                         core_t::TTime startTime,
                                         unsigned int sampleCountOverride) {
    switch (gathererType) {
    case model_t::E_EventRate:
    case model_t::E_PopulationEventRate:
    case model_t::E_PeersEventRate:
        m_Gatherers.push_back(new CEventRateBucketGatherer(
            *this, summaryCountFieldName, personFieldName, attributeFieldName, valueFieldName, influenceFieldNames, startTime));
        break;
    case model_t::E_Metric:
    case model_t::E_PopulationMetric:
    case model_t::E_PeersMetric:
        m_SampleCounts.reset(new CSampleCounts(sampleCountOverride));
        m_Gatherers.push_back(new CMetricBucketGatherer(
            *this, summaryCountFieldName, personFieldName, attributeFieldName, valueFieldName, influenceFieldNames, startTime));
        break;
    }
}
}
}
