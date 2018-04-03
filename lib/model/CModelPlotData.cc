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
#include <model/CModelPlotData.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <utility>

namespace ml {
namespace model {
namespace {

const std::string DATA_PER_FEATURE_TAG("a");
const std::string TIME_TAG("b");
const std::string PARTITION_FIELD_NAME_TAG("c");
const std::string PARTITION_FIELD_VALUE_TAG("d");
const std::string OVER_FIELD_NAME_TAG("e");
const std::string BY_FIELD_NAME_TAG("f");

const std::string LOWER_BOUND_TAG("a");
const std::string UPPER_BOUND_TAG("b");
const std::string MEDIAN_TAG("c");
const std::string VALUES_PER_OVERFIELD_TAG("d");
}

CModelPlotData::CModelPlotData(void) : m_Time(0) {
}

CModelPlotData::CModelPlotData(core_t::TTime time,
                               const std::string& partitionFieldName,
                               const std::string& partitionFieldValue,
                               const std::string& overFieldName,
                               const std::string& byFieldName,
                               core_t::TTime bucketSpan,
                               int detectorIndex)
    : m_Time(time),
      m_PartitionFieldName(partitionFieldName),
      m_PartitionFieldValue(partitionFieldValue),
      m_OverFieldName(overFieldName),
      m_ByFieldName(byFieldName),
      m_BucketSpan(bucketSpan),
      m_DetectorIndex(detectorIndex) {
}

CModelPlotData::SByFieldData::SByFieldData(void) : s_LowerBound(0.0), s_UpperBound(0.0), s_Median(0.0), s_ValuesPerOverField() {
}

CModelPlotData::SByFieldData::SByFieldData(double lowerBound, double upperBound, double median)
    : s_LowerBound(lowerBound), s_UpperBound(upperBound), s_Median(median), s_ValuesPerOverField() {
}

void CModelPlotData::SByFieldData::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(LOWER_BOUND_TAG, s_LowerBound, inserter);
    core::CPersistUtils::persist(UPPER_BOUND_TAG, s_UpperBound, inserter);
    core::CPersistUtils::persist(MEDIAN_TAG, s_Median, inserter);
    core::CPersistUtils::persist(VALUES_PER_OVERFIELD_TAG, s_ValuesPerOverField, inserter);
}

bool CModelPlotData::SByFieldData::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == LOWER_BOUND_TAG) {
            if (!core::CPersistUtils::restore(LOWER_BOUND_TAG, s_LowerBound, traverser)) {
                return false;
            }
        } else if (name == UPPER_BOUND_TAG) {
            if (!core::CPersistUtils::restore(UPPER_BOUND_TAG, s_UpperBound, traverser)) {
                return false;
            }
        } else if (name == MEDIAN_TAG) {
            if (!core::CPersistUtils::restore(MEDIAN_TAG, s_Median, traverser)) {
                return false;
            }
        } else if (name == VALUES_PER_OVERFIELD_TAG) {
            if (!core::CPersistUtils::restore(VALUES_PER_OVERFIELD_TAG, s_ValuesPerOverField, traverser)) {
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

void CModelPlotData::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    TIntStrByFieldDataUMapUMap data(m_DataPerFeature.begin(), m_DataPerFeature.end());
    core::CPersistUtils::persist(DATA_PER_FEATURE_TAG, data, inserter);
    core::CPersistUtils::persist(TIME_TAG, m_Time, inserter);
    core::CPersistUtils::persist(PARTITION_FIELD_NAME_TAG, m_PartitionFieldName, inserter);
    core::CPersistUtils::persist(PARTITION_FIELD_VALUE_TAG, m_PartitionFieldValue, inserter);
    core::CPersistUtils::persist(OVER_FIELD_NAME_TAG, m_OverFieldName, inserter);
    core::CPersistUtils::persist(BY_FIELD_NAME_TAG, m_ByFieldName, inserter);
}

bool CModelPlotData::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == DATA_PER_FEATURE_TAG) {
            TIntStrByFieldDataUMapUMap data;
            if (!core::CPersistUtils::restore(DATA_PER_FEATURE_TAG, data, traverser)) {
                return false;
            }
            m_DataPerFeature.clear();

            for (TIntStrByFieldDataUMapUMap::const_iterator i = data.begin(); i != data.end(); ++i) {
                m_DataPerFeature.insert(TFeatureStrByFieldDataUMapPr(model_t::EFeature(i->first), i->second));
            }
        } else if (name == TIME_TAG) {
            if (!core::CPersistUtils::restore(TIME_TAG, m_Time, traverser)) {
                return false;
            }
        } else if (name == PARTITION_FIELD_NAME_TAG) {
            if (!core::CPersistUtils::restore(PARTITION_FIELD_NAME_TAG, m_PartitionFieldName, traverser)) {
                return false;
            }
        } else if (name == PARTITION_FIELD_VALUE_TAG) {
            if (!core::CPersistUtils::restore(PARTITION_FIELD_VALUE_TAG, m_PartitionFieldValue, traverser)) {
                return false;
            }
        } else if (name == OVER_FIELD_NAME_TAG) {
            if (!core::CPersistUtils::restore(OVER_FIELD_NAME_TAG, m_OverFieldName, traverser)) {
                return false;
            }
        } else if (name == BY_FIELD_NAME_TAG) {
            if (!core::CPersistUtils::restore(BY_FIELD_NAME_TAG, m_ByFieldName, traverser)) {
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

const std::string& CModelPlotData::partitionFieldName(void) const {
    return m_PartitionFieldName;
}

const std::string& CModelPlotData::partitionFieldValue(void) const {
    return m_PartitionFieldValue;
}

const std::string& CModelPlotData::overFieldName(void) const {
    return m_OverFieldName;
}

const std::string& CModelPlotData::byFieldName(void) const {
    return m_ByFieldName;
}

core_t::TTime CModelPlotData::time(void) const {
    return m_Time;
}

core_t::TTime CModelPlotData::bucketSpan(void) const {
    return m_BucketSpan;
}

int CModelPlotData::detectorIndex(void) const {
    return m_DetectorIndex;
}

void CModelPlotData::SByFieldData::addValue(const std::string& personName, double value) {
    s_ValuesPerOverField.emplace_back(personName, value);
}

CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr CModelPlotData::begin(void) const {
    return m_DataPerFeature.begin();
}

CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr CModelPlotData::end(void) const {
    return m_DataPerFeature.end();
}

CModelPlotData::SByFieldData& CModelPlotData::get(const model_t::EFeature& feature, const std::string& byFieldValue) {
    // note: This creates/inserts! elements and returns a reference for writing
    // data insert happens here
    return m_DataPerFeature[feature][byFieldValue];
}

std::string CModelPlotData::print(void) const {
    return "nothing";
}
}
}
