/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CPopulationModelDetail_h
#define INCLUDED_ml_model_CPopulationModelDetail_h

#include <model/CPopulationModel.h>

#include <model/CDataGatherer.h>

namespace ml {
namespace model {

template<typename T>
CPopulationModel::TSizeSizePr CPopulationModel::personRange(const T& data, std::size_t pid) {
    const std::size_t minCid = 0u;
    const std::size_t maxCid = std::numeric_limits<std::size_t>::max();
    auto begin = std::lower_bound(data.begin(), data.end(), std::make_pair(pid, minCid),
                                  maths::COrderings::SFirstLess());
    auto end = std::upper_bound(begin, data.end(), std::make_pair(pid, maxCid),
                                maths::COrderings::SFirstLess());
    return {static_cast<std::size_t>(begin - data.begin()),
            static_cast<std::size_t>(end - data.begin())};
}

template<typename T>
typename T::const_iterator
CPopulationModel::find(const T& data, std::size_t pid, std::size_t cid) {
    auto i = std::lower_bound(data.begin(), data.end(), std::make_pair(pid, cid),
                              maths::COrderings::SFirstLess());
    if (i != data.end() && (CDataGatherer::extractPersonId(*i) != pid ||
                            CDataGatherer::extractAttributeId(*i) != cid)) {
        i = data.end();
    }
    return i;
}

inline CPopulationModel::TDouble1Vec
CPopulationModel::extractValue(model_t::EFeature /*feature*/,
                               const std::pair<TSizeSizePr, SEventRateFeatureData>& data) {
    return TDouble1Vec{static_cast<double>(CDataGatherer::extractData(data).s_Count)};
}

inline CPopulationModel::TDouble1Vec
CPopulationModel::extractValue(model_t::EFeature feature,
                               const std::pair<TSizeSizePr, SMetricFeatureData>& data) {
    return CDataGatherer::extractData(data).s_BucketValue
               ? CDataGatherer::extractData(data).s_BucketValue->value(model_t::dimension(feature))
               : TDouble1Vec();
}

template<typename T, typename PERSON_FILTER, typename ATTRIBUTE_FILTER>
void CPopulationModel::applyFilters(bool updateStatistics,
                                    const PERSON_FILTER& personFilter,
                                    const ATTRIBUTE_FILTER& attributeFilter,
                                    T& data) const {
    std::size_t initialSize = data.size();
    if (this->params().s_ExcludeFrequent & model_t::E_XF_Over) {
        data.erase(std::remove_if(data.begin(), data.end(), personFilter), data.end());
    }
    if (this->params().s_ExcludeFrequent & model_t::E_XF_By) {
        data.erase(std::remove_if(data.begin(), data.end(), attributeFilter), data.end());
    }
    if (updateStatistics && data.size() != initialSize) {
        core::CProgramCounters::counter(counter_t::E_TSADNumberExcludedFrequentInvocations)++;
    }
}
}
}

#endif // INCLUDED_ml_model_CPopulationModelDetail_h
