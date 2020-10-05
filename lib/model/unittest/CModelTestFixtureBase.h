/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CModelTestFixtureBase_h
#define INCLUDED_CModelTestFixtureBase_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CModel.h>
#include <maths/CMultivariatePrior.h>

#include <model/CAnnotatedProbability.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CIndividualModel.h>
#include <model/CModelFactory.h>
#include <model/CResourceMonitor.h>

#include <boost/optional.hpp>

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

class CModelTestFixtureBase {
public:
    static const std::string EMPTY_STRING;

    using TBoolVec = std::vector<bool>;

    using TDouble1Vec = ml::core::CSmallVector<double, 1>;
    using TDouble2Vec = ml::core::CSmallVector<double, 2>;
    using TDouble4Vec = ml::core::CSmallVector<double, 4>;
    using TDouble4Vec1Vec = ml::core::CSmallVector<TDouble4Vec, 1>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleStrPr = std::pair<double, std::string>;
    using TDoubleStrPrVec = std::vector<TDoubleStrPr>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;

    using TMathsModelPtr = std::shared_ptr<ml::maths::CModel>;
    using TMeanAccumulator = ml::maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMultivariatePriorPtr = std::shared_ptr<ml::maths::CMultivariatePrior>;

    using TOptionalDouble = boost::optional<double>;
    using TOptionalDoubleVec = std::vector<TOptionalDouble>;
    using TOptionalStr = boost::optional<std::string>;
    using TOptionalUInt64 = boost::optional<uint64_t>;

    using TPriorPtr = std::shared_ptr<ml::maths::CPrior>;

    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr1Vec = ml::core::CSmallVector<TSizeDoublePr, 1>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePrVec = std::vector<TSizeSizePr>;
    using TSizeSizePrVecVec = std::vector<TSizeSizePrVec>;
    using TSizeSizePrUInt64Map = std::map<TSizeSizePr, uint64_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeVecVecVec = std::vector<TSizeVecVec>;
    using TStrSizePr = std::pair<std::string, std::size_t>;
    using TStrSizePrVec = std::vector<TStrSizePr>;
    using TStrSizePrVecVec = std::vector<TStrSizePrVec>;
    using TStrSizePrVecVecVec = std::vector<TStrSizePrVecVec>;
    using TStrUInt64Map = std::map<std::string, uint64_t>;

    using TStrVec = std::vector<std::string>;
    using TStrVecVec = std::vector<TStrVec>;

    using TTimeDoublePr = std::pair<ml::core_t::TTime, double>;
    using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
    using TOptionalTimeDoublePr = boost::optional<TTimeDoublePr>;
    using TTimeStrVecPr = std::pair<ml::core_t::TTime, TStrVec>;
    using TTimeStrVecPrVec = std::vector<TTimeStrVecPr>;
    using TTimeVec = std::vector<ml::core_t::TTime>;

    using TUInt64Vec = std::vector<uint64_t>;
    using TUIntVec = std::vector<unsigned int>;

protected:
    struct SAnomaly {
        SAnomaly() : s_Bucket(0u), s_Person(), s_Attributes() {}
        SAnomaly(std::size_t bucket, const std::string& person, const TDoubleStrPrVec& attributes)
            : s_Bucket(bucket), s_Person(person), s_Attributes(attributes) {}

        std::size_t s_Bucket;
        std::string s_Person;
        TDoubleStrPrVec s_Attributes;

        bool operator<(const SAnomaly& other) const {
            return s_Bucket < other.s_Bucket;
        }

        std::string print() const {
            std::ostringstream result;
            result << "[" << s_Bucket << ", " + s_Person << ",";
            for (std::size_t i = 0u; i < s_Attributes.size(); ++i) {
                if (s_Attributes[i].first < 0.01) {
                    result << " " << s_Attributes[i].second;
                }
            }
            result << "]";
            return result.str();
        }
    };

    using TAnomalyVec = std::vector<SAnomaly>;
    using TDoubleAnomalyPr = std::pair<double, SAnomaly>;
    using TAnomalyAccumulator =
        ml::maths::CBasicStatistics::COrderStatisticsHeap<TDoubleAnomalyPr, ml::maths::COrderings::SFirstLess>;

    struct SMessage {
        SMessage(ml::core_t::TTime time,
                 const std::string& person,
                 const std::string& attribute,
                 const TDouble1Vec& value)
            : s_Time(time), s_Person(person), s_Attribute(attribute), s_Value(value) {}

        SMessage(ml::core_t::TTime time, const std::string& person, const std::string& attribute)
            : s_Time(time), s_Person(person), s_Attribute(attribute) {}

        bool operator<(const SMessage& other) const {
            return ml::maths::COrderings::lexicographical_compare(
                s_Time, s_Person, s_Attribute, other.s_Time, other.s_Person,
                other.s_Attribute);
        }

        ml::core_t::TTime s_Time;
        std::string s_Person;
        std::string s_Attribute;
        TDouble1Vec s_Value{};
    };
    using TMessageVec = std::vector<SMessage>;

protected:
    std::size_t addPerson(const std::string& p,
                          const ml::model::CModelFactory::TDataGathererPtr& gatherer,
                          std::size_t numInfluencers = 0,
                          TOptionalStr value = TOptionalStr());

    void addArrival(ml::model::CDataGatherer& gatherer,
                    ml::core_t::TTime time,
                    const std::string& person,
                    const TOptionalStr& inf1 = TOptionalStr(),
                    const TOptionalStr& inf2 = TOptionalStr(),
                    const TOptionalStr& value = TOptionalStr());

    std::string valueAsString(const TDouble1Vec& value);

    ml::model::CEventData addArrival(const SMessage& message,
                                     ml::model::CModelFactory::TDataGathererPtr& gatherer);

    void addArrival(ml::model::CDataGatherer& gatherer,
                    ml::core_t::TTime time,
                    const std::string& person,
                    double value,
                    const TOptionalStr& inf1 = TOptionalStr(),
                    const TOptionalStr& inf2 = TOptionalStr(),
                    const TOptionalStr& count = TOptionalStr());

    void addArrival(ml::model::CDataGatherer& gatherer,
                    ml::core_t::TTime time,
                    const std::string& person,
                    double lat,
                    double lng,
                    const TOptionalStr& inf1 = TOptionalStr(),
                    const TOptionalStr& inf2 = TOptionalStr());

    void processBucket(ml::core_t::TTime time,
                       ml::core_t::TTime bucketLength,
                       const TDoubleVec& bucket,
                       const TStrVec& influencerValues,
                       ml::model::CDataGatherer& gatherer,
                       ml::model::CAnomalyDetectorModel& model,
                       ml::model::SAnnotatedProbability& probability);

    void processBucket(ml::core_t::TTime time,
                       ml::core_t::TTime bucketLength,
                       const TDoubleVec& bucket,
                       ml::model::CDataGatherer& gatherer,
                       ml::model::CAnomalyDetectorModel& model,
                       ml::model::SAnnotatedProbability& probability,
                       ml::model::SAnnotatedProbability& probability2);

    void processBucket(ml::core_t::TTime time,
                       ml::core_t::TTime bucketLength,
                       const TDoubleStrPrVec& bucket,
                       ml::model::CDataGatherer& gatherer,
                       ml::model::CAnomalyDetectorModel& model,
                       ml::model::SAnnotatedProbability& probability);

    void generateOrderedAnomalies(std::size_t numAnomalies,
                                  ml::core_t::TTime startTime,
                                  ml::core_t::TTime bucketLength,
                                  const TMessageVec& messages,
                                  ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                  ml::model::CAnomalyDetectorModel& model,
                                  TAnomalyVec& orderedAnomalies);

    ml::model::CEventData makeEventData(ml::core_t::TTime time,
                                        std::size_t pid,
                                        TDouble1Vec value = TDoubleVec(1, 0.0),
                                        const TOptionalStr& influence = TOptionalStr(),
                                        const TOptionalStr& stringValue = TOptionalStr());

    using TKeyCompareFunc =
        std::function<void(ml::model::CSearchKey expectedKey, ml::model::CSearchKey actualKey)>;

    static void generateAndCompareKey(const ml::model::function_t::TFunctionVec& countFunctions,
                                      const std::string& fieldName,
                                      const std::string& overFieldName,
                                      TKeyCompareFunc keyCompare);

protected:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif //INCLUDED_CModelTestFixtureBase_h
