/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CModelTestFixtureBase.h"

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CModelFactory.h>

#include <string>

const std::string CModelTestFixtureBase::EMPTY_STRING;

std::size_t CModelTestFixtureBase::addPerson(const std::string& p,
                                             const ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                             std::size_t numInfluencers,
                                             TOptionalStr value) {
    std::string i("i");

    ml::model::CDataGatherer::TStrCPtrVec person{&p};
    if (numInfluencers == 0) {
        person.resize(gatherer->fieldsOfInterest().size(), nullptr);
    }
    for (std::size_t j = 0; j < numInfluencers; ++j) {
        person.push_back(&i);
    }
    if (value) {
        person.push_back(&(value.get()));
    }
    ml::model::CEventData result;
    gatherer->processFields(person, result, m_ResourceMonitor);
    return *result.personId();
}

std::string CModelTestFixtureBase::valueAsString(const TDouble1Vec& value) {
    std::string result{ml::core::CStringUtils::typeToStringPrecise(
        value[0], ml::core::CIEEE754::E_DoublePrecision)};
    for (std::size_t i = 1; i < value.size(); ++i) {
        result += ml::model::CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER +
                  ml::core::CStringUtils::typeToStringPrecise(
                      value[i], ml::core::CIEEE754::E_DoublePrecision);
    }
    return result;
}

ml::model::CEventData
CModelTestFixtureBase::addArrival(const SMessage& message,
                                  ml::model::CModelFactory::TDataGathererPtr& gatherer) {
    ml::model::CDataGatherer::TStrCPtrVec fields{&message.s_Person};
    if (message.s_Attribute) {
        fields.push_back(&message.s_Attribute.get());
    }
    std::string value;
    if (message.s_Dbl1Vec) {
        value = {valueAsString(message.s_Dbl1Vec.get())};
        fields.push_back(&value);
    }
    if (message.s_Inf1) {
        fields.push_back(&(message.s_Inf1.get()));
    }
    if (message.s_Inf2) {
        fields.push_back(&(message.s_Inf2.get()));
    }
    if (message.s_Value) {
        fields.push_back(&(message.s_Value.get()));
    }
    std::string dblAsString;
    if (message.s_Dbl) {
        dblAsString = ml::core::CStringUtils::typeToStringPrecise(
            message.s_Dbl.get(), ml::core::CIEEE754::E_DoublePrecision);
        fields.push_back(&dblAsString);
    }
    std::string delimitedDblPr;
    if (message.s_DblPr) {
        delimitedDblPr += ml::core::CStringUtils::typeToStringPrecise(
            message.s_DblPr.get().first, ml::core::CIEEE754::E_DoublePrecision);
        delimitedDblPr += ml::model::CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER;
        delimitedDblPr += ml::core::CStringUtils::typeToStringPrecise(
            message.s_DblPr.get().second, ml::core::CIEEE754::E_DoublePrecision);
        fields.push_back(&delimitedDblPr);
    }

    ml::model::CEventData result;
    result.time(message.s_Time);
    gatherer->addArrival(fields, result, m_ResourceMonitor);

    return result;
}

void CModelTestFixtureBase::processBucket(ml::core_t::TTime time,
                                          ml::core_t::TTime bucketLength,
                                          const TDoubleVec& bucket,
                                          const TStrVec& influencerValues,
                                          ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                          ml::model::CAnomalyDetectorModel& model,
                                          ml::model::SAnnotatedProbability& probability) {
    for (std::size_t i = 0; i < bucket.size(); ++i) {
        this->addArrival(
            SMessage(time, "p", bucket[i], {}, TOptionalStr(influencerValues[i])), gatherer);
    }
    model.sample(time, time + bucketLength, m_ResourceMonitor);
    ml::model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    LOG_DEBUG(<< "influences = "
              << ml::core::CContainerPrinter::print(probability.s_Influences));
}

void CModelTestFixtureBase::processBucket(ml::core_t::TTime time,
                                          ml::core_t::TTime bucketLength,
                                          const TDoubleVec& bucket,
                                          ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                          ml::model::CAnomalyDetectorModel& model,
                                          ml::model::SAnnotatedProbability& probability,
                                          ml::model::SAnnotatedProbability& probability2) {
    const std::string person("p");
    const std::string person2("q");
    for (std::size_t i = 0; i < bucket.size(); ++i) {
        ml::model::CDataGatherer::TStrCPtrVec fieldValues;
        if (i % 2 == 0) {
            fieldValues.push_back(&person);
        } else {
            fieldValues.push_back(&person2);
        }

        std::string valueAsString(ml::core::CStringUtils::typeToStringPrecise(
            bucket[i], ml::core::CIEEE754::E_DoublePrecision));
        fieldValues.push_back(&valueAsString);

        ml::model::CEventData eventData;
        eventData.time(time);

        gatherer->addArrival(fieldValues, eventData, m_ResourceMonitor);
    }
    model.sample(time, time + bucketLength, m_ResourceMonitor);
    ml::model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    model.computeProbability(1 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability2);
}

void CModelTestFixtureBase::processBucket(ml::core_t::TTime time,
                                          ml::core_t::TTime bucketLength,
                                          const TDoubleStrPrVec& bucket,
                                          ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                          ml::model::CAnomalyDetectorModel& model,
                                          ml::model::SAnnotatedProbability& probability) {
    const std::string person{"p"};
    const std::string attribute{"a"};
    for (auto& pr : bucket) {
        const std::string valueAsString{ml::core::CStringUtils::typeToStringPrecise(
            pr.first, ml::core::CIEEE754::E_DoublePrecision)};

        ml::model::CDataGatherer::TStrCPtrVec fieldValues{
            &person, &attribute, &pr.second, &valueAsString};

        ml::model::CEventData eventData;
        eventData.time(time);

        gatherer->addArrival(fieldValues, eventData, m_ResourceMonitor);
    }
    model.sample(time, time + bucketLength, m_ResourceMonitor);
    ml::model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
    model.computeProbability(0 /*pid*/, time, time + bucketLength,
                             partitioningFields, 1, probability);
    LOG_DEBUG(<< "influences = "
              << ml::core::CContainerPrinter::print(probability.s_Influences));
}

void CModelTestFixtureBase::generateOrderedAnomalies(std::size_t numAnomalies,
                                                     ml::core_t::TTime startTime,
                                                     ml::core_t::TTime bucketLength,
                                                     const TMessageVec& messages,
                                                     ml::model::CModelFactory::TDataGathererPtr& gatherer,
                                                     ml::model::CAnomalyDetectorModel& model,
                                                     TAnomalyVec& orderedAnomalies) {

    TAnomalyAccumulator anomalies(numAnomalies);

    for (std::size_t i = 0u, bucket = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= startTime + bucketLength) {
            LOG_DEBUG(<< "Updating and testing bucket = [" << startTime << ","
                      << startTime + bucketLength << ")");

            model.sample(startTime, startTime + bucketLength, m_ResourceMonitor);

            ml::model::CPartitioningFields partitioningFields(EMPTY_STRING, EMPTY_STRING);
            ml::model::SAnnotatedProbability annotatedProbability;
            for (std::size_t pid = 0; pid < gatherer->numberActivePeople(); ++pid) {
                model.computeProbability(pid, startTime, startTime + bucketLength,
                                         partitioningFields, 2, annotatedProbability);

                std::string person = model.personName(pid);
                TDoubleStrPrVec attributes;
                for (const auto& probability : annotatedProbability.s_AttributeProbabilities) {
                    attributes.emplace_back(probability.s_Probability,
                                            *probability.s_Attribute);
                }
                LOG_DEBUG(<< "Adding anomaly " << pid);
                anomalies.add({annotatedProbability.s_Probability,
                               {bucket, person, attributes}});
            }

            startTime += bucketLength;
            ++bucket;
        }

        this->addArrival(messages[i], gatherer);
    }

    anomalies.sort();
    LOG_DEBUG(<< "Anomalies = " << anomalies.print());

    for (std::size_t i = 0; i < anomalies.count(); ++i) {
        orderedAnomalies.push_back(anomalies[i].second);
    }

    std::sort(orderedAnomalies.begin(), orderedAnomalies.end());

    LOG_DEBUG(<< "orderedAnomalies = "
              << ml::core::CContainerPrinter::print(orderedAnomalies));
}

ml::model::CEventData CModelTestFixtureBase::makeEventData(ml::core_t::TTime time,
                                                           std::size_t pid,
                                                           TDouble1Vec value,
                                                           const TOptionalStr& influence,
                                                           const TOptionalStr& stringValue) {
    ml::model::CEventData result;
    result.time(time);
    result.person(pid);
    result.addAttribute(std::size_t(0));
    result.addValue(value);
    result.addInfluence(influence);
    if (stringValue) {
        result.stringValue(stringValue.get());
    }
    return result;
}

void CModelTestFixtureBase::generateAndCompareKey(const ml::model::function_t::TFunctionVec& countFunctions,
                                                  const std::string& fieldName,
                                                  const std::string& overFieldName,
                                                  TKeyCompareFunc keyCompare) {
    TStrVec byFields{"", "by"};
    TStrVec partitionFields{"", "partition"};

    ml::model::CAnomalyDetectorModelConfig config =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig();

    int detectorIndex{0};
    for (const auto& countFunction : countFunctions) {
        for (bool usingNull : {true, false}) {
            for (const auto& byField : byFields) {
                for (const auto& partitionField : partitionFields) {
                    ml::model::CSearchKey key(++detectorIndex, countFunction, usingNull,
                                              ml::model_t::E_XF_None, fieldName,
                                              byField, overFieldName, partitionField);

                    ml::model::CAnomalyDetectorModelConfig::TModelFactoryCPtr factory =
                        config.factory(key);

                    LOG_DEBUG(<< "expected key = " << key);
                    LOG_DEBUG(<< "actual key   = " << factory->searchKey());
                    keyCompare(key, factory->searchKey());
                }
            }
        }
    }
}
