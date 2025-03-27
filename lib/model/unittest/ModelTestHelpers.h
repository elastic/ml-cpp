/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_model_ModelTestHelpers_h
#define INCLUDED_ml_model_ModelTestHelpers_h

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <model/CDataGatherer.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>

#include <boost/test/unit_test.hpp>

namespace ml {
namespace model {

const CSearchKey KEY;
const std::string EMPTY_STRING;

static void testPersistence(const SModelParams& params,
                            const CDataGatherer& origGatherer,
                            model_t::EAnalysisCategory category) {
    // Test persistence. (We check for idempotency.)
    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, [&origGatherer](core::CJsonStatePersistInserter& inserter) {
            origGatherer.acceptPersistInserter(inserter);
        });

    LOG_DEBUG(<< "gatherer JSON size " << origJson.str().size());
    LOG_TRACE(<< "gatherer JSON representation:\n" << origJson.str());

    // Restore the JSON into a new filter
    // The traverser expects the state json in a embedded document
    std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);

    CBucketGatherer::SBucketGathererInitData bucketGathererInitData{
        EMPTY_STRING,
        EMPTY_STRING,
        EMPTY_STRING,
        EMPTY_STRING,
        {},
        0,
        0};

    CDataGatherer restoredGatherer(category, model_t::E_None, params,
                                   EMPTY_STRING, KEY, bucketGathererInitData, traverser);

    BOOST_REQUIRE_EQUAL(origGatherer.checksum(), restoredGatherer.checksum());

    // The JSON representation of the new filter should be the
    // same as the original
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, [&restoredGatherer](core::CJsonStatePersistInserter& inserter) {
            restoredGatherer.acceptPersistInserter(inserter);
        });
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

static void testGathererAttributes(const CDataGatherer& gatherer,
                                   core_t::TTime startTime,
                                   core_t::TTime bucketLength) {

    BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
    BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
    BOOST_REQUIRE_EQUAL(std::string("p"), gatherer.personName(0));
    BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(1));
    std::size_t pid;
    BOOST_TEST_REQUIRE(gatherer.personId("p", pid));
    BOOST_REQUIRE_EQUAL(0, pid);
    BOOST_TEST_REQUIRE(!gatherer.personId("a.n.other p", pid));

    BOOST_REQUIRE_EQUAL(0, gatherer.numberActiveAttributes());
    BOOST_REQUIRE_EQUAL(0, gatherer.numberOverFieldValues());

    BOOST_REQUIRE_EQUAL(startTime, gatherer.currentBucketStartTime());
    BOOST_REQUIRE_EQUAL(bucketLength, gatherer.bucketLength());
}

class CDataGathererBuilder {
public:
    using TFeatureVec = CDataGatherer::TFeatureVec;
    using TStrVec = CDataGatherer::TStrVec;

public:
    CDataGathererBuilder(model_t::EAnalysisCategory gathererType,
                         const TFeatureVec& features,
                         const SModelParams& params,
                         const CSearchKey& searchKey,
                         const core_t::TTime startTime)
        : m_Features(features), m_Params(params), m_StartTime(startTime),
          m_SearchKey(searchKey), m_GathererType(gathererType) {}

    CDataGatherer build() const {
        CBucketGatherer::SBucketGathererInitData bucketGathererInitData{
            m_SummaryCountFieldName,
            m_PersonFieldName,
            m_AttributeFieldName,
            m_ValueFieldName,
            m_InfluenceFieldNames,
            m_StartTime,
            static_cast<unsigned int>(m_SampleCountOverride)};
        return {m_GathererType, m_SummaryMode, m_Params,
                             m_PartitionFieldValue, m_SearchKey,
                m_Features,
                bucketGathererInitData};
    }

    std::shared_ptr<CDataGatherer> buildSharedPtr() const {
        return std::make_shared<CDataGatherer>(
            m_GathererType, m_SummaryMode, m_Params, m_SummaryCountFieldName,
            m_PartitionFieldValue, m_PersonFieldName, m_AttributeFieldName,
            m_ValueFieldName, m_InfluenceFieldNames, m_SearchKey, m_Features,
            m_StartTime, m_SampleCountOverride);
    }

    CDataGathererBuilder& partitionFieldValue(std::string_view partitionFieldValue) {
        m_PartitionFieldValue = partitionFieldValue;
        return *this;
    }

    std::shared_ptr<CDataGatherer> buildSharedPtr() const {
        CBucketGatherer::SBucketGathererInitData bucketGathererInitData{
            m_SummaryCountFieldName,
            m_PersonFieldName,
            m_AttributeFieldName,
            m_ValueFieldName,
            m_InfluenceFieldNames,
            m_StartTime,
            static_cast<unsigned int>(m_SampleCountOverride)};
        return std::make_shared<CDataGatherer>(m_GathererType, m_SummaryMode, m_Params,
                             m_PartitionFieldValue, m_SearchKey, m_Features,
                             bucketGathererInitData);
    }

    CDataGathererBuilder& partitionFieldValue(const std::string& partitionFieldValue) {
        m_PartitionFieldValue = partitionFieldValue;
        return *this;
    }

    CDataGathererBuilder& personFieldName(std::string_view personFieldName) {
        m_PersonFieldName = personFieldName;
        return *this;
    }

    CDataGathererBuilder& valueFieldName(std::string_view valueFieldName) {
        m_ValueFieldName = valueFieldName;
        return *this;
    }

    CDataGathererBuilder& influenceFieldNames(const TStrVec& influenceFieldName) {
        m_InfluenceFieldNames = influenceFieldName;
        return *this;
    }

    CDataGathererBuilder& attributeFieldName(std::string_view attributeFieldName) {
        m_AttributeFieldName = attributeFieldName;
        return *this;
    }

    CDataGathererBuilder& gathererType(model_t::EAnalysisCategory gathererType) {
        m_GathererType = gathererType;
        return *this;
    }

    CDataGathererBuilder& sampleCountOverride(std::size_t sampleCount) {
        m_SampleCountOverride = static_cast<int>(sampleCount);
        return *this;
    }

private:
    const TFeatureVec& m_Features;
    const SModelParams& m_Params;
    core_t::TTime m_StartTime;
    const CSearchKey& m_SearchKey;
    model_t::EAnalysisCategory m_GathererType;
    model_t::ESummaryMode m_SummaryMode{model_t::E_None};
    std::string m_SummaryCountFieldName{EMPTY_STRING};
    std::string m_PartitionFieldValue{EMPTY_STRING};
    std::string m_PersonFieldName{EMPTY_STRING};
    std::string m_AttributeFieldName{EMPTY_STRING};
    std::string m_ValueFieldName{EMPTY_STRING};
    TStrVec m_InfluenceFieldNames;
    int m_SampleCountOverride{0};
};
}
}

#endif // INCLUDED_ml_model_ModelTestHelpers_h
