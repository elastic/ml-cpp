/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CMockDataAdder.h"
#include "CMockSearcher.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>
#include <model/CStringStore.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CStringStoreTest)

using namespace ml;

namespace {
size_t countBuckets(const std::string& key, const std::string& output) {
    size_t count = 0;
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(output);
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_TEST_REQUIRE(doc.IsArray());

    const rapidjson::Value& allRecords = doc.GetArray();
    for (auto& r : allRecords.GetArray()) {
        rapidjson::Value::ConstMemberIterator recordsIt = r.GetObject().FindMember(key);
        if (recordsIt != r.GetObject().MemberEnd()) {
            ++count;
        }
    }

    return count;
}

core_t::TTime playData(core_t::TTime start,
                       core_t::TTime span,
                       int numBuckets,
                       int numPeople,
                       int numPartitions,
                       int anomaly,
                       api::CAnomalyJob& job) {
    std::string people[] = {"Elgar", "Holst",   "Delius", "Vaughan Williams",
                            "Bliss", "Warlock", "Walton"};
    if (numPeople > 7) {
        LOG_ERROR(<< "Too many people: " << numPeople);
        return start;
    }
    std::string partitions[] = {"tuba", "flute", "violin", "triangle", "jew's harp"};
    if (numPartitions > 5) {
        LOG_ERROR(<< "Too many partitions: " << numPartitions);
        return start;
    }
    std::stringstream ss;
    ss << "time,notes,composer,instrument\n";
    core_t::TTime t;
    int bucketNum = 0;
    for (t = start; t < start + span * numBuckets; t += span, bucketNum++) {
        for (int i = 0; i < numPeople; i++) {
            for (int j = 0; j < numPartitions; j++) {
                ss << t << "," << (people[i].size() * partitions[j].size()) << ",";
                ss << people[i] << "," << partitions[j] << "\n";
            }
        }
        if (bucketNum == anomaly) {
            ss << t << "," << 5564 << "," << people[numPeople - 1] << ","
               << partitions[numPartitions - 1] << "\n";
        }
    }

    api::CCsvInputParser parser(ss);

    BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(
        std::bind(&api::CAnomalyJob::handleRecord, &job, std::placeholders::_1)));

    return t;
}

//! Helper class to look up a string in core::CStoredStringPtr set
struct SLookup {
    std::size_t operator()(const std::string& key) const {
        boost::hash<std::string> hasher;
        return hasher(key);
    }

    bool operator()(const std::string& lhs, const core::CStoredStringPtr& rhs) const {
        return lhs == *rhs;
    }
};

} // namespace

bool CStringStoreTest::nameExists(const std::string& string) {
    model::CStringStore::TStoredStringPtrUSet names =
        model::CStringStore::names().m_Strings;
    return names.find(string, ::SLookup(), ::SLookup()) != names.end();
}

bool CStringStoreTest::influencerExists(const std::string& string) {
    model::CStringStore::TStoredStringPtrUSet names =
        model::CStringStore::influencers().m_Strings;
    return names.find(string, ::SLookup(), ::SLookup()) != names.end();
}

BOOST_AUTO_TEST_CASE(testPersonStringPruning) {
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("max(notes)");
    clause.push_back("by");
    clause.push_back("composer");
    clause.push_back("partitionfield=instrument");

    BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);
    modelConfig.decayRate(0.001);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG(<< "Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE(<< "Setting up job");

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);
        wrappedOutputStream.syncFlush();

        BOOST_REQUIRE_EQUAL(std::size_t(0), countBuckets("records", outputStrm.str() + "]"));

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // "", "count", "max", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("max"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        time += BUCKET_SPAN * 100;
        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);

        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
        wrappedOutputStream.syncFlush();

        BOOST_REQUIRE_EQUAL(std::size_t(1), countBuckets("records", outputStrm.str() + "]"));
    }

    LOG_DEBUG(<< "Restoring job");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // "", "count", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        // play some data in a lot later, to bring about pruning
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 3, 1, 101, job);

        job.finalise();
        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
    }
    LOG_DEBUG(<< "Restoring job again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // While the 3 composers from the second partition should have been culled in the prune,
        // their names still exist in the first partition, so will still be in the string store
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        // Play some more data to cull out the third person
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 2, 2, 101, job);

        job.finalise();
        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
    }
    LOG_DEBUG(<< "Restoring yet again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // One composer should have been culled!
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));
        BOOST_TEST_REQUIRE(!this->nameExists("Delius"));
    }
}

BOOST_AUTO_TEST_CASE(testAttributeStringPruning) {
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("dc(notes)");
    clause.push_back("over");
    clause.push_back("composer");
    clause.push_back("partitionfield=instrument");

    BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);
    modelConfig.decayRate(0.001);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG(<< "Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE(<< "Setting up job");
        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);
        wrappedOutputStream.syncFlush();
        BOOST_REQUIRE_EQUAL(std::size_t(0), countBuckets("records", outputStrm.str() + "]"));

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // "", "count", "distinct_count", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        LOG_DEBUG(<< core::CContainerPrinter::print(model::CStringStore::names().m_Strings));
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("distinct_count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        time += BUCKET_SPAN * 100;
        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);

        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
        wrappedOutputStream.syncFlush();
        BOOST_REQUIRE_EQUAL(std::size_t(1), countBuckets("records", outputStrm.str() + "]"));
    }
    LOG_DEBUG(<< "Restoring job");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // "", "count", "distinct_count", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        // play some data in a lot later, to bring about pruning
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 3, 1, 101, job);

        job.finalise();
        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
    }
    LOG_DEBUG(<< "Restoring job again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // While the 3 composers from the second partition should have been culled in the prune,
        // their names still exist in the first partition, so will still be in the string store
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));

        // Play some more data to cull out the third person
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 2, 2, 101, job);

        job.finalise();
        BOOST_TEST_REQUIRE(job.persistState(adder, ""));
    }
    LOG_DEBUG(<< "Restoring yet again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        BOOST_TEST_REQUIRE(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());

        // One composer should have been culled!
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));
        BOOST_TEST_REQUIRE(this->nameExists("composer"));
        BOOST_TEST_REQUIRE(this->nameExists("instrument"));
        BOOST_TEST_REQUIRE(this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(this->nameExists("flute"));
        BOOST_TEST_REQUIRE(this->nameExists("tuba"));
        BOOST_TEST_REQUIRE(!this->nameExists("Delius"));
    }
}

BOOST_AUTO_TEST_CASE(testInfluencerStringPruning) {
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("max(notes)");
    clause.push_back("influencerfield=instrument");
    clause.push_back("influencerfield=composer");

    BOOST_TEST_REQUIRE(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG(<< "Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        BOOST_REQUIRE_EQUAL(std::size_t(0),
                          model::CStringStore::influencers().m_Strings.size());
        BOOST_REQUIRE_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE(<< "Setting up job");
        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        // Play in a few buckets with influencers, and see that they stick around for
        // 3 buckets
        LOG_DEBUG(<< "Running 20 buckets");
        time = playData(time, BUCKET_SPAN, 20, 7, 5, 99, job);

        LOG_TRACE(<< core::CContainerPrinter::print(model::CStringStore::names().m_Strings));
        LOG_TRACE(<< core::CContainerPrinter::print(
                      model::CStringStore::influencers().m_Strings));

        BOOST_TEST_REQUIRE(this->influencerExists("Delius"));
        BOOST_TEST_REQUIRE(this->influencerExists("Walton"));
        BOOST_TEST_REQUIRE(this->influencerExists("Holst"));
        BOOST_TEST_REQUIRE(this->influencerExists("Vaughan Williams"));
        BOOST_TEST_REQUIRE(this->influencerExists("Warlock"));
        BOOST_TEST_REQUIRE(this->influencerExists("Bliss"));
        BOOST_TEST_REQUIRE(this->influencerExists("Elgar"));
        BOOST_TEST_REQUIRE(this->influencerExists("flute"));
        BOOST_TEST_REQUIRE(this->influencerExists("tuba"));
        BOOST_TEST_REQUIRE(this->influencerExists("violin"));
        BOOST_TEST_REQUIRE(this->influencerExists("triangle"));
        BOOST_TEST_REQUIRE(this->influencerExists("jew's harp"));

        BOOST_TEST_REQUIRE(!this->nameExists("Delius"));
        BOOST_TEST_REQUIRE(!this->nameExists("Walton"));
        BOOST_TEST_REQUIRE(!this->nameExists("Holst"));
        BOOST_TEST_REQUIRE(!this->nameExists("Vaughan Williams"));
        BOOST_TEST_REQUIRE(!this->nameExists("Warlock"));
        BOOST_TEST_REQUIRE(!this->nameExists("Bliss"));
        BOOST_TEST_REQUIRE(!this->nameExists("Elgar"));
        BOOST_TEST_REQUIRE(!this->nameExists("flute"));
        BOOST_TEST_REQUIRE(!this->nameExists("tuba"));
        BOOST_TEST_REQUIRE(!this->nameExists("violin"));
        BOOST_TEST_REQUIRE(!this->nameExists("triangle"));
        BOOST_TEST_REQUIRE(!this->nameExists("jew's harp"));
        BOOST_TEST_REQUIRE(this->nameExists("count"));
        BOOST_TEST_REQUIRE(this->nameExists("max"));
        BOOST_TEST_REQUIRE(this->nameExists("notes"));

        LOG_DEBUG(<< "Running 3 buckets");
        time = playData(time, BUCKET_SPAN, 3, 3, 2, 99, job);

        BOOST_TEST_REQUIRE(this->influencerExists("Delius"));
        BOOST_TEST_REQUIRE(this->influencerExists("Walton"));
        BOOST_TEST_REQUIRE(this->influencerExists("Holst"));
        BOOST_TEST_REQUIRE(this->influencerExists("Vaughan Williams"));
        BOOST_TEST_REQUIRE(this->influencerExists("Warlock"));
        BOOST_TEST_REQUIRE(this->influencerExists("Bliss"));
        BOOST_TEST_REQUIRE(this->influencerExists("Elgar"));
        BOOST_TEST_REQUIRE(this->influencerExists("flute"));
        BOOST_TEST_REQUIRE(this->influencerExists("tuba"));
        BOOST_TEST_REQUIRE(this->influencerExists("violin"));
        BOOST_TEST_REQUIRE(this->influencerExists("triangle"));
        BOOST_TEST_REQUIRE(this->influencerExists("jew's harp"));

        // They should be purged after 3 buckets
        LOG_DEBUG(<< "Running 2 buckets");
        time = playData(time, BUCKET_SPAN, 2, 3, 2, 99, job);
        BOOST_TEST_REQUIRE(this->influencerExists("Delius"));
        BOOST_TEST_REQUIRE(!this->influencerExists("Walton"));
        BOOST_TEST_REQUIRE(this->influencerExists("Holst"));
        BOOST_TEST_REQUIRE(!this->influencerExists("Vaughan Williams"));
        BOOST_TEST_REQUIRE(!this->influencerExists("Warlock"));
        BOOST_TEST_REQUIRE(!this->influencerExists("Bliss"));
        BOOST_TEST_REQUIRE(this->influencerExists("Elgar"));
        BOOST_TEST_REQUIRE(this->influencerExists("flute"));
        BOOST_TEST_REQUIRE(this->influencerExists("tuba"));
        BOOST_TEST_REQUIRE(!this->influencerExists("violin"));
        BOOST_TEST_REQUIRE(!this->influencerExists("triangle"));
        BOOST_TEST_REQUIRE(!this->influencerExists("jew's harp"));

        // Most should reappear
        LOG_DEBUG(<< "Running 1 bucket");
        time = playData(time, BUCKET_SPAN, 1, 6, 3, 99, job);
        BOOST_TEST_REQUIRE(this->influencerExists("Delius"));
        BOOST_TEST_REQUIRE(!this->influencerExists("Walton"));
        BOOST_TEST_REQUIRE(this->influencerExists("Holst"));
        BOOST_TEST_REQUIRE(this->influencerExists("Vaughan Williams"));
        BOOST_TEST_REQUIRE(this->influencerExists("Warlock"));
        BOOST_TEST_REQUIRE(this->influencerExists("Bliss"));
        BOOST_TEST_REQUIRE(this->influencerExists("Elgar"));
        BOOST_TEST_REQUIRE(this->influencerExists("flute"));
        BOOST_TEST_REQUIRE(this->influencerExists("tuba"));
        BOOST_TEST_REQUIRE(this->influencerExists("violin"));
        BOOST_TEST_REQUIRE(!this->influencerExists("triangle"));
        BOOST_TEST_REQUIRE(!this->influencerExists("jew's harp"));
    }
}

BOOST_AUTO_TEST_SUITE_END()
