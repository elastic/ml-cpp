/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CStringStoreTest.h"
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

#include <sstream>

using namespace ml;

namespace
{
size_t countBuckets(const std::string &key, const std::string &output)
{
    size_t count = 0;
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(output);
    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT(doc.IsArray());

    const rapidjson::Value &allRecords = doc.GetArray();
    for (auto &r : allRecords.GetArray())
    {
        rapidjson::Value::ConstMemberIterator recordsIt = r.GetObject().FindMember(key);
        if (recordsIt != r.GetObject().MemberEnd())
        {
            ++count;
        }
    }

    return count;
}

core_t::TTime playData(core_t::TTime start, core_t::TTime span, int numBuckets,
                       int numPeople, int numPartitions, int anomaly,
                       api::CAnomalyJob &job)
{
    std::string people[] = { "Elgar", "Holst", "Delius", "Vaughan Williams", "Bliss", "Warlock", "Walton" };
    if (numPeople > 7)
    {
        LOG_ERROR("Too many people: " << numPeople);
        return start;
    }
    std::string partitions[] = { "tuba", "flute", "violin", "triangle", "jew's harp" };
    if (numPartitions > 5)
    {
        LOG_ERROR("Too many partitions: " << numPartitions);
        return start;
    }
    std::stringstream ss;
    ss << "time,notes,composer,instrument\n";
    core_t::TTime t;
    int bucketNum = 0;
    for (t = start; t < start + span * numBuckets; t += span, bucketNum++)
    {
        for (int i = 0; i < numPeople; i++)
        {
            for (int j = 0; j < numPartitions; j++)
            {
                ss << t << "," << (people[i].size() * partitions[j].size()) << ",";
                ss << people[i] << "," << partitions[j] << "\n";
            }
        }
        if (bucketNum == anomaly)
        {
            ss << t << "," << 5564 << "," << people[numPeople - 1] << "," << partitions[numPartitions - 1] << "\n";
        }
    }

    api::CCsvInputParser parser(ss);

    CPPUNIT_ASSERT(parser.readStream(boost::bind(&api::CAnomalyJob::handleRecord,
                                                 &job,
                                                 _1)));

    return t;
}

//! Helper class to look up a string in core::CStoredStringPtr set
struct SLookup
{
    std::size_t operator()(const std::string &key) const
    {
        boost::hash<std::string> hasher;
        return hasher(key);
    }

    bool operator()(const std::string &lhs,
                    const core::CStoredStringPtr &rhs) const
    {
        return lhs == *rhs;
    }
};


} // namespace

bool CStringStoreTest::nameExists(const std::string &string)
{
    model::CStringStore::TStoredStringPtrUSet names = model::CStringStore::names().m_Strings;
    return names.find(string,
                      ::SLookup(),
                      ::SLookup()) != names.end();
}

bool CStringStoreTest::influencerExists(const std::string &string)
{
    model::CStringStore::TStoredStringPtrUSet names = model::CStringStore::influencers().m_Strings;
    return names.find(string,
                      ::SLookup(),
                      ::SLookup()) != names.end();
}

void CStringStoreTest::testPersonStringPruning(void)
{
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("max(notes)");
    clause.push_back("by");
    clause.push_back("composer");
    clause.push_back("partitionfield=instrument");

    CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);
    modelConfig.decayRate(0.001);
    modelConfig.bucketResultsDelay(2);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG("Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE("Setting up job");

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream);

        // There will be one anomaly in this batch, which will be stuck in the
        // results queue.

        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);
        wrappedOutputStream.syncFlush();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), countBuckets("records", outputStrm.str() + "]"));

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // "", "count", "max", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("max"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        time += BUCKET_SPAN * 100;
        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);

        CPPUNIT_ASSERT(job.persistState(adder));
        wrappedOutputStream.syncFlush();

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), countBuckets("records", outputStrm.str() + "]"));
    }

    LOG_DEBUG("Restoring job");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);
        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // "", "count", "max", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("max"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        // play some data in a lot later, to bring about pruning
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 3, 1, 101, job);

        job.finalise();
        CPPUNIT_ASSERT(job.persistState(adder));
    }
    LOG_DEBUG("Restoring job again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);
        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // While the 3 composers from the second partition should have been culled in the prune,
        // their names still exist in the first partition, so will still be in the string store
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("max"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        // Play some more data to cull out the third person
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 2, 2, 101, job);

        job.finalise();
        CPPUNIT_ASSERT(job.persistState(adder));
    }
    LOG_DEBUG("Restoring yet again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);
        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // One composer should have been culled!
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("max"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));
        CPPUNIT_ASSERT(!this->nameExists("Delius"));
    }
}


void CStringStoreTest::testAttributeStringPruning(void)
{
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("dc(notes)");
    clause.push_back("over");
    clause.push_back("composer");
    clause.push_back("partitionfield=instrument");

    CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);
    modelConfig.decayRate(0.001);
    modelConfig.bucketResultsDelay(2);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG("Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE("Setting up job");
        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream);

        // There will be one anomaly in this batch, which will be stuck in the
        // results queue.

        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);
        wrappedOutputStream.syncFlush();
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), countBuckets("records", outputStrm.str() + "]"));

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // "", "count", "distinct_count", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        LOG_DEBUG(core::CContainerPrinter::print(model::CStringStore::names().m_Strings));
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("distinct_count"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        time += BUCKET_SPAN * 100;
        time = playData(time, BUCKET_SPAN, 100, 3, 2, 99, job);

        CPPUNIT_ASSERT(job.persistState(adder));
        wrappedOutputStream.syncFlush();
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), countBuckets("records", outputStrm.str() + "]"));

    }
    LOG_DEBUG("Restoring job");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // "", "count", "distinct_count", "notes", "composer", "instrument", "Elgar", "Holst", "Delius", "flute", "tuba"
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("distinct_count"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        // play some data in a lot later, to bring about pruning
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 3, 1, 101, job);

        job.finalise();
        CPPUNIT_ASSERT(job.persistState(adder));
    }
    LOG_DEBUG("Restoring job again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // While the 3 composers from the second partition should have been culled in the prune,
        // their names still exist in the first partition, so will still be in the string store
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("distinct_count"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("Delius"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));

        // Play some more data to cull out the third person
        time += BUCKET_SPAN * 5000;
        time = playData(time, BUCKET_SPAN, 100, 2, 2, 101, job);

        job.finalise();
        CPPUNIT_ASSERT(job.persistState(adder));
    }
    LOG_DEBUG("Restoring yet again");
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream,
                               api::CAnomalyJob::TPersistCompleteFunc());

        core_t::TTime completeToTime(0);
        CPPUNIT_ASSERT(job.restoreState(searcher, completeToTime));
        adder.clear();

        // No influencers in this configuration
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());

        // One composer should have been culled!
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("distinct_count"));
        CPPUNIT_ASSERT(this->nameExists("notes"));
        CPPUNIT_ASSERT(this->nameExists("composer"));
        CPPUNIT_ASSERT(this->nameExists("instrument"));
        CPPUNIT_ASSERT(this->nameExists("Elgar"));
        CPPUNIT_ASSERT(this->nameExists("Holst"));
        CPPUNIT_ASSERT(this->nameExists("flute"));
        CPPUNIT_ASSERT(this->nameExists("tuba"));
        CPPUNIT_ASSERT(!this->nameExists("Delius"));

    }
}


void CStringStoreTest::testInfluencerStringPruning(void)
{
    core_t::TTime BUCKET_SPAN(10000);
    core_t::TTime time = 100000000;

    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clause;
    clause.push_back("max(notes)");
    clause.push_back("influencerfield=instrument");
    clause.push_back("influencerfield=composer");

    CPPUNIT_ASSERT(fieldConfig.initFromClause(clause));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SPAN);
    modelConfig.bucketResultsDelay(2);

    model::CLimits limits;

    CMockDataAdder adder;
    CMockSearcher searcher(adder);

    LOG_DEBUG("Setting up job");
    // Test that the stringstore entries are pruned correctly on persist/restore
    {
        model::CStringStore::influencers().clearEverythingTestOnly();
        model::CStringStore::names().clearEverythingTestOnly();

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::influencers().m_Strings.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), model::CStringStore::names().m_Strings.size());

        LOG_TRACE("Setting up job");
        std::ostringstream outputStrm;
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream (outputStrm);

        api::CAnomalyJob job("job",
                               limits,
                               fieldConfig,
                               modelConfig,
                               wrappedOutputStream);

        // Play in a few buckets with influencers, and see that they stick around for
        // 3 buckets
        LOG_DEBUG("Running 20 buckets");
        time = playData(time, BUCKET_SPAN, 20, 7, 5, 99, job);

        LOG_TRACE(core::CContainerPrinter::print(model::CStringStore::names().m_Strings));
        LOG_TRACE(core::CContainerPrinter::print(model::CStringStore::influencers().m_Strings));

        CPPUNIT_ASSERT(this->influencerExists("Delius"));
        CPPUNIT_ASSERT(this->influencerExists("Walton"));
        CPPUNIT_ASSERT(this->influencerExists("Holst"));
        CPPUNIT_ASSERT(this->influencerExists("Vaughan Williams"));
        CPPUNIT_ASSERT(this->influencerExists("Warlock"));
        CPPUNIT_ASSERT(this->influencerExists("Bliss"));
        CPPUNIT_ASSERT(this->influencerExists("Elgar"));
        CPPUNIT_ASSERT(this->influencerExists("flute"));
        CPPUNIT_ASSERT(this->influencerExists("tuba"));
        CPPUNIT_ASSERT(this->influencerExists("violin"));
        CPPUNIT_ASSERT(this->influencerExists("triangle"));
        CPPUNIT_ASSERT(this->influencerExists("jew's harp"));

        CPPUNIT_ASSERT(!this->nameExists("Delius"));
        CPPUNIT_ASSERT(!this->nameExists("Walton"));
        CPPUNIT_ASSERT(!this->nameExists("Holst"));
        CPPUNIT_ASSERT(!this->nameExists("Vaughan Williams"));
        CPPUNIT_ASSERT(!this->nameExists("Warlock"));
        CPPUNIT_ASSERT(!this->nameExists("Bliss"));
        CPPUNIT_ASSERT(!this->nameExists("Elgar"));
        CPPUNIT_ASSERT(!this->nameExists("flute"));
        CPPUNIT_ASSERT(!this->nameExists("tuba"));
        CPPUNIT_ASSERT(!this->nameExists("violin"));
        CPPUNIT_ASSERT(!this->nameExists("triangle"));
        CPPUNIT_ASSERT(!this->nameExists("jew's harp"));
        CPPUNIT_ASSERT(this->nameExists("count"));
        CPPUNIT_ASSERT(this->nameExists("max"));
        CPPUNIT_ASSERT(this->nameExists("notes"));

        LOG_DEBUG("Running 3 buckets");
        time = playData(time, BUCKET_SPAN, 3, 3, 2, 99, job);

        CPPUNIT_ASSERT(this->influencerExists("Delius"));
        CPPUNIT_ASSERT(this->influencerExists("Walton"));
        CPPUNIT_ASSERT(this->influencerExists("Holst"));
        CPPUNIT_ASSERT(this->influencerExists("Vaughan Williams"));
        CPPUNIT_ASSERT(this->influencerExists("Warlock"));
        CPPUNIT_ASSERT(this->influencerExists("Bliss"));
        CPPUNIT_ASSERT(this->influencerExists("Elgar"));
        CPPUNIT_ASSERT(this->influencerExists("flute"));
        CPPUNIT_ASSERT(this->influencerExists("tuba"));
        CPPUNIT_ASSERT(this->influencerExists("violin"));
        CPPUNIT_ASSERT(this->influencerExists("triangle"));
        CPPUNIT_ASSERT(this->influencerExists("jew's harp"));

        // They should be purged after 3 buckets
        LOG_DEBUG("Running 2 buckets");
        time = playData(time, BUCKET_SPAN, 2, 3, 2, 99, job);
        CPPUNIT_ASSERT(this->influencerExists("Delius"));
        CPPUNIT_ASSERT(!this->influencerExists("Walton"));
        CPPUNIT_ASSERT(this->influencerExists("Holst"));
        CPPUNIT_ASSERT(!this->influencerExists("Vaughan Williams"));
        CPPUNIT_ASSERT(!this->influencerExists("Warlock"));
        CPPUNIT_ASSERT(!this->influencerExists("Bliss"));
        CPPUNIT_ASSERT(this->influencerExists("Elgar"));
        CPPUNIT_ASSERT(this->influencerExists("flute"));
        CPPUNIT_ASSERT(this->influencerExists("tuba"));
        CPPUNIT_ASSERT(!this->influencerExists("violin"));
        CPPUNIT_ASSERT(!this->influencerExists("triangle"));
        CPPUNIT_ASSERT(!this->influencerExists("jew's harp"));

        // Most should reappear
        LOG_DEBUG("Running 1 bucket");
        time = playData(time, BUCKET_SPAN, 1, 6, 3, 99, job);
        CPPUNIT_ASSERT(this->influencerExists("Delius"));
        CPPUNIT_ASSERT(!this->influencerExists("Walton"));
        CPPUNIT_ASSERT(this->influencerExists("Holst"));
        CPPUNIT_ASSERT(this->influencerExists("Vaughan Williams"));
        CPPUNIT_ASSERT(this->influencerExists("Warlock"));
        CPPUNIT_ASSERT(this->influencerExists("Bliss"));
        CPPUNIT_ASSERT(this->influencerExists("Elgar"));
        CPPUNIT_ASSERT(this->influencerExists("flute"));
        CPPUNIT_ASSERT(this->influencerExists("tuba"));
        CPPUNIT_ASSERT(this->influencerExists("violin"));
        CPPUNIT_ASSERT(!this->influencerExists("triangle"));
        CPPUNIT_ASSERT(!this->influencerExists("jew's harp"));
    }
}


CppUnit::Test* CStringStoreTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStringStoreTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStringStoreTest>(
                                   "CStringStoreTest::testPersonStringPruning",
                                   &CStringStoreTest::testPersonStringPruning) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStringStoreTest>(
                                   "CStringStoreTest::testAttributeStringPruning",
                                   &CStringStoreTest::testAttributeStringPruning) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStringStoreTest>(
                                   "CStringStoreTest::testInfluencerStringPruning",
                                   &CStringStoreTest::testInfluencerStringPruning) );
    return suiteOfTests;
}
