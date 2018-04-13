/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CReportWriterTest.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/Constants.h>

#include <config/CDataSummaryStatistics.h>
#include <config/CReportWriter.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;

void CReportWriterTest::testPretty() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+---------------------------------+");
    LOG_DEBUG(<< "|  CReportWriterTest::testPretty  |");
    LOG_DEBUG(<< "+---------------------------------+");

    test::CRandomNumbers rng;

    core_t::TTime startTime = 1459468810;
    core_t::TTime endTime = startTime + 7 * core::constants::DAY;

    std::string fields[] = {std::string("name"), std::string("phylum"),
                            std::string("species"), std::string("code"),
                            std::string("weight")};

    std::string categories1[] = {std::string("Annelida"), std::string("Nematoda"),
                                 std::string("Arthropoda"), std::string("Chordata")};

    std::size_t breaks[] = {0, 6, 10, 13, 20};
    std::string categories2[] = {
        // Annelida
        std::string("Amage auricula"), std::string("Eunice purpurea"),
        std::string("Dorvillea kastjani"), std::string("Dalhousiella carpenteri"),
        std::string("Dysponetus gracilisi"), std::string("Macellicephala incerta"),
        // Nematoda
        std::string("Microlaimus robustidens"), std::string("Theristus longisetosus"),
        std::string("Rhynchonema cemae"), std::string("Contracaecum chubutensis"),
        // Arthropoda
        std::string("black widow"), std::string("Daddy longleg"), std::string("Lobster"),
        // Chordata
        std::string("hag fish"), std::string("hen"), std::string("elephant"),
        std::string("dog"), std::string("shrew"), std::string("weasel"),
        std::string("lemming")};

    TStrVec codes;
    rng.generateWords(6, 2000, codes);

    double weights[] = {0.01,   0.05,      0.1,    0.05, 0.01,  0.5,   0.001,
                        0.0003, 0.01,      0.0004, 1.3,  1.1,   520.0, 1200.0,
                        810.1,  1000000.0, 5334.0, 70.0, 180.0, 100.3};

    config::CDataSummaryStatistics stats1;
    config::CCategoricalDataSummaryStatistics stats2[] = {
        config::CCategoricalDataSummaryStatistics(10),
        config::CCategoricalDataSummaryStatistics(10),
        config::CCategoricalDataSummaryStatistics(10)};
    config::CNumericDataSummaryStatistics stats3(false);

    uint64_t n = 0;

    double lastProgress = 0.0;
    TDoubleVec dt;
    TDoubleVec weight;
    TSizeVec index;
    for (core_t::TTime time = startTime; time < endTime;
         time += static_cast<double>(dt[0])) {
        double progress = static_cast<double>(time - startTime) /
                          static_cast<double>((endTime - startTime));
        if (progress > lastProgress + 0.05) {
            LOG_DEBUG(<< "Processed " << progress * 100.0 << "%");
            lastProgress = progress;
        }
        ++n;

        stats1.add(time);

        rng.generateUniformSamples(0, boost::size(categories1), 1, index);
        const std::string& phylum = categories1[index[0]];
        stats2[0].add(time, phylum);

        rng.generateUniformSamples(breaks[index[0]], breaks[index[0] + 1], 1, index);
        const std::string& species = categories2[index[0]];
        stats2[1].add(time, species);

        double weight_ = weights[index[0]];

        rng.generateUniformSamples(0, codes.size(), 1, index);
        const std::string& code = codes[index[0]];
        stats2[2].add(time, code);

        double range = weight_ > 1.0 ? std::sqrt(weight_) : weight_ * weight_;
        rng.generateUniformSamples(weight_ - range / 2.0, weight_ + range / 2.0, 1, weight);
        stats3.add(time, core::CStringUtils::typeToString(weight[0]));

        rng.generateUniformSamples(0.0, 10.0, 1, dt);
    }

    std::ostringstream o;

    config::CReportWriter writer(o);

    writer.addTotalRecords(n);
    writer.addInvalidRecords(0);
    writer.addFieldStatistics(fields[0], config_t::E_UndeterminedType, stats1);
    writer.addFieldStatistics(fields[1], config_t::E_Categorical, stats2[0]);
    writer.addFieldStatistics(fields[2], config_t::E_Categorical, stats2[1]);
    writer.addFieldStatistics(fields[3], config_t::E_Categorical, stats2[2]);
    writer.addFieldStatistics(fields[4], config_t::E_PositiveReal, stats3);

    writer.write();

    LOG_DEBUG(<< o.str());
}

void CReportWriterTest::testJSON() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+-------------------------------+");
    LOG_DEBUG(<< "|  CReportWriterTest::testJSON  |");
    LOG_DEBUG(<< "+-------------------------------+");
}

CppUnit::Test* CReportWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CReportWriterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CReportWriterTest>(
        "CReportWriterTest::testPretty", &CReportWriterTest::testPretty));

    return suiteOfTests;
}
