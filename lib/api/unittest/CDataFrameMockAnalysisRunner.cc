/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameMockAnalysisRunner.h"

#include <core/CDataFrame.h>
#include <core/CLoopProgress.h>

CDataFrameMockAnalysisRunner::CDataFrameMockAnalysisRunner(const ml::api::CDataFrameAnalysisSpecification& spec)
    : ml::api::CDataFrameAnalysisRunner{spec} {
}

std::size_t CDataFrameMockAnalysisRunner::numberExtraColumns() const {
    return 2;
}

void CDataFrameMockAnalysisRunner::writeOneRow(const ml::core::CDataFrame&,
                                               const TRowRef&,
                                               ml::core::CRapidJsonConcurrentLineWriter&) const {
}

void CDataFrameMockAnalysisRunner::runImpl(ml::core::CDataFrame&) {
    auto progressRecorder = [&](double fractionalProgress) {
        this->state().updateProgress(fractionalProgress);
    };
    ml::core::CLoopProgress progress{31, progressRecorder};
    for (std::size_t i = 0; i < 31; ++i, progress.increment()) {
        std::vector<std::size_t> wait;
        ms_Rng.generateUniformSamples(1, 20, 1, wait);
        std::this_thread::sleep_for(std::chrono::milliseconds(wait[0]));
    }
}

std::size_t CDataFrameMockAnalysisRunner::estimateBookkeepingMemoryUsage(std::size_t,
                                                                         std::size_t,
                                                                         std::size_t,
                                                                         std::size_t) const {
    return 0;
}

ml::test::CRandomNumbers CDataFrameMockAnalysisRunner::ms_Rng;

const std::string& CDataFrameMockAnalysisRunnerFactory::name() const {
    return NAME;
}

CDataFrameMockAnalysisRunnerFactory::TRunnerUPtr
CDataFrameMockAnalysisRunnerFactory::makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec) const {
    return std::make_unique<CDataFrameMockAnalysisRunner>(spec);
}

CDataFrameMockAnalysisRunnerFactory::TRunnerUPtr
CDataFrameMockAnalysisRunnerFactory::makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec,
                                              const rapidjson::Value&) const {
    return std::make_unique<CDataFrameMockAnalysisRunner>(spec);
}

const std::string CDataFrameMockAnalysisRunnerFactory::NAME{"test"};
