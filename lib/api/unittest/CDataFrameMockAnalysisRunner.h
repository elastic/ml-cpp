/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameMockAnalysisRunner_h
#define INCLUDED_CDataFrameMockAnalysisRunner_h

#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <test/CRandomNumbers.h>

#include <functional>

class CDataFrameMockAnalysisState final : public ml::api::CDataFrameAnalysisInstrumentation {
protected:
    ml::counter_t::ECounterTypes memoryCounterType() override;
};

class CDataFrameMockAnalysisRunner final : public ml::api::CDataFrameAnalysisRunner {
public:
    CDataFrameMockAnalysisRunner(const ml::api::CDataFrameAnalysisSpecification& spec);

    std::size_t numberExtraColumns() const override;
    void writeOneRow(const ml::core::CDataFrame&,
                     const TRowRef&,
                     ml::core::CRapidJsonConcurrentLineWriter&) const override;

    const ml::api::CDataFrameAnalysisInstrumentation& instrumentation() const override;
    ml::api::CDataFrameAnalysisInstrumentation& instrumentation() override;

private:
    void runImpl(ml::core::CDataFrame&) override;
    std::size_t estimateBookkeepingMemoryUsage(std::size_t,
                                               std::size_t,
                                               std::size_t,
                                               std::size_t) const override;

private:
    static ml::test::CRandomNumbers ms_Rng;
    CDataFrameMockAnalysisState m_Instrumentation;
};

class CDataFrameMockAnalysisRunnerFactory final : public ml::api::CDataFrameAnalysisRunnerFactory {
public:
    const std::string& name() const override;

private:
    static const std::string NAME;

private:
    TRunnerUPtr makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec) const override;
    TRunnerUPtr makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec,
                         const rapidjson::Value&) const override;
};

#endif // INCLUDED_CDataFrameMockAnalysisRunner_h
