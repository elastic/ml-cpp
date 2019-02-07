/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameMockAnalysisRunner_h
#define INCLUDED_CDataFrameMockAnalysisRunner_h

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <test/CRandomNumbers.h>

#include <functional>

class CDataFrameMockAnalysisRunner : public ml::api::CDataFrameAnalysisRunner {
public:
    CDataFrameMockAnalysisRunner(const ml::api::CDataFrameAnalysisSpecification& spec);

    virtual std::size_t numberOfPartitions() const;
    virtual std::size_t numberExtraColumns() const;
    virtual void writeOneRow(TRowRef, ml::core::CRapidJsonConcurrentLineWriter&) const;

protected:
    void runImpl(ml::core::CDataFrame&);

private:
    virtual std::size_t
        estimateBookkeepingMemoryUsage(std::size_t, std::size_t, std::size_t) const;

private:
    static ml::test::CRandomNumbers ms_Rng;
};

class CDataFrameMockAnalysisRunnerFactory : public ml::api::CDataFrameAnalysisRunnerFactory {
public:
    virtual const char* name() const;

private:
    virtual TRunnerUPtr makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec) const;
    virtual TRunnerUPtr makeImpl(const ml::api::CDataFrameAnalysisSpecification& spec,
                                 const rapidjson::Value&) const;
};

#endif // INCLUDED_CDataFrameMockAnalysisRunner_h
