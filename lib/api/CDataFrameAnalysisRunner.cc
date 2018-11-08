/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisRunner.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>

#include <api/CDataFrameAnalysisSpecification.h>

namespace ml {
namespace api {

CDataFrameAnalysisRunner::CDataFrameAnalysisRunner(const CDataFrameAnalysisSpecification& spec)
    : m_Spec{spec}, m_Finished{false}, m_FractionalProgress{0.0} {
}

CDataFrameAnalysisRunner::~CDataFrameAnalysisRunner() {
    this->waitToFinish();
}

void CDataFrameAnalysisRunner::run(core::CDataFrame& frame) {
    if (m_Runner.joinable()) {
        LOG_INFO(<< "Already running analysis");
    } else if (m_Spec.bad()) {
        LOG_ERROR(<< "Bad specification: not running analysis");
        m_FractionalProgress.store(1.0);
        m_Finished.store(true);
    } else {
        m_FractionalProgress.store(0.0);
        m_Finished.store(false);
        m_Runner = std::thread([this, &frame]() { this->runImpl(frame); });
    }
}

void CDataFrameAnalysisRunner::waitToFinish() {
    if (m_Runner.joinable()) {
        m_Runner.join();
    }
}

bool CDataFrameAnalysisRunner::bad() const {
    return m_Bad;
}

bool CDataFrameAnalysisRunner::finished() const {
    return m_Finished.load();
}

double CDataFrameAnalysisRunner::progress() const {
    return m_FractionalProgress.load();
}

const CDataFrameAnalysisSpecification& CDataFrameAnalysisRunner::spec() const {
    return m_Spec;
}

CDataFrameAnalysisRunner::TStrVec CDataFrameAnalysisRunner::errors() const {
    core::CScopedFastLock lock(m_Mutex);
    auto result = m_Errors;
    return result;
}

void CDataFrameAnalysisRunner::setToBad() {
    m_Bad = true;
}

void CDataFrameAnalysisRunner::setToFinished() {
    m_Finished.store(true);
    m_FractionalProgress.store(1.0);
}

void CDataFrameAnalysisRunner::updateProgress(double fractionalProgress) {
    m_FractionalProgress.store(fractionalProgress);
}

void CDataFrameAnalysisRunner::addError(const std::string& error) {
    core::CScopedFastLock lock(m_Mutex);
    m_Errors.push_back(error);
}

core::CFastMutex CDataFrameAnalysisRunner::m_Mutex;
}
}
