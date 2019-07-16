/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CLimits.h>

#include <core/CStreamUtils.h>

#include <model/CResourceMonitor.h>

#include <boost/property_tree/ini_parser.hpp>

#include <fstream>

namespace ml {
namespace model {

// Initialise statics
const size_t CLimits::DEFAULT_AUTOCONFIG_EVENTS(10000);
const size_t CLimits::DEFAULT_ANOMALY_MAX_FIELD_VALUES(100000);
const size_t CLimits::DEFAULT_ANOMALY_MAX_TIME_BUCKETS(1000000);
const size_t CLimits::DEFAULT_RESULTS_MAX_EXAMPLES(4);
// The probability threshold is stored as a percentage in the config file
const double CLimits::DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD(3.5);

CLimits::CLimits(bool persistenceInForeground, double byteLimitMargin)
    : m_AutoConfigEvents(DEFAULT_AUTOCONFIG_EVENTS),
      m_AnomalyMaxTimeBuckets(DEFAULT_ANOMALY_MAX_TIME_BUCKETS),
      m_MaxExamples(DEFAULT_RESULTS_MAX_EXAMPLES),
      m_UnusualProbabilityThreshold(DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD),
      m_MemoryLimitMB(CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB),
      m_ResourceMonitor(persistenceInForeground, byteLimitMargin) {
}

bool CLimits::init(const std::string& configFile) {
    boost::property_tree::ptree propTree;
    try {
        std::ifstream strm(configFile.c_str());
        if (!strm.is_open()) {
            LOG_ERROR(<< "Error opening config file " << configFile);
            return false;
        }
        core::CStreamUtils::skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, propTree);
    } catch (boost::property_tree::ptree_error& e) {
        LOG_ERROR(<< "Error reading config file " << configFile << " : " << e.what());
        return false;
    }

    if (this->processSetting(propTree, "autoconfig.events",
                             DEFAULT_AUTOCONFIG_EVENTS, m_AutoConfigEvents) == false ||
        this->processSetting(propTree, "anomaly.maxtimebuckets", DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                             m_AnomalyMaxTimeBuckets) == false ||
        this->processSetting(propTree, "results.maxexamples",
                             DEFAULT_RESULTS_MAX_EXAMPLES, m_MaxExamples) == false ||
        this->processSetting(propTree, "results.unusualprobabilitythreshold",
                             DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD,
                             m_UnusualProbabilityThreshold) == false ||
        this->processSetting(propTree, "memory.modelmemorylimit", CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB,
                             m_MemoryLimitMB) == false) {
        LOG_ERROR(<< "Error processing config file " << configFile);
        return false;
    }

    m_ResourceMonitor.memoryLimit(m_MemoryLimitMB);

    return true;
}

size_t CLimits::autoConfigEvents() const {
    return m_AutoConfigEvents;
}

size_t CLimits::anomalyMaxTimeBuckets() const {
    return m_AnomalyMaxTimeBuckets;
}

size_t CLimits::maxExamples() const {
    return m_MaxExamples;
}

double CLimits::unusualProbabilityThreshold() const {
    return m_UnusualProbabilityThreshold / 100.0;
}

size_t CLimits::memoryLimitMB() const {
    return m_MemoryLimitMB;
}

CResourceMonitor& CLimits::resourceMonitor() {
    return m_ResourceMonitor;
}
}
}
