/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <model/CLimits.h>
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


CLimits::CLimits() : m_AutoConfigEvents(DEFAULT_AUTOCONFIG_EVENTS),
                     m_AnomalyMaxTimeBuckets(DEFAULT_ANOMALY_MAX_TIME_BUCKETS),
                     m_MaxExamples(DEFAULT_RESULTS_MAX_EXAMPLES),
                     m_UnusualProbabilityThreshold(DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD),
                     m_MemoryLimitMB(CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB),
                     m_ResourceMonitor() {}

CLimits::~CLimits() {}

bool CLimits::init(const std::string &configFile) {
    boost::property_tree::ptree propTree;
    try {
        std::ifstream strm(configFile.c_str());
        if (!strm.is_open()) {
            LOG_ERROR("Error opening config file " << configFile);
            return false;
        }
        this->skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, propTree);
    } catch (boost::property_tree::ptree_error &e) {
        LOG_ERROR("Error reading config file " << configFile <<
                  " : " << e.what());
        return false;
    }

    if (this->processSetting(propTree,
                             "autoconfig.events",
                             DEFAULT_AUTOCONFIG_EVENTS,
                             m_AutoConfigEvents) == false ||
        this->processSetting(propTree,
                             "anomaly.maxtimebuckets",
                             DEFAULT_ANOMALY_MAX_TIME_BUCKETS,
                             m_AnomalyMaxTimeBuckets) == false ||
        this->processSetting(propTree,
                             "results.maxexamples",
                             DEFAULT_RESULTS_MAX_EXAMPLES,
                             m_MaxExamples) == false ||
        this->processSetting(propTree,
                             "results.unusualprobabilitythreshold",
                             DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD,
                             m_UnusualProbabilityThreshold) == false ||
        this->processSetting(propTree,
                             "memory.modelmemorylimit",
                             CResourceMonitor::DEFAULT_MEMORY_LIMIT_MB,
                             m_MemoryLimitMB) == false) {
        LOG_ERROR("Error processing config file " << configFile);
        return false;
    }

    m_ResourceMonitor.memoryLimit(m_MemoryLimitMB);

    return true;
}

size_t CLimits::autoConfigEvents(void) const {
    return m_AutoConfigEvents;
}

size_t CLimits::anomalyMaxTimeBuckets(void) const {
    return m_AnomalyMaxTimeBuckets;
}

size_t CLimits::maxExamples(void) const {
    return m_MaxExamples;
}

double CLimits::unusualProbabilityThreshold(void) const {
    return m_UnusualProbabilityThreshold / 100.0;
}

size_t CLimits::memoryLimitMB(void) const {
    return m_MemoryLimitMB;
}

CResourceMonitor &CLimits::resourceMonitor(void) {
    return m_ResourceMonitor;
}

void CLimits::skipUtf8Bom(std::ifstream &strm) {
    if (strm.tellg() != std::streampos(0)) {
        return;
    }
    std::ios_base::iostate origState(strm.rdstate());
    // The 3 bytes 0xEF, 0xBB, 0xBF form a UTF-8 byte order marker (BOM)
    if (strm.get() == 0xEF) {
        if (strm.get() == 0xBB) {
            if (strm.get() == 0xBF) {
                LOG_DEBUG("Skipping UTF-8 BOM");
                return;
            }
        }
    }
    // Set the stream state back to how it was originally so subsequent code can
    // report errors
    strm.clear(origState);
    // There was no BOM, so seek back to the beginning of the file
    strm.seekg(0);
}


}
}

