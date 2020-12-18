/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CConfigUpdater.h>

#include <model/CLimits.h>

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace ml {
namespace api {

const std::string CConfigUpdater::MODEL_DEBUG_CONFIG("modelPlotConfig");
const std::string CConfigUpdater::DETECTOR_RULES("detectorRules");
const std::string CConfigUpdater::DETECTOR_INDEX("detectorIndex");
const std::string CConfigUpdater::RULES_JSON("rulesJson");
const std::string CConfigUpdater::FILTERS("filters");
const std::string CConfigUpdater::SCHEDULED_EVENTS("scheduledEvents");

CConfigUpdater::CConfigUpdater(CAnomalyJobConfig& jobConfig,
                               model::CAnomalyDetectorModelConfig& modelConfig)
    : m_JobConfig(jobConfig), m_ModelConfig(modelConfig) {
}

bool CConfigUpdater::update(const std::string& config) {
    boost::property_tree::ptree propTree;

    try {
        std::istringstream strm(config);
        boost::property_tree::ini_parser::read_ini(strm, propTree);
    } catch (boost::property_tree::ptree_error& e) {
        LOG_ERROR(<< "Error parsing config from '" << config << "' : " << e.what());
        return false;
    }

    for (boost::property_tree::ptree::const_iterator stanzaItr = propTree.begin();
         stanzaItr != propTree.end(); ++stanzaItr) {
        const std::string& stanzaName = stanzaItr->first;
        const boost::property_tree::ptree& subTree = stanzaItr->second;

        if (stanzaName == MODEL_DEBUG_CONFIG) {
            if (m_ModelConfig.configureModelPlot(subTree) == false) {
                LOG_ERROR(<< "Could not parse modelPlotConfig");
                return false;
            }
        } else if (stanzaName == DETECTOR_RULES) {
            std::string detectorIndexString = subTree.get(DETECTOR_INDEX, std::string());
            int detectorIndex;
            if (core::CStringUtils::stringToType(detectorIndexString, detectorIndex) == false) {
                LOG_ERROR(<< "Invalid detector index: " << detectorIndexString);
                return false;
            }
            std::string rulesJson = subTree.get(RULES_JSON, std::string());
            if (m_JobConfig.analysisConfig().parseRules(detectorIndex, rulesJson) == false) {
                LOG_ERROR(<< "Failed to update detector rules for detector: " << detectorIndex);
                return false;
            }
        } else if (stanzaName == FILTERS) {
            // TODO: Move to JSON format for config updates.
            if (m_JobConfig.analysisConfig().updateFilters(subTree) == false) {
                LOG_ERROR(<< "Failed to update filters");
                return false;
            }
        } else if (stanzaName == SCHEDULED_EVENTS) {
            // TODO: Move to JSON format for config updates.
            if (m_JobConfig.analysisConfig().updateScheduledEvents(subTree) == false) {
                LOG_ERROR(<< "Failed to update scheduled events");
                return false;
            }
        } else {
            LOG_WARN(<< "Ignoring unknown update config stanza: " << stanzaName);
            return false;
        }
    }

    return true;
}
}
}
