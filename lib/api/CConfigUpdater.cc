/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CConfigUpdater.h>

#include <model/CLimits.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>

namespace ml {
namespace api {

CConfigUpdater::CConfigUpdater(CAnomalyJobConfig& jobConfig,
                               model::CAnomalyDetectorModelConfig& modelConfig)
    : m_JobConfig(jobConfig), m_ModelConfig(modelConfig) {
}

bool CConfigUpdater::update(const std::string& json) {
    rapidjson::Document doc;
    if (doc.Parse<0>(json.c_str()).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing pattern set from JSON: " +
                         std::string(rapidjson::GetParseError_En(doc.GetParseError())));
        return false;
    }

    if (doc.IsObject() == false) {
        LOG_ERROR(<< "Input error: expected JSON object but input was '" << json
                  << "'. Please report this problem.");
        return false;
    }

    if (doc.HasMember(CAnomalyJobConfig::MODEL_PLOT_CONFIG)) {
        if (doc[CAnomalyJobConfig::MODEL_PLOT_CONFIG].IsObject() == false) {
            LOG_ERROR(<< "Input error: expected "
                      << CAnomalyJobConfig::MODEL_PLOT_CONFIG
                      << " to be JSON object but input was '"
                      << json
                      << "'. Please report this problem.");
            return false;
        }
        const rapidjson::Value& value = doc[CAnomalyJobConfig::MODEL_PLOT_CONFIG];

        m_JobConfig.modelPlotConfig().parse(value);
        const ml::api::CAnomalyJobConfig::CModelPlotConfig& modelPlotConfig =
            m_JobConfig.modelPlotConfig();
        m_ModelConfig.configureModelPlot(modelPlotConfig.enabled(),
                                         modelPlotConfig.annotationsEnabled(),
                                         modelPlotConfig.terms());
    } else if (doc.HasMember(CAnomalyJobConfig::FILTERS)) {
        if (m_JobConfig.parseFilterConfig(json) == false) {
            LOG_ERROR(<< "Failed to parse filter config update: " << json);
            return false;
        }
        m_JobConfig.initRuleFilters();
    } else if (doc.HasMember(CAnomalyJobConfig::EVENTS)) {
        if (m_JobConfig.parseEventConfig(json) == false) {
            LOG_ERROR(<< "Failed to parse events config update: " << json);
            return false;
        }
        m_JobConfig.initScheduledEvents();
    } else if (doc.HasMember(CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_RULES)) {
        return m_JobConfig.analysisConfig().parseRulesUpdate(
            doc[CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_RULES]);
    } else {
        LOG_ERROR(<< "Unexpected JSON update message: " << json);
        return false;
    }
    return true;
}
}
}
