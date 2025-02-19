/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <api/CConfigUpdater.h>

#include <model/CLimits.h>

#include <boost/json.hpp>

namespace json = boost::json;

namespace ml {
namespace api {

CConfigUpdater::CConfigUpdater(CAnomalyJobConfig& jobConfig,
                               model::CAnomalyDetectorModelConfig& modelConfig)
    : m_JobConfig(jobConfig), m_ModelConfig(modelConfig) {
}

bool CConfigUpdater::update(const std::string& json) {
    json::parser p;
    json::error_code ec;
    p.write_some(json, ec);
    if (ec) {
        LOG_ERROR(<< "An error occurred while parsing pattern set from JSON: "
                  << ec.message());
        return false;
    }
    json::value doc = p.release();
    if (doc.is_object() == false) {
        LOG_ERROR(<< "Input error: expected JSON object but input was '" << json
                  << "'. Please report this problem.");
        return false;
    }

    json::object obj = doc.as_object();

    for (const auto& kv : obj) {
        if (kv.key() == CAnomalyJobConfig::MODEL_PLOT_CONFIG) {
            LOG_INFO(<< "Updating model plot config");

            if (kv.value().is_object() == false) {
                LOG_ERROR(<< "Input error: expected " << CAnomalyJobConfig::MODEL_PLOT_CONFIG
                          << " to be JSON object but input was '" << json
                          << "'. Please report this problem.");
                return false;
            }

            m_JobConfig.modelPlotConfig().parse(kv.value());
            const ml::api::CAnomalyJobConfig::CModelPlotConfig& modelPlotConfig =
                m_JobConfig.modelPlotConfig();
            m_ModelConfig.configureModelPlot(modelPlotConfig.enabled(),
                                             modelPlotConfig.annotationsEnabled(),
                                             modelPlotConfig.terms());
        } else if (kv.key() == CAnomalyJobConfig::FILTERS) {
            LOG_INFO(<< "Updating filters config");

            if (m_JobConfig.parseFilterConfig(json) == false) {
                LOG_ERROR(<< "Failed to parse filter config update: " << json);
                return false;
            }
            LOG_INFO(<< "Calling m_JobConfig.initRuleFilters");

            m_JobConfig.initRuleFilters();

            LOG_INFO(<< "Done calling m_JobConfig.initRuleFilters");

        } else if (kv.key() == CAnomalyJobConfig::EVENTS) {
            LOG_INFO(<< "Updating events config");

            if (m_JobConfig.parseEventConfig(json) == false) {
                LOG_ERROR(<< "Failed to parse events config update: " << json);
                return false;
            }
            m_JobConfig.initScheduledEvents();
        } else if (kv.key() == CAnomalyJobConfig::CAnalysisConfig::CDetectorConfig::DETECTOR_RULES) {
            LOG_INFO(<< "Updating detector rules config");
            return m_JobConfig.analysisConfig().parseRulesUpdate(kv.value());
        } else {
            LOG_ERROR(<< "Unexpected JSON update message: " << json);
            return false;
        }
    }

    return true;
}
}
}
