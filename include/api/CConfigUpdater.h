/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CConfigUpdater_h
#define INCLUDED_ml_api_CConfigUpdater_h

#include <model/CAnomalyDetectorModelConfig.h>

#include <api/CFieldConfig.h>
#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! \brief
//! Parse a configuration and apply requested configuration updates.
//!
//! DESCRIPTION:\n
//! When a process is already running and the user requests an
//! update, a control message is being sent with the requested
//! configuration changes. This class is responsible for parsing
//! text with the requested configuration changes and apply them.
//! The changes are expected in an ini type of syntax.
//!
//! IMPLEMENTATION DECISIONS:\n
//! As long as the parsing of the configuration changes is
//! done successfully, the updater tries to apply as many
//! changes as possible even if it fails on a particular
//! change (e.g. unknown stanza name).
//!
class API_EXPORT CConfigUpdater {
public:
    CConfigUpdater(CFieldConfig& fieldConfig, model::CAnomalyDetectorModelConfig& modelConfig);

    //! Update from given config changes
    //! \param config the requested changes in an ini syntax
    bool update(const std::string& config);

private:
    static const std::string MODEL_DEBUG_CONFIG;
    static const std::string DETECTOR_RULES;
    static const std::string DETECTOR_INDEX;
    static const std::string RULES_JSON;
    static const std::string FILTERS;
    static const std::string SCHEDULED_EVENTS;

private:
    CFieldConfig& m_FieldConfig;
    model::CAnomalyDetectorModelConfig& m_ModelConfig;
};
}
}

#endif // INCLUDED_ml_api_CConfigUpdater_h
