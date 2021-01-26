/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CConfigUpdater_h
#define INCLUDED_ml_api_CConfigUpdater_h

#include <model/CAnomalyDetectorModelConfig.h>

#include <api/CAnomalyJobConfig.h>
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
//! The changes are expected in a JSON document.
//!
//! IMPLEMENTATION DECISIONS:\n
//! As long as the parsing of the configuration changes is
//! done successfully, the updater tries to apply as many
//! changes as possible even if it fails on a particular
//! change.
//!
class API_EXPORT CConfigUpdater {
public:
    CConfigUpdater(CAnomalyJobConfig& jobConfig,
                   model::CAnomalyDetectorModelConfig& modelConfig);

    //! Update from given config changes
    //! \param config the requested changes in an ini syntax
    bool update(const std::string& config);

private:
    CAnomalyJobConfig& m_JobConfig;
    model::CAnomalyDetectorModelConfig& m_ModelConfig;
};
}
}

#endif // INCLUDED_ml_api_CConfigUpdater_h
