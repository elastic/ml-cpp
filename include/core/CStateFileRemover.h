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
#ifndef INCLUDED_ml_core_CStateFileRemover_h
#define INCLUDED_ml_core_CStateFileRemover_h

#include <core/CLogger.h>

#include <cstdlib>

namespace ml {
namespace core {

//! \brief
//! Ensures that deletion of state files occurs even on process failure.
//!
//! DESCRIPTION:\n
//! A helper to ensure that quantiles state files always get deleted on failure.
//! They may also be explicitly be deleted on request as well but that is handled separately by the happy path.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable or moveable. No default construction.
class CStateFileRemover {
public:
    CStateFileRemover() = delete;
    CStateFileRemover(const CStateFileRemover&) = delete;
    CStateFileRemover& operator=(const CStateFileRemover&) = delete;
    CStateFileRemover(CStateFileRemover&&) = delete;
    CStateFileRemover& operator=(CStateFileRemover&&) = delete;
    explicit CStateFileRemover(const std::string& quantilesStateFile,
                               bool deleteStateFiles = false)
        : m_QuantilesStateFile{quantilesStateFile}, m_DeleteStateFiles{deleteStateFiles} {}
    ~CStateFileRemover() {
        // Always delete quantiles state files if requested to do so, even on failure,
        // else we run the risk of filling the disk after repeated failures.
        // They should still exist in ES should they need to be examined.
        if (m_QuantilesStateFile.empty() || m_DeleteStateFiles == false) {
            return;
        }
        LOG_DEBUG(<< "Deleting quantiles state file '" << m_QuantilesStateFile << "'");
        if (std::remove(m_QuantilesStateFile.c_str()) != 0) {
            LOG_WARN(<< "Failed to delete quantiles state file '"
                     << m_QuantilesStateFile << "': " << strerror(errno));
        }
    }

private:
    std::string m_QuantilesStateFile;
    bool m_DeleteStateFiles{false};
};
}
}

#endif // INCLUDED_ml_core_CStateFileRemover_h
