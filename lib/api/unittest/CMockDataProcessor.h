/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CMockDataProcessor_h
#define INCLUDED_ml_api_CMockDataProcessor_h

#include <core/CoreTypes.h>

#include <api/CDataProcessor.h>

#include <cstdint>
#include <string>

namespace ml {
namespace api {
class COutputHandler;
}
}

//! \brief
//! Mock object for unit tests
//!
//! DESCRIPTION:\n
//! Mock object for testing the OutputChainer class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only the minimal set of required functions are implemented.
//!
class CMockDataProcessor final : public ml::api::CDataProcessor {
public:
    CMockDataProcessor(ml::api::COutputHandler& outputHandler);

    bool handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) override;

    bool handleRecord(const TStrStrUMap& dataRowFields) {
        return this->handleRecord(dataRowFields, TOptionalTime{});
    }

    void finalise() override;

    bool isPersistenceNeeded(const std::string& description) const override;

    //! Restore previously saved state
    bool restoreState(ml::core::CDataSearcher& restoreSearcher,
                      ml::core_t::TTime& completeToTime) override;

    //! Persist current state
    bool persistStateInForeground(ml::core::CDataAdder& persister,
                                  const std::string& descriptionPrefix) override;

    //! How many records did we handle?
    std::uint64_t numRecordsHandled() const override;

private:
    ml::api::COutputHandler& m_OutputHandler;

    //! Empty field overrides
    TStrStrUMap m_FieldOverrides;

    uint64_t m_NumRecordsHandled;

    bool m_WriteFieldNames;
};

#endif // INCLUDED_ml_api_CMockDataProcessor_h
