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
#ifndef INCLUDED_ml_model_CFieldValueTruncator_h
#define INCLUDED_ml_model_CFieldValueTruncator_h

#include <model/ImportExport.h>

#include <string>

namespace ml {
namespace model {

//! \brief Truncates field values to prevent memory amplification.
//!
//! DESCRIPTION:\n
//! Field values (by, over, partition, influencer) are term fields
//! in the anomaly detection domain. They are categorical identifiers,
//! not free text. Their length must be bounded to prevent excessive
//! memory consumption that could cause the autodetect process to crash.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The limit of 256 characters aligns with Elasticsearch's
//! ignore_above default for keyword fields. This is sufficient for
//! meaningful anomaly detection field values while preventing memory
//! amplification from extremely long strings (e.g., 77K+ characters)
//! that have been observed to crash the autodetect process.
class MODEL_EXPORT CFieldValueTruncator {
public:
    //! Maximum length for analysis term fields (by, over, partition, influencer).
    //! Values longer than this are truncated to prevent excessive memory usage.
    static constexpr std::size_t MAX_FIELD_VALUE_LENGTH = 256;

    //! In-place truncation of a field value.
    //! \return true if truncation occurred, false if value was within limit.
    static bool truncate(std::string& value) {
        if (value.size() <= MAX_FIELD_VALUE_LENGTH) {
            return false;
        }
        value.resize(MAX_FIELD_VALUE_LENGTH);
        return true;
    }

    //! Returns a truncated copy of the field value. Original unchanged.
    static std::string truncated(const std::string& value) {
        if (value.size() <= MAX_FIELD_VALUE_LENGTH) {
            return value;
        }
        return value.substr(0, MAX_FIELD_VALUE_LENGTH);
    }
};
}
}

#endif // INCLUDED_ml_model_CFieldValueTruncator_h
