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

#include <core/CHashing.h>

#include <model/ImportExport.h>

#include <cstdio>
#include <string>

namespace ml {
namespace model {

//! \brief Enforces term field length constraints with collision prevention.
//!
//! In the anomaly detection domain, term fields (by, over, partition, influencer)
//! are categorical identifiers that must satisfy two invariants:
//! 1. **Bounded Length** - Prevent memory amplification and OOM crashes
//! 2. **Unique Identity** - Distinct field values must remain distinguishable
//!
//! Values exceeding MAX_FIELD_VALUE_LENGTH (256 chars) are transformed using
//! collision-safe truncation:
//!   - Retain PREFIX_LENGTH (239) characters of original value
//!   - Append HASH_SEPARATOR ('$')
//!   - Append HASH_HEX_DIGITS (16) character hex hash of complete original value
//!
//! Format: "<prefix_239_chars>$<hash_16_hex_chars>"
//! Example: "very_long_field_value_that_exceeds_limit_(...)$a1b2c3d4e5f67890"
//!
//! The 256-character limit aligns with Elasticsearch's ignore_above default
//! for keyword fields. The hash suffix ensures data integrity while maintaining
//! human readability (first 239 characters visible) and compatibility with
//! prefix-based filtering.
class MODEL_EXPORT CFieldValueTruncator {
public:
    //! Maximum length for term fields in anomaly detection.
    static constexpr std::size_t MAX_FIELD_VALUE_LENGTH = 256;

    //! Collision prevention format components
    static constexpr char HASH_SEPARATOR = '$';
    static constexpr std::size_t HASH_HEX_DIGITS = 16; // 16 hex chars = full 64-bit hash
    static constexpr std::size_t HASH_SUFFIX_LENGTH = 1 /* separator */ + HASH_HEX_DIGITS; // 17 total

    //! Content prefix length (readable portion after truncation)
    static constexpr std::size_t PREFIX_LENGTH = MAX_FIELD_VALUE_LENGTH - HASH_SUFFIX_LENGTH; // 239

    // Domain invariants (enforced at compile-time)
    static_assert(PREFIX_LENGTH + HASH_SUFFIX_LENGTH == MAX_FIELD_VALUE_LENGTH,
                  "Term field format invariant: prefix + suffix = total length");
    static_assert(PREFIX_LENGTH >= 200,
                  "Readable prefix must be substantial for human comprehension");

    //! Check if a term field value exceeds the domain constraint.
    //! \return true if the value requires length enforcement
    static bool needsTruncation(const std::string& value) {
        return value.size() > MAX_FIELD_VALUE_LENGTH;
    }

    //! Enforce term field length constraint in-place.
    //! Applies collision-safe truncation for values exceeding the limit.
    //! \param[in,out] value Field value to constrain
    //! \return true if truncation was applied, false if already within limit
    static bool truncate(std::string& value) {
        if (needsTruncation(value) == false) {
            return false;
        }

        std::string originalValue = std::move(value);
        value.assign(originalValue, 0, PREFIX_LENGTH);
        appendCollisionPreventionSuffix(originalValue, value);

        return true;
    }

    //! Enforce term field length constraint, returning constrained copy.
    //! \param value Original field value
    //! \return Copy with length constraint enforced
    static std::string truncated(const std::string& value) {
        if (needsTruncation(value) == false) {
            return value;
        }

        std::string result;
        result.reserve(MAX_FIELD_VALUE_LENGTH);
        result.assign(value, 0, PREFIX_LENGTH);
        appendCollisionPreventionSuffix(value, result);

        return result;
    }

private:
    //! \brief Hash encoding for collision prevention.
    //!
    //! Encapsulates the technical details of hash computation and formatting.
    //! Separated from domain logic for clarity and testability.
    struct HashEncoding {
        //! Compute collision-resistant identity hash.
        //! Uses safeMurmurHash64 (endian-neutral) for state persistence safety.
        static std::uint64_t compute(const std::string& value) {
            return core::CHashing::safeMurmurHash64(value.data(),
                                                    static_cast<int>(value.size()),
                                                    0); // Fixed seed for determinism
        }

        //! Format 64-bit hash as zero-padded lowercase hex string.
        //! \param hash The hash value to format
        //! \param[out] buffer Must be at least HASH_HEX_DIGITS + 1 bytes
        //! \return Pointer to null-terminated hex string in buffer
        static const char* toHex(std::uint64_t hash, char* buffer) {
            // %016llx produces 16-char zero-padded lowercase hex (full 64 bits)
            std::snprintf(buffer, HASH_HEX_DIGITS + 1, "%016llx",
                          static_cast<unsigned long long>(hash));
            return buffer;
        }
    };

    //! Append collision-prevention suffix: separator + hash.
    //! \param originalValue Complete untruncated value for hash computation
    //! \param[in,out] prefix Truncated prefix to which suffix is appended
    static void appendCollisionPreventionSuffix(const std::string& originalValue,
                                                std::string& prefix) {
        std::uint64_t identityHash = HashEncoding::compute(originalValue);

        prefix.reserve(MAX_FIELD_VALUE_LENGTH);
        prefix.push_back(HASH_SEPARATOR);

        char hashHexBuffer[HASH_HEX_DIGITS + 1];
        prefix.append(HashEncoding::toHex(identityHash, hashHexBuffer));
    }
};
}
}

#endif // INCLUDED_ml_model_CFieldValueTruncator_h
