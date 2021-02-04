/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_RestoreMacros_h
#define INCLUDED_ml_core_RestoreMacros_h

namespace ml {
namespace core {

#define VIOLATES_INVARIANT(lhs, op, rhs)                                       \
    do {                                                                       \
        if (lhs op rhs) {                                                      \
            LOG_ABORT(<< "Invariance check failed: " #lhs " " #op " " #rhs "." \
                      << " [" << lhs << " " << #op << " " << rhs << "]");      \
        }                                                                      \
    } while (0)

#define VIOLATES_INVARIANT_NO_EVALUATION(lhs, op, rhs)                           \
    do {                                                                         \
        if (lhs op rhs) {                                                        \
            LOG_ABORT(<< "Invariance check failed: " #lhs " " #op " " #rhs "."); \
        }                                                                        \
    } while (0)

#define RESTORE(tag, restore)                                                          \
    if (name == tag) {                                                                 \
        if ((restore) == false) {                                                      \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
        continue;                                                                      \
    }

#define RESTORE_NO_LOOP(tag, restore)                                                  \
    if (name == tag) {                                                                 \
        if ((restore) == false) {                                                      \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
    }

#define RESTORE_BUILT_IN(tag, target)                                                  \
    if (name == tag) {                                                                 \
        if (core::CStringUtils::stringToType(traverser.value(), target) == false) {    \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
        continue;                                                                      \
    }

#define RESTORE_BOOL(tag, target)                                                      \
    if (name == tag) {                                                                 \
        int value;                                                                     \
        if (core::CStringUtils::stringToType(traverser.value(), value) == false) {     \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
        target = (value != 0);                                                         \
        continue;                                                                      \
    }

#define RESTORE_ENUM(tag, target, enumtype)                                            \
    if (name == tag) {                                                                 \
        int value;                                                                     \
        if (core::CStringUtils::stringToType(traverser.value(), value) == false) {     \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
        target = enumtype(value);                                                      \
        continue;                                                                      \
    }

#define RESTORE_ENUM_CHECKED(tag, target, enumtype, restoreSuccess)            \
    if (name == tag) {                                                         \
        restoreSuccess = true;                                                 \
        RESTORE_ENUM(tag, target, enumtype)                                    \
    }

#define RESTORE_SETUP_TEARDOWN(tag, setup, restore, teardown)                          \
    if (name == tag) {                                                                 \
        setup;                                                                         \
        if ((restore) == false) {                                                      \
            if (traverser.value().empty()) {                                           \
                LOG_ERROR(<< "Failed to restore " #tag);                               \
            } else {                                                                   \
                LOG_ERROR(<< "Failed to restore " #tag ", got " << traverser.value()); \
            }                                                                          \
            return false;                                                              \
        }                                                                              \
        teardown;                                                                      \
        continue;                                                                      \
    }

#define RESTORE_NO_ERROR(tag, restore)                                         \
    if (name == tag) {                                                         \
        restore;                                                               \
        continue;                                                              \
    }
}
}

#endif // INCLUDED_ml_core_RestoreMacros_h
