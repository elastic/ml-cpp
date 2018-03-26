/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_core_RestoreMacros_h
#define INCLUDED_ml_core_RestoreMacros_h

namespace ml {
namespace core {

#define RESTORE(tag, restore)                                                                                                              \
    if (name == tag) {                                                                                                                     \
        if ((restore) == false) {                                                                                                          \
            LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());                                                            \
            return false;                                                                                                                  \
        }                                                                                                                                  \
        continue;                                                                                                                          \
    }

#define RESTORE_BUILT_IN(tag, target)                                                                                                      \
    if (name == tag) {                                                                                                                     \
        if (core::CStringUtils::stringToType(traverser.value(), target) == false) {                                                        \
            LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());                                                            \
            return false;                                                                                                                  \
        }                                                                                                                                  \
        continue;                                                                                                                          \
    }

#define RESTORE_BOOL(tag, target)                                                                                                          \
    if (name == tag) {                                                                                                                     \
        int value;                                                                                                                         \
        if (core::CStringUtils::stringToType(traverser.value(), value) == false) {                                                         \
            LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());                                                            \
            return false;                                                                                                                  \
        }                                                                                                                                  \
        target = (value != 0);                                                                                                             \
        continue;                                                                                                                          \
    }

#define RESTORE_ENUM(tag, target, enumtype)                                           \
        if (name == tag)                                                              \
        {                                                                             \
            int value;                                                                \
            if (core::CStringUtils::stringToType(traverser.value(), value) == false)  \
            {                                                                         \
                LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());   \
                return false;                                                         \
            }                                                                         \
            target = enumtype(value);                                                 \
            continue;                                                                 \
        }

#define RESTORE_ENUM_CHECKED(tag, target, enumtype, restoreSuccess)                   \
        if (name == tag)                                                              \
        {                                                                             \
            restoreSuccess = true;                                                    \
            RESTORE_ENUM(tag, target, enumtype)                                       \
        }

#define RESTORE_SETUP_TEARDOWN(tag, setup, restore, teardown)                                                                              \
    if (name == tag) {                                                                                                                     \
        setup;                                                                                                                             \
        if ((restore) == false) {                                                                                                          \
            LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());                                                            \
            return false;                                                                                                                  \
        }                                                                                                                                  \
        teardown;                                                                                                                          \
        continue;                                                                                                                          \
    }

#define RESTORE_NO_ERROR(tag, restore)                                                                                                     \
    if (name == tag) {                                                                                                                     \
        restore;                                                                                                                           \
        continue;                                                                                                                          \
    }
}
}

#endif // INCLUDED_ml_core_RestoreMacros_h
