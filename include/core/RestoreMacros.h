/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_RestoreMacros_h
#define INCLUDED_ml_core_RestoreMacros_h

namespace ml
{
namespace core
{

#define RESTORE(tag, restore)                                                       \
        if (name == tag)                                                            \
        {                                                                           \
            if ((restore) == false)                                                 \
            {                                                                       \
                LOG_ERROR("Failed to restore " #tag ", got " << traverser.value()); \
                return false;                                                       \
            }                                                                       \
            continue;                                                               \
        }

#define RESTORE_BUILT_IN(tag, target)                                                 \
        if (name == tag)                                                              \
        {                                                                             \
            if (core::CStringUtils::stringToType(traverser.value(), target) == false) \
            {                                                                         \
                LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());   \
                return false;                                                         \
            }                                                                         \
            continue;                                                                 \
        }

#define RESTORE_BOOL(tag, target)                                                     \
        if (name == tag)                                                              \
        {                                                                             \
            int value;                                                                \
            if (core::CStringUtils::stringToType(traverser.value(), value) == false)  \
            {                                                                         \
                LOG_ERROR("Failed to restore " #tag ", got " << traverser.value());   \
                return false;                                                         \
            }                                                                         \
            target = (value != 0);                                                    \
            continue;                                                                 \
        }

#define RESTORE_SETUP_TEARDOWN(tag, setup, restore, teardown)                       \
        if (name == tag)                                                            \
        {                                                                           \
            setup;                                                                  \
            if ((restore) == false)                                                 \
            {                                                                       \
                LOG_ERROR("Failed to restore " #tag ", got " << traverser.value()); \
                return false;                                                       \
            }                                                                       \
            teardown;                                                               \
            continue;                                                               \
        }

#define RESTORE_NO_ERROR(tag, restore) \
        if (name == tag)               \
        {                              \
            restore;                   \
            continue;                  \
        }

}
}

#endif // INCLUDED_ml_core_RestoreMacros_h
