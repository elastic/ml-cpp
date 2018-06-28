/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CAutoconfigurerFieldRolePenalties_h
#define INCLUDED_ml_config_CAutoconfigurerFieldRolePenalties_h

#include <core/CNonCopyable.h>

#include <config/ImportExport.h>

#include <memory>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CPenalty;

//! \brief Defines the functions for penalizing field roles.
//!
//! DESCRIPTION:\n
//! This defines the penalties for using a field as either a function argument,
//! "by", "over" or "partition" field in a detector.
//!
//! IMPLEMENTATION:\n
//! This provides a single definition point for a logical group of penalties
//! and has been factored into its own class to avoid CAutoconfigurer becoming
//! monolithic.
class CONFIG_EXPORT CAutoconfigurerFieldRolePenalties : core::CNonCopyable {
public:
    CAutoconfigurerFieldRolePenalties(const CAutoconfigurerParams& params);

    //! Get the penalty for categorical function arguments.
    const CPenalty& categoricalFunctionArgumentPenalty() const;

    //! Get the penalty for metric function arguments.
    const CPenalty& metricFunctionArgumentPenalty() const;

    //! Get the penalty for "by" fields.
    const CPenalty& byPenalty() const;

    //! Get the penalty for "by" fields of rare commands.
    const CPenalty& rareByPenalty() const;

    //! Get the penalty for "over" fields.
    const CPenalty& overPenalty() const;

    //! Get the penalty for "partition" fields.
    const CPenalty& partitionPenalty() const;

private:
    using TPenaltyCPtr = std::shared_ptr<const CPenalty>;

private:
    //! The penalties.
    TPenaltyCPtr m_Penalties[6];
};
}
}

#endif // INCLUDED_ml_config_CAutoconfigurerFieldRolePenalties_h
