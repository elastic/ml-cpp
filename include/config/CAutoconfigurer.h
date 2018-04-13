/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CAutoconfigurer_h
#define INCLUDED_ml_config_CAutoconfigurer_h

#include <api/CDataProcessor.h>

#include <config/ImportExport.h>

#include <boost/shared_ptr.hpp>

namespace ml {
namespace config {
class CAutoconfigurerImpl;
class CAutoconfigurerParams;
class CReportWriter;

//! \brief Responsible for automatic configuration.
//!
//! DESCRTIPTION:\n
//! This is the main entry point to automatic configuration. It delegates
//! the various tasks, such as discovering data semantics, creating summary
//! statistics and performing the analysis needed to define candidate searches
//! to other objects.
//!
//! IMPLEMENTATION:\n
//! This is a member of the api::CDataProcessor hierarchy so that commands
//! using this can make use of api::CCmdSkeleton.
//!
//! We use the pimpl idiom to isolate the internals of this library from the
//! automatic configuration commands.
class CONFIG_EXPORT CAutoconfigurer : public api::CDataProcessor {
public:
    CAutoconfigurer(const CAutoconfigurerParams& params, CReportWriter& reportWriter);

    //! We're going to be writing to a new output stream.
    virtual void newOutputStream();

    //! Receive a single record to be processed.
    virtual bool handleRecord(const TStrStrUMap& fieldValues);

    //! Generate the report.
    virtual void finalise();

    //! No-op.
    virtual bool restoreState(core::CDataSearcher& restoreSearcher, core_t::TTime& completeToTime);

    //! No-op.
    virtual bool persistState(core::CDataAdder& persister);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled() const;

    //! Access the output handler.
    virtual api::COutputHandler& outputHandler();

private:
    using TImplPtr = boost::shared_ptr<CAutoconfigurerImpl>;

private:
    //! The pointer to the actual implementation.
    TImplPtr m_Impl;
};
}
}

#endif // INCLUDED_ml_config_CAutoconfigurer_h
