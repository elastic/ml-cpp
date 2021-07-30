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
#ifndef INCLUDED_ml_core_CTimeGm_h
#define INCLUDED_ml_core_CTimeGm_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <time.h>

namespace ml {
namespace core {

//! \brief
//! Convert tm into a time_t without making a timezone adjustment.
//! See timegm man page for details.
//!
//! DESCRIPTION:\n
//! Convert tm into a time_t without making a timezone adjustment.
//! See timegm man page for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//!
class CORE_EXPORT CTimeGm : private CNonInstantiatable {
public:
    static time_t timeGm(struct tm* ts);
};
}
}

#endif // INCLUDED_ml_core_CTimeGm_h
