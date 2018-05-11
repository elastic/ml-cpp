/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_seccomp_CSystemCallFilter_h
#define INCLUDED_ml_seccomp_CSystemCallFilter_h

namespace ml {
namespace seccomp {

//! \brief
//! Installs Sandbox/sys call filters
//!
//! DESCRIPTION:\n
//!
//! IMPLEMENTATION DECISIONS:\n
//! Platform specific
//!
//!
class CSystemCallFilter {
public:
    CSystemCallFilter();
};
}
}

#endif // INCLUDED_ml_seccomp_CSystemCallFilter_h
