/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_controller_CResponseJsonWriter_h
#define INCLUDED_ml_controller_CResponseJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace controller {

//! \brief
//! Write a response to a controller command in JSON format.
//!
//! DESCRIPTION:\n
//! Output documents are of the form:
//!
//! { "id" : 123, "success" : true, "reason" : "message explaining success/failure" }
//!
//! They are written into a JSON array, i.e. the overall output looks
//! something like this:
//!
//! [{ "id" : 1, "success" : true, "reason" : "all ok" }
//! ,{ "id" : 2, "success" : false, "reason" : "something went wrong" }
//! ,{ "id" : 3, "success" : true, "reason" : "ok again" }
//! ]
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses the concurrent line writer.  There's no need for thread safety
//! with the current design, but in future commands might be processed
//! concurrently.
//!
class CResponseJsonWriter {
public:
    //! \param[in] responseStream The stream to which to write responses.
    CResponseJsonWriter(std::ostream& responseStream);

    //! Writes a response in JSON format.
    void writeResponse(std::uint32_t id, bool success, const std::string& reason);

private:
    //! Wrapped output stream
    core::CJsonOutputStreamWrapper m_WrappedOutputStream;

    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_controller_CResponseJsonWriter_h
