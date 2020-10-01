/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_controller_CResponseJsonWriter_h
#define INCLUDED_ml_controller_CResponseJsonWriter_h

#include <core/CRapidJsonLineWriter.h>

#include <rapidjson/ostreamwrapper.h>

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
//! A newline is written after each document, i.e. the output is ND-JSON.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not using the concurrent line writer, as there's no need for thread
//! safety.
//!
class CResponseJsonWriter {
public:
    //! \param[in] responseStream The stream to which to write responses.
    CResponseJsonWriter(std::ostream& responseStream);

    //! Writes a response in JSON format.
    void writeResponse(std::uint32_t id, bool success, const std::string& reason);

private:
    //! JSON writer ostream wrapper
    rapidjson::OStreamWrapper m_WriteStream;

    using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

    //! JSON writer
    TGenericLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_controller_CResponseJsonWriter_h
