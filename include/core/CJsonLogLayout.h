/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CJsonLogLayout_h
#define INCLUDED_ml_core_CJsonLogLayout_h

#include <core/ImportExport.h>

#include <boost/log/core/record_view.hpp>
#include <boost/log/utility/formatting_ostream_fwd.hpp>

#include <string>
#include <utility>

namespace CJsonLogLayoutTest {
struct testFoo;
}

namespace ml {
namespace core {

//! \brief
//! Output log messages as ND-JSON.
//!
//! DESCRIPTION:\n
//! Logs messages as ND-JSON (one log message on one line).  The
//! fields are based on those used by log4cxx's XMLLayout class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This was originally implemented as a log4cxx layout extension,
//! and the JSON documents it creates are designed to feed into
//! log4j on the Java side, so the field names are still in the
//! log4cxx/log4j style rather than Boost.Log.
//!
class CORE_EXPORT CJsonLogLayout {
public:
    CJsonLogLayout();

    //! Accessors for location info (i.e. should file/line be included in
    //! log output?
    void locationInfo(bool locationInfo);
    bool locationInfo() const;

    //! Formats a Boost.Log record as JSON.
    void operator()(const boost::log::record_view& rec,
                    boost::log::formatting_ostream& strm) const;

private:
    using TStrStrPr = std::pair<std::string, std::string>;

private:
    //! Keep just the last element of a path.
    //!
    //! Example:
    //!
    //! /usr/include/unistd.h
    //!
    //! gets mapped to:
    //!
    //! unistd.h
    //!
    static std::string cropPath(const std::string& filename);

    //! Split a __PRETTY_FUNCTION__ or __FUNCSIG__ value into a class name and a
    //! method name.
    //!
    //! Example:
    //!
    //! Pretty function = std::string ns1::ns2::clazz::someMethod(int arg1, char arg2)
    //!
    //! gets mapped to:
    //!
    //! Class = ns1::ns2::clazz
    //! Method = someMethod
    //!
    //! \param prettyFunctionSig A value from __PRETTY_FUNCTION__ or __FUNCSIG__.
    //! \return A pair of the form {class name, method name}
    static TStrStrPr extractClassAndMethod(std::string prettyFunctionSig);

private:
    //! Include location info by default
    bool m_LocationInfo;

    // For unit testing
    friend struct CJsonLogLayoutTest::testFoo;
};
}
}

#endif // INCLUDED_ml_core_CJsonLogLayout_h
