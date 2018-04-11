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
#ifndef INCLUDED_ml_core_CJsonLogLayout_h
#define INCLUDED_ml_core_CJsonLogLayout_h

#include <core/ImportExport.h>

#include <log4cxx/layout.h>

class CJsonLogLayoutTest;

// NB: log4cxx extensions have to go in the log4cxx namespace, hence cannot
// stick to the convention of our code being in the ml namespace.  This
// is due to use of (log4cxx mandated) macros in the implementation.
namespace log4cxx {
namespace helpers {

//! \brief
//! Output log messages as lineified JSON.
//!
//! DESCRIPTION:\n
//! Logs messages as lineified JSON (one log message on one line).  The
//! fields are based on those used by log4cxx's built in XMLLayout class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Violates several aspects of the Ml coding standards in order
//! to work with log4cxx macros and other conventions.
//!
class CORE_EXPORT CJsonLogLayout : public Layout {
public:
    DECLARE_LOG4CXX_OBJECT(CJsonLogLayout)
    BEGIN_LOG4CXX_CAST_MAP()
    LOG4CXX_CAST_ENTRY(CJsonLogLayout)
    LOG4CXX_CAST_ENTRY_CHAIN(Layout)
    END_LOG4CXX_CAST_MAP()

    CJsonLogLayout();

    //! Accessors for location info (i.e. should file/line be included in
    //! log output?
    void locationInfo(bool locationInfo);
    bool locationInfo() const;

    //! Accessors for whether MDC key-value pairs should be output.
    void properties(bool properties);
    bool properties() const;

    //! No options to activate.
    void activateOptions(Pool& p);

    //! Set options.
    virtual void setOption(const LogString& option, const LogString& value);

    //! Formats a LoggingEvent as JSON.
    virtual void
    format(LogString& output, const spi::LoggingEventPtr& event, Pool& p) const;

    //! The CJsonLogLayout prints and does not ignore exceptions.
    virtual bool ignoresThrowable() const;

private:
    //! Include location info by default
    bool m_LocationInfo;
    bool m_Properties;

    static std::string cropPath(const std::string& filename);

    // For unit testing
    friend class ::CJsonLogLayoutTest;
};

LOG4CXX_PTR_DEF(CJsonLogLayout);

} // end helpers

namespace classes {
extern const helpers::ClassRegistration& CJsonLogLayoutRegistration;
}

} // end log4cxx

#endif // INCLUDED_ml_core_CJsonLogLayout_h
