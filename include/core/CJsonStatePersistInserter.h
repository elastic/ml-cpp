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
#ifndef INCLUDED_ml_core_CJsonStatePersistInserter_h
#define INCLUDED_ml_core_CJsonStatePersistInserter_h

#include <core/CBoostJsonLineWriter.h>
#include <core/CStatePersistInserter.h>
#include <core/CStreamWriter.h>
#include <core/ImportExport.h>

#include <iosfwd>
#include <ostream>

namespace ml {
namespace core {

//! \brief
//! For persisting state in JSON format.
//!
//! DESCRIPTION:\n
//! Concrete implementation of the CStatePersistInserter interface
//! that persists state in JSON format.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Output is streaming rather than building up an in-memory JSON
//! document.
//!
class CORE_EXPORT CJsonStatePersistInserter : public CStatePersistInserter {
public:
    template<class Func>
    static void persist(std::ostream& oss, const Func& func) {
        ml::core::CJsonStatePersistInserter inserter(oss);
        func(inserter);
    }

public:
    //! Root node has no attributes
    CJsonStatePersistInserter(std::ostream& outputStream);

    //! Destructor flushes
    ~CJsonStatePersistInserter() override;

    //! Store a name/value
    void insertValue(const std::string& name, const std::string& value) override;

    //! Write as an integer avoiding the string conversion
    //! overloads
    void insertInteger(const std::string& name, size_t value);

    // Bring extra base class overloads into scope
    using CStatePersistInserter::insertValue;

    //! Flush the underlying output stream
    void flush();

protected:
    //! Start a new level with the given name
    void newLevel(const std::string& name) override;

    //! End the current level
    void endLevel() override;

private:
    //! JSON writer ostream
    std::ostream& m_WriteStream;

    using TGenericLineWriter = CStreamWriter;

    //! JSON writer
    TGenericLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_core_CJsonStatePersistInserter_h
