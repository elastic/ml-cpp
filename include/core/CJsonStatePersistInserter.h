/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CJsonStatePersistInserter_h
#define INCLUDED_ml_core_CJsonStatePersistInserter_h

#include <core/CRapidJsonLineWriter.h>
#include <core/CStatePersistInserter.h>
#include <core/ImportExport.h>

#include <rapidjson/ostreamwrapper.h>

#include <iosfwd>
#include <ostream>

namespace ml
{
namespace core
{


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
//! Unlike the CRapidXmlStatePersistInserter, there is no possibility
//! of including attributes on the root node (because JSON does not
//! have attributes).  This may complicate code that needs to be 100%
//! JSON/XML agnostic.
//!
class CORE_EXPORT CJsonStatePersistInserter : public CStatePersistInserter
{
    public:
        //! Root node has no attributes
        CJsonStatePersistInserter(std::ostream &outputStream);

        //! Destructor flushes
        virtual ~CJsonStatePersistInserter(void);

        //! Store a name/value
        virtual void insertValue(const std::string &name,
                                 const std::string &value);

        //! Write as an integer avoiding the string conversion
        //! overloads
        void insertInteger(const std::string &name, size_t value);

        // Bring extra base class overloads into scope
        using CStatePersistInserter::insertValue;

        //! Flush the underlying output stream
        void flush(void);

    protected:
        //! Start a new level with the given name
        virtual void newLevel(const std::string &name);

        //! End the current level
        virtual void endLevel(void);

    private:
        //! JSON writer ostream wrapper
        rapidjson::OStreamWrapper     m_WriteStream;

        using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

        //! JSON writer
        TGenericLineWriter            m_Writer;
};


}
}

#endif // INCLUDED_ml_core_CJsonStatePersistInserter_h

