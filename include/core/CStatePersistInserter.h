/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStatePersistInserter_h
#define INCLUDED_ml_core_CStatePersistInserter_h

#include <core/CNonCopyable.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <limits>
#include <string>


namespace ml
{
namespace core
{


//! \brief
//! Abstract interface for persisting state.
//!
//! DESCRIPTION:\n
//! Classes that need to persist state may accept this interface
//! as a means to generically state which values they need to
//! persist.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable.
//!
//! All values are stored as strings.
//!
class CORE_EXPORT CStatePersistInserter : private CNonCopyable
{
    public:
        //! Virtual destructor for abstract class
        virtual ~CStatePersistInserter(void);

        //! Store a name/value
        virtual void insertValue(const std::string &name,
                                 const std::string &value) = 0;

        //! Store an arbitrary type that can be converted to a string
        template <typename TYPE>
        void insertValue(const std::string &name,
                         const TYPE &value)
        {
            this->insertValue(name, CStringUtils::typeToString(value));
        }

        //! Store a floating point number with a given level of precision
        void insertValue(const std::string &name,
                         double value,
                         CIEEE754::EPrecision precision);

        //! Store a nested level of state, to be populated by the supplied
        //! function or function object
        template <typename FUNC>
        void insertLevel(const std::string &name,
                         FUNC f)
        {
            CAutoLevel level(name, *this);
            f(*this);
        }

    protected:
        //! Start a new level with the given name
        virtual void newLevel(const std::string &name) = 0;

        //! End the current level
        virtual void endLevel(void) = 0;

    private:
        //! Class to implement RAII for moving to the next level
        class CORE_EXPORT CAutoLevel : private CNonCopyable
        {
            public:
                CAutoLevel(const std::string &name,
                           CStatePersistInserter &inserter);
                ~CAutoLevel(void);

            private:
                CStatePersistInserter &m_Inserter;
        };

};


}
}

#endif // INCLUDED_ml_core_CStatePersistInserter_h

