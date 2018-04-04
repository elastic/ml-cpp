/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMockDataAdder_h
#define INCLUDED_CMockDataAdder_h

#include <core/CDataAdder.h>

#include <map>
#include <string>
#include <vector>


//! \brief
//! Mock data adder for unit testing.
//!
//! DESCRIPTION:\n
//! The CDataAdder class can add data into a live ES instance,
//! but for unit testing it's desirable to mock this class.
//!
//! IMPLEMENTATION DECISIONS:\n
//!
class CMockDataAdder : public ml::core::CDataAdder
{
    public:
        using TStrVec = std::vector<std::string>;
        using TStrStrVecMap = std::map<std::string, TStrVec>;
        using TStrStrVecMapCItr = TStrStrVecMap::const_iterator;
        using TStrOStreamPMap = std::map<std::string, TOStreamP>;
        using TStrOStreamPMapCItr = TStrOStreamPMap::const_iterator;
        using TStrOStreamPMapItr = TStrOStreamPMap::iterator;

    public:
        CMockDataAdder();

        //! Add streamed data - return of NULL stream indicates failure.
        //! Since the data to be written isn't known at the time this function
        //! returns it is not possible to detect all error conditions
        //! immediately.  If the stream goes bad whilst being written to then
        //! this also indicates failure.
        virtual TOStreamP addStreamed(const std::string &index,
                                      const std::string &id);

        //! Clients that get a stream using addStreamed() must call this
        //! method one they've finished sending data to the stream.
        //! They should set force to true when the very last stream is
        //! complete, in case the persister needs to close off some
        //! sort of cached data structure.
        virtual bool streamComplete(TOStreamP &strm,
                                    bool force);

        //! Access persisted events
        const TStrStrVecMap &events() const;

        //! Wipe the contents of the data store
        void clear();

    private:
        //! Persisted events
        TStrStrVecMap   m_Events;

        TStrOStreamPMap m_Streams;
};

#endif // INCLUDED_CMockDataAdder_h

