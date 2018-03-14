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
#ifndef INCLUDED_CMockSearcher_h
#define INCLUDED_CMockSearcher_h

#include <core/CDataSearcher.h>

class CMockDataAdder;

//! \brief
//! Mock searcher for unit testing.
//!
//! DESCRIPTION:\n
//! The CSearcher class can search for data in a live server instance,
//! but for unit testing it's desirable to mock this class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The mock searcher's returns all events added to a mock data adder that
//! appear to be for the searched index.  The actual search string is NOT
//! properly applied.  This is OK for the current scope of the unit testing.
//!
class CMockSearcher : public ml::core::CDataSearcher {
    public:
        CMockSearcher(const CMockDataAdder &mockDataAdder);

        //! Do a search that results in an input stream.
        //! A return value of NULL indicates a technical problem with the
        //! creation of the stream.  Other errors may be indicated by the
        //! returned stream going into the "bad" state.
        virtual TIStreamP search(size_t currentDocNum, size_t limit);

    private:
        const CMockDataAdder &m_MockDataAdder;
};

#endif // INCLUDED_CMockSearcher_h
