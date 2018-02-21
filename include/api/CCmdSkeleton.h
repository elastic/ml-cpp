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
#ifndef INCLUDED_ml_api_CCmdSkeleton_h
#define INCLUDED_ml_api_CCmdSkeleton_h

#include <core/CNonCopyable.h>

#include <api/ImportExport.h>

#include <string>


namespace ml
{
namespace core
{
class CDataAdder;
class CDataSearcher;
}
namespace api
{
class CDataProcessor;
class CInputParser;

//! \brief
//! Basic implementation of an API command.
//!
//! DESCRIPTION:\n
//! Factors out some boilerplate code.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Input/output streams are held by reference.  They must outlive objects of
//! this class, which, in practice, means that the CIoManager object managing
//! them must outlive this object.
//!
class API_EXPORT CCmdSkeleton : private core::CNonCopyable
{
    public:
        CCmdSkeleton(core::CDataSearcher *restoreSearcher,
                     core::CDataAdder *persister,
                     CInputParser &inputParser,
                     CDataProcessor &processor);

        //! Pass input to the processor until it's consumed as much as it can.
        bool ioLoop(void);

    private:
        //! Persists the state of the models
        bool persistState(void);

    private:
        //! NULL if state restoration is not required.
        core::CDataSearcher *m_RestoreSearcher;

        //! NULL if state persistence is not required.
        core::CDataAdder    *m_Persister;

        //! Input data parser.
        CInputParser        &m_InputParser;

        //! Reference to the object that's going to do the command-specific
        //! processing of the data.
        CDataProcessor      &m_Processor;
};


}
}

#endif // INCLUDED_ml_api_CCmdSkeleton_h

