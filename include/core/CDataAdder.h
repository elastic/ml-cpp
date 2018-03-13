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
#ifndef INCLUDED_ml_core_CDataAdder_h
#define INCLUDED_ml_core_CDataAdder_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <functional>
#include <iosfwd>
#include <string>

#include <stdint.h>

namespace ml {
namespace core {

//! \brief
//! Abstract interface for data adder.
//!
//! DESCRIPTION:\n
//! Contains methods that require data be known in advance, plus a
//! method that returns a stream to which the data to be persisted
//! can be written.  This latter method is obviously much more
//! memory-efficient in cases where is is a viable option.
//!
//! IMPLEMENTATION DECISIONS:\n
//! There's an assumption that persisted state will be saved to a
//! data store that can retrieve based on 2 values: index and ID.
//! Elasticsearch supports this.
//!
class CORE_EXPORT CDataAdder : private CNonCopyable {
public:
    typedef boost::shared_ptr<std::ostream> TOStreamP;
    typedef boost::shared_ptr<CDataAdder> TDataAdderP;

    typedef std::function<bool(CDataAdder &)> TPersistFunc;

public:
    virtual ~CDataAdder(void);

    //! Add streamed data - return of NULL stream indicates failure.
    //! Since the data to be written isn't known at the time this function
    //! returns it is not possible to detect all error conditions
    //! immediately.  If the stream goes bad whilst being written to then
    //! this also indicates failure.
    virtual TOStreamP addStreamed(const std::string &index, const std::string &id) = 0;

    //! Clients that get a stream using addStreamed() must call this
    //! method one they've finished sending data to the stream.
    //! They should set force to true when the very last stream is
    //! complete, in case the persister needs to close off some
    //! sort of cached data structure.
    virtual bool streamComplete(TOStreamP &strm, bool force) = 0;

    //! The max number of documents that can go in a single
    //! batch save
    virtual std::size_t maxDocumentsPerBatchSave(void) const;

    //! The max size of a document - to be determined by the
    //! underlying storage medium
    virtual std::size_t maxDocumentSize(void) const;

    //! Get the current document ID given a base ID and current document
    //! document number.  The ID is of the form baseId#currentDocNum if
    //! baseId is not empty, and simply currentDocNum converted to a string
    //! if baseId is empty.
    static std::string makeCurrentDocId(const std::string &baseId, size_t currentDocNum);
};
}
}

#endif// INCLUDED_ml_core_CDataAdder_h
