/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CSingleStreamSearcher_h
#define INCLUDED_ml_api_CSingleStreamSearcher_h

#include <core/CDataSearcher.h>

#include <api/ImportExport.h>

namespace ml {
namespace api {

//! \brief
//! Retrieves data from a single C++ stream.
//!
//! DESCRIPTION:\n
//! Retrieves data from a single C++ stream.
//!
//! The format must consist of one or more blocks of input in the
//! format returned by Elasticsearch's get API, separated by zero
//! bytes ('\0').  This class will return the same stream for
//! every search.  The caller must not close the stream.
//!
//! Implements the CDataSearcher interface for loading previously
//! persisted data.
//!
//! IMPLEMENTATION DECISIONS:\n
//! It's seemingly ridiculous to return the same stream over and over
//! again, but doing this enables the interface to be used in cases
//! where different streams are returned for each request.
//!
class API_EXPORT CSingleStreamSearcher : public core::CDataSearcher {
public:
    //! The \p stream must already be open when the constructor is
    //! called.
    CSingleStreamSearcher(const TIStreamP& stream);

    //! Ensure destructor is called to consume stream to its end
    ~CSingleStreamSearcher() override;

    //! Get the stream to retrieve data from.
    //! \return Pointer to the input stream.
    //! Some errors cannot be detected by this call itself, and are
    //! indicated by the stream going into the "bad" state as it is
    //! read from.
    TIStreamP search(size_t currentDocNum, size_t limit) override;

private:
    void consumeStream();

private:
    //! The stream we're reading from.
    TIStreamP m_Stream;
};
}
}

#endif // INCLUDED_ml_api_CSingleStreamSearcher_h
