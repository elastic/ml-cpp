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
#ifndef INCLUDED_ml_test_CMultiFileSearcher_h
#define INCLUDED_ml_test_CMultiFileSearcher_h

#include <core/CDataSearcher.h>

#include <test/ImportExport.h>

#include <string>

namespace ml {
namespace test {

//! \brief
//! Retrieves data previously persisted to file.
//!
//! DESCRIPTION:\n
//! Implements the CDataSearcher interface for loading previously
//! persisted data.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Failure to open a file stream is not considered an error by this
//! class - it is left to the caller to decide whether it is a problem.
//!
//! This class is designed to reload data persisted by the CMultiFileDataAdder
//! class.  For data persisted by the CSingleStreamDataAdder class, use the
//! CSingleStreamSearcher.
//!
class TEST_EXPORT CMultiFileSearcher : public core::CDataSearcher {
public:
    //! File extension for persisted files.
    static const std::string JSON_FILE_EXT;

public:
    //! Constructor uses the pass-by-value-and-move idiom
    CMultiFileSearcher(std::string baseFilename,
                       std::string baseDocId,
                       std::string fileExtension = JSON_FILE_EXT);

    //! Load the file
    //! \return Pointer to the input stream - may be NULL
    virtual TIStreamP search(size_t currentDocNum, size_t limit);

private:
    //! Name of the file to serialise models to
    std::string m_BaseFilename;

    //! Base ID for stored documents
    std::string m_BaseDocId;

    //! The extension for the peristed files
    std::string m_FileExtension;
};
}
}

#endif// INCLUDED_ml_test_CMultiFileSearcher_h
