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
#ifndef INCLUDED_ml_test_CMultiFileDataAdder_h
#define INCLUDED_ml_test_CMultiFileDataAdder_h

#include <core/CDataAdder.h>

#include <test/ImportExport.h>

#include <string>

namespace ml {
namespace test {

//! \brief
//! A file based persister for writing Ml models.
//!
//! DESCRIPTION:\n
//! Output file paths are a concatenation of the baseFilename
//! passed to the constructor, a hardcoded "_index" directory,
//! the "id" argument to the persistence method and the file
//! extension passed to the constructor (default '.json').
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only stream-based methods are presented here, as persistence
//! with large models can run to many gigabytes of data
//!
//! Data is added in multiple streams, each written to a different
//! file.  For an option to persist an entire model to a single
//! file, use the CSingleStreamDataAdder class.
//!
class TEST_EXPORT CMultiFileDataAdder : public core::CDataAdder {
public:
    //! Default file extension for persisted files.
    static const std::string JSON_FILE_EXT;

public:
    //! Constructor uses the pass-by-value-and-swap idiom
    CMultiFileDataAdder(std::string baseFilename, std::string fileExtension = JSON_FILE_EXT);

    //! Add streamed data
    //! \param id File name (without extension)
    TOStreamP addStreamed(const std::string& id) override;

    //! Clients that get a stream using addStreamed() must call this
    //! method one they've finished sending data to the stream.
    bool streamComplete(TOStreamP& strm, bool force) override;

private:
    //! Make a file name of the form base/_index/id.extension
    std::string makeFilename(const std::string& id) const;

private:
    //! Name of the file to serialise models to
    std::string m_BaseFilename;

    //! The extension for the peristed files
    std::string m_FileExtension;
};
}
}

#endif // INCLUDED_ml_test_CMultiFileDataAdder_h
