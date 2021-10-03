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
#include <test/CMultiFileSearcher.h>

#include <core/CDataAdder.h>
#include <core/CLogger.h>

#include <fstream>
#include <memory>

namespace ml {
namespace test {

const std::string CMultiFileSearcher::JSON_FILE_EXT{".json"};

CMultiFileSearcher::CMultiFileSearcher(std::string baseFilename,
                                       std::string baseDocId,
                                       std::string fileExtension)
    : m_BaseFilename{std::move(baseFilename)}, m_BaseDocId{std::move(baseDocId)},
      m_FileExtension{std::move(fileExtension)} {
}

CMultiFileSearcher::TIStreamP CMultiFileSearcher::search(std::size_t currentDocNum,
                                                         std::size_t limit) {
    if (limit != 1) {
        LOG_ERROR(<< "File searcher can only operate with a limit of 1");
        return TIStreamP{};
    }

    // NB: The logic in here must mirror that of CMultiFileDataAdder::makeFilename()

    std::string filename{m_BaseFilename};
    filename += "/_index/";
    filename += core::CDataAdder::makeCurrentDocId(m_BaseDocId, currentDocNum);
    filename += m_FileExtension;

    // Failure to open the file is not necessarily an error - the calling method
    // will decide.  Therefore, return a pointer to the stream even if it's not
    // in the "good" state.
    return std::make_shared<std::ifstream>(filename);
}
}
}
