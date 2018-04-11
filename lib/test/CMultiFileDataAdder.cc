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
#include <test/CMultiFileDataAdder.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include <exception>
#include <fstream>

namespace ml {
namespace test {

const std::string CMultiFileDataAdder::JSON_FILE_EXT = ".json";

CMultiFileDataAdder::CMultiFileDataAdder(std::string baseFilename, std::string fileExtension) {
    m_BaseFilename.swap(baseFilename);
    m_FileExtension.swap(fileExtension);
}

CMultiFileDataAdder::TOStreamP CMultiFileDataAdder::addStreamed(const std::string& index, const std::string& id) {
    const std::string& filename = this->makeFilename(index, id);

    TOStreamP strm(boost::make_shared<std::ofstream>(filename.c_str()));
    if (!strm->good()) {
        LOG_ERROR(<< "Failed to create new output stream for file " << filename);
        strm.reset();
    }

    return strm;
}

bool CMultiFileDataAdder::streamComplete(TOStreamP& strm, bool /*force*/) {
    std::ofstream* ofs(dynamic_cast<std::ofstream*>(strm.get()));
    if (ofs == nullptr) {
        return false;
    }

    ofs->close();

    return !ofs->bad();
}

std::string CMultiFileDataAdder::makeFilename(const std::string& index, const std::string& id) const {
    // NB: The logic in here must mirror that of CMultiFileSearcher::search()

    std::string filename(m_BaseFilename);
    if (!index.empty()) {
        filename += "/_";
        filename += index;
    }

    try {
        // Prior existence of the directory is not considered an error by
        // boost::filesystem, and this is what we want
        boost::filesystem::path directoryPath(filename);
        boost::filesystem::create_directories(directoryPath);
    } catch (std::exception& e) { LOG_ERROR(<< "Failed to create directory " << filename << " - " << e.what()); }

    filename += '/';
    filename += id;
    filename += m_FileExtension;

    return filename;
}
}
}
