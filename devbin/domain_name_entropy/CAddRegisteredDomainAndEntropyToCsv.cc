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
#include "CAddRegisteredDomainAndEntropyToCsv.h"

#include <core/CLogger.h>
#include <core/CTextFileWatcher.h>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>

#include <iostream>
#include <iterator>
#include <stdexcept>

#include "CTopLevelDomainDb.h"

namespace ml {
namespace domain_name_entropy {

CAddRegisteredDomainAndEntropyToCsv::CAddRegisteredDomainAndEntropyToCsv(
    const CTopLevelDomainDb& topLevelDomainDb,
    const std::string& csvFileName,
    const std::string& domainNameFieldName,
    const std::string& timeFieldName,
    const std::string& entropyFieldName)
    : m_TopLevelDomainDb(topLevelDomainDb), m_CsvFileName(csvFileName),
      m_DomainNameFieldName(domainNameFieldName),
      m_TimeFieldName(timeFieldName), m_EntropyFieldName(entropyFieldName),
      m_DomainNameFieldIndex(0), m_TimeFieldIndex(0) {
}

bool CAddRegisteredDomainAndEntropyToCsv::init(void) {
    core::CTextFileWatcher watcher;

    if (watcher.init(m_CsvFileName, "\r?\n", core::CTextFileWatcher::E_Start) == false) {
        LOG_ERROR(<< "Can not open " << m_CsvFileName);
        return false;
    }

    bool readHeader(false);
    std::string lastTime;

    std::string remainder;
    if (watcher.readAllLines(boost::bind(&CAddRegisteredDomainAndEntropyToCsv::readLine, this,
                                         boost::ref(readHeader), boost::ref(lastTime), _1),
                             remainder) == false) {
        LOG_ERROR(<< "Error reading " << m_CsvFileName);
        return false;
    }

    this->flush(lastTime);

    return true;
}

bool CAddRegisteredDomainAndEntropyToCsv::readLine(bool& readHeader,
                                                   std::string& lastTime,
                                                   const std::string& line) {
    static int count(0);

    ++count;

    if (count % 100 == 0) {
        LOG_DEBUG(<< "Read " << count << " lines");
    }

    // Split csv line
    core::CStringUtils::TStrVec tokens;
    std::string remainder;

    core::CStringUtils::tokenise(",", line, tokens, remainder);

    if (!remainder.empty()) {
        tokens.push_back(remainder);
    }

    // Read the header if not done already
    if (readHeader == false) {
        core::CStringUtils::TStrVecCItr itr2 =
            std::find(tokens.begin(), tokens.end(), m_DomainNameFieldName);
        if (itr2 == tokens.end()) {
            LOG_ERROR(<< m_DomainNameFieldName << " not in header line " << line);
            return false;
        }
        core::CStringUtils::TStrVecCItr itr1 = tokens.begin();
        m_DomainNameFieldIndex = std::distance(itr1, itr2);

        itr2 = std::find(tokens.begin(), tokens.end(), m_TimeFieldName);
        if (itr2 == tokens.end()) {
            LOG_ERROR(<< m_TimeFieldName << " not in header line " << line);
            return false;
        }
        m_TimeFieldIndex = std::distance(itr1, itr2);

        readHeader = true;

        return true;
    }

    // We can not get here without a valid header
    if (m_DomainNameFieldIndex >= tokens.size() || m_TimeFieldIndex >= tokens.size()) {
        LOG_ERROR(<< "Out of range " << tokens.size() << " "
                  << m_DomainNameFieldIndex << " " << m_TimeFieldIndex);
        return false;
    }

    std::string hostName = tokens.at(m_DomainNameFieldIndex);
    const std::string& time = tokens.at(m_TimeFieldIndex);

    if (time != lastTime) {
        this->flush(lastTime);
        lastTime = time;
    }

    // Remove quotes from hostName (if they are there)
    hostName.erase(remove(hostName.begin(), hostName.end(), '\"'), hostName.end());

    std::string subDomain;
    std::string domain;
    std::string suffix;

    // Split the domain name
    m_TopLevelDomainDb.splitHostName(hostName, subDomain, domain, suffix);

    // Create a 'registered' domain
    std::string pivotDomain = domain + "." + suffix;

    TStrCompressUtilsPMapItr itr = m_RegisteredDomainEntropy.find(pivotDomain);
    if (itr == m_RegisteredDomainEntropy.end()) {
        TCompressUtilsP compressedSubDomainsP(new CCompressUtils);

        itr = m_RegisteredDomainEntropy
                  .insert(TStrCompressUtilsPMap::value_type(pivotDomain, compressedSubDomainsP))
                  .first;
    }

    if (itr->second->compressString(false, subDomain) == false) {
        LOG_ERROR(<< "Unable to compress " << hostName);
    }

    return true;
}

void CAddRegisteredDomainAndEntropyToCsv::flush(const std::string& time) {
    // Finish all strings and dump
    for (TStrCompressUtilsPMapCItr itr = m_RegisteredDomainEntropy.begin();
         itr != m_RegisteredDomainEntropy.end(); ++itr) {
        size_t length;
        if (itr->second->compressedStringLength(true, length) == false) {
            LOG_ERROR(<< "Unable to process " << itr->first);
        } else {
            std::cout << time << "," << itr->first << "," << length << std::endl;
        }
    }

    m_RegisteredDomainEntropy.clear();
}
}
}
