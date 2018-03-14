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
#ifndef INCLUDED_ml_domain_name_entropy_CAddRegisteredDomainAndEntropyToCsv_h
#define INCLUDED_ml_domain_name_entropy_CAddRegisteredDomainAndEntropyToCsv_h

#include <map>
#include <set>
#include <string>
#include <vector>

#include <core/CNonCopyable.h>

#include <boost/shared_ptr.hpp>

#include "CCompressUtils.h"

namespace ml {
namespace domain_name_entropy {
class CTopLevelDomainDb;

//! \brief
//!
//! DESCRIPTION:\n
//!
//! IMPLEMENTATION DECISIONS:\n
//!
class CAddRegisteredDomainAndEntropyToCsv : private core::CNonCopyable {
public:
    CAddRegisteredDomainAndEntropyToCsv(const CTopLevelDomainDb& topLevelDomainDb,
                                        const std::string& csvFileName,
                                        const std::string& domainNameFieldName,
                                        const std::string& timeFieldName,
                                        const std::string& entropyFieldName);

    bool init(void);

    void flush(const std::string& time);

private:
    //! Read a line from the csv file
    bool readLine(bool& readHeader, std::string& lastTime, const std::string& line);

private:
    const CTopLevelDomainDb& m_TopLevelDomainDb;
    const std::string m_CsvFileName;
    const std::string m_DomainNameFieldName;
    const std::string m_TimeFieldName;
    const std::string m_EntropyFieldName;

    typedef std::vector<std::string> TStrVec;

    TStrVec::size_type m_DomainNameFieldIndex;
    TStrVec::size_type m_TimeFieldIndex;

    typedef boost::shared_ptr<CCompressUtils> TCompressUtilsP;
    typedef std::map<std::string, TCompressUtilsP> TStrCompressUtilsPMap;
    typedef TStrCompressUtilsPMap::iterator TStrCompressUtilsPMapItr;
    typedef TStrCompressUtilsPMap::const_iterator TStrCompressUtilsPMapCItr;

    TStrCompressUtilsPMap m_RegisteredDomainEntropy;
};
}
}

#endif // INCLUDED_ml_domain_name_entropy_CAddRegisteredDomainAndEntropyToCsv_h
