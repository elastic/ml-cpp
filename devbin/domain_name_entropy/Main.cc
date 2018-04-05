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
//! \brief
//! Given a list of domain names, use the Mozilla top level
//! domain name list:
//!
//! https://publicsuffix.org/list/effective_tld_names.dat
//!
//! to resolve the 'registered domain name' and compute
//! the total information content of all subdomain names
//! for a a registered domain name.
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV data on STDIN,
//! and sends its CSV results to STDOUT.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CLogger.h>
#include <core/CStatistics.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include "CAddRegisteredDomainAndEntropyToCsv.h"
#include "CCmdLineParser.h"
#include "CTopLevelDomainDb.h"

#include <iostream>
#include <string>

using namespace ml;
using namespace domain_name_entropy;

int main(int argc, char** argv) {
    // Read command line options
    std::string csvFileName;
    std::string domainNameFieldName;
    std::string timeFieldName;
    if (CCmdLineParser::parse(argc, argv, csvFileName, domainNameFieldName, timeFieldName) == false) {
        return EXIT_FAILURE;
    }

    // This must be done from the program, and NOT a shared library, as each
    // program statically links its own version library.
    LOG_INFO(ml::ver::CBuildInfo::fullInfo());

    // Start
    CTopLevelDomainDb tldDb("./effective_tld_names.txt");

    LOG_DEBUG("tldDb.init()");
    if (tldDb.init() == false) {
        LOG_ERROR("Can not initialise TLD DB");
        return EXIT_FAILURE;
    }
    LOG_DEBUG("tldDb.init() done");

    // Read in a CSV file
    CAddRegisteredDomainAndEntropyToCsv csvReader(tldDb, csvFileName, domainNameFieldName, timeFieldName, "entropy");

    LOG_DEBUG("csvReader.init()");
    if (csvReader.init() == false) {
        LOG_ERROR("Can not initialise reader");
        return EXIT_FAILURE;
    }
    LOG_DEBUG("csvReader.init() done");

    return EXIT_SUCCESS;
}
