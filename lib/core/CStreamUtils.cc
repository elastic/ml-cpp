/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStreamUtils.h>

#include <core/CLogger.h>

#include <fstream>

namespace ml
{
namespace core
{

void CStreamUtils::skipUtf8Bom(std::ifstream &strm)
{
    if (strm.tellg() != std::streampos(0))
    {
        return;
    }
    std::ios_base::iostate origState(strm.rdstate());
    // The 3 bytes 0xEF, 0xBB, 0xBF form a UTF-8 byte order marker (BOM)
    if (strm.get() == 0xEF && strm.get() == 0xBB && strm.get() == 0xBF)
    {
        LOG_DEBUG("Skipping UTF-8 BOM");
        return;
    }
    // Set the stream state back to how it was originally so subsequent
    // code can report errors
    strm.clear(origState);
    // There was no BOM, so seek back to the beginning of the file
    strm.seekg(0);
}

}
}
