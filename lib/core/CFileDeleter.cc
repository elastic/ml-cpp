/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CFileDeleter.h>

#include <core/CLogger.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>


namespace ml
{
namespace core
{


CFileDeleter::CFileDeleter(const std::string &fileName)
    : m_FileName(fileName)
{
}

CFileDeleter::~CFileDeleter(void)
{
    if (m_FileName.empty())
    {
        return;
    }

    if (::remove(m_FileName.c_str()) == -1)
    {
        LOG_WARN("Failed to remove file " << m_FileName <<
                 " : " << ::strerror(errno));
    }
}


}
}

