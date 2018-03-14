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
#include <core/CFileDeleter.h>

#include <core/CLogger.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>

namespace ml {
namespace core {

CFileDeleter::CFileDeleter(const std::string& fileName) : m_FileName(fileName) {}

CFileDeleter::~CFileDeleter(void) {
    if (m_FileName.empty()) {
        return;
    }

    if (::remove(m_FileName.c_str()) == -1) {
        LOG_WARN("Failed to remove file " << m_FileName << " : " << ::strerror(errno));
    }
}
}
}
