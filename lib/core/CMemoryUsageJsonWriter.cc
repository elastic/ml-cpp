/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMemoryUsageJsonWriter.h>

namespace
{
const std::string MEMORY("memory");
const std::string UNUSED("unused");
}

namespace ml
{
namespace core
{

CMemoryUsageJsonWriter::CMemoryUsageJsonWriter(std::ostream &outStream) :
    m_WriteStream(outStream), m_Writer(m_WriteStream), m_Finalised(false)
{
}

CMemoryUsageJsonWriter::~CMemoryUsageJsonWriter()
{
    this->finalise();
}

void CMemoryUsageJsonWriter::startObject()
{
    m_Writer.StartObject();
}

void CMemoryUsageJsonWriter::endObject()
{
    m_Writer.EndObject();
}

void CMemoryUsageJsonWriter::startArray(const std::string &description)
{
    m_Writer.String(description);
    m_Writer.StartArray();
}

void CMemoryUsageJsonWriter::endArray()
{
    m_Writer.EndArray();
}

void CMemoryUsageJsonWriter::addItem(const CMemoryUsage::SMemoryUsage &item)
{
    m_Writer.String(item.s_Name);
    m_Writer.StartObject();

    m_Writer.String(MEMORY);
    m_Writer.Int64(item.s_Memory);
    if (item.s_Unused)
    {
        m_Writer.String(UNUSED);
        m_Writer.Uint64(item.s_Unused);
    }
    m_Writer.EndObject();
}

void CMemoryUsageJsonWriter::finalise()
{
    if (m_Finalised)
    {
        return;
    }
    m_WriteStream.Flush();
    m_Finalised = true;
}


} // core
} // ml
