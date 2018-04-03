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
#include <api/CLineifiedXmlOutputWriter.h>

#include <core/CRapidXmlParser.h>
#include <core/CSleep.h>
#include <core/CXmlNodeWithChildren.h>

#include <ostream>


namespace ml
{
namespace api
{

namespace
{
const std::string EMPTY_STRING;
}


CLineifiedXmlOutputWriter::CLineifiedXmlOutputWriter(const std::string &rootName)
    : m_RootName(rootName),
      m_OutStream(m_StringOutputBuf)
{
}

CLineifiedXmlOutputWriter::CLineifiedXmlOutputWriter(const std::string &rootName,
                                                     std::ostream &strmOut)
    : m_RootName(rootName),
      m_OutStream(strmOut)
{
}

CLineifiedXmlOutputWriter::~CLineifiedXmlOutputWriter()
{
    // Since we didn't flush the stream whilst working, we flush it on
    // destruction
    m_OutStream.flush();

    // We don't want the program to die before the remote end of the link has
    // had a chance to read from any pipe to which our output stream might be
    // connected, so sleep briefly here
    core::CSleep::sleep(20);
}

bool CLineifiedXmlOutputWriter::fieldNames(const TStrVec &/*fieldNames*/,
                                           const TStrVec &/*extraFieldNames*/)
{
    return true;
}

const CLineifiedXmlOutputWriter::TStrVec &CLineifiedXmlOutputWriter::fieldNames() const
{
    return EMPTY_FIELD_NAMES;
}

bool CLineifiedXmlOutputWriter::writeRow(const TStrStrUMap &dataRowFields,
                                         const TStrStrUMap &overrideDataRowFields)
{
    core::CXmlNodeWithChildren::TXmlNodeWithChildrenP root(m_Pool.newNode(m_RootName,
                                                                          EMPTY_STRING));

    TStrStrUMapCItr fieldValueIter = dataRowFields.begin();
    for (; fieldValueIter != dataRowFields.end(); ++fieldValueIter)
    {
        const std::string &name = fieldValueIter->first;
        const std::string &value = fieldValueIter->second;

        // Only output fields that aren't overridden
        if (overrideDataRowFields.find(name) == overrideDataRowFields.end())
        {
            root->addChildP(m_Pool.newNode(name, value));
        }
    }

    fieldValueIter = overrideDataRowFields.begin();
    for (; fieldValueIter != overrideDataRowFields.end(); ++fieldValueIter)
    {
        const std::string &name = fieldValueIter->first;
        const std::string &value = fieldValueIter->second;

        root->addChildP(m_Pool.newNode(name, value));
    }

    std::string xml;
    core::CRapidXmlParser::convert(*root, xml);
    m_OutStream << core::CRapidXmlParser::toOneLine(xml) << '\n';
    m_Pool.recycle(root);

    return true;
}

std::string CLineifiedXmlOutputWriter::internalString() const
{
    const_cast<std::ostream &>(m_OutStream).flush();

    // This is only of any value if the first constructor was used - it's up to
    // the caller to know this
    return m_StringOutputBuf.str();
}


}
}

