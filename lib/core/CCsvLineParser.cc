/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCsvLineParser.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CoreTypes.h>

namespace ml {
namespace core {

// Initialise statics
const char CCsvLineParser::COMMA(',');
const char CCsvLineParser::QUOTE('"');

CCsvLineParser::CCsvLineParser(char separator)
    : m_Separator(separator), m_SeparatorAfterLastField(false), m_Line(nullptr),
      m_LineCurrent(nullptr), m_LineEnd(nullptr), m_WorkFieldEnd(nullptr),
      m_WorkFieldCapacity(0) {
}

void CCsvLineParser::reset(const std::string& line) {
    m_SeparatorAfterLastField = false;

    m_Line = &line;
    m_LineCurrent = line.data();
    m_LineEnd = line.data() + line.length();

    // Ensure that m_WorkField is big enough to hold the entire record, even if
    // it turns out to be a single field - this avoids the need to check if it's
    // big enough when it's populated (unlike std::vector or std::string)
    size_t minCapacity(line.length() + 1);
    if (m_WorkFieldCapacity < minCapacity) {
        m_WorkFieldCapacity = minCapacity;
        m_WorkField.reset(new char[minCapacity]);
    }
    m_WorkFieldEnd = m_WorkField.get();
}

bool CCsvLineParser::parseNext(std::string& value) {
    if (m_Line == nullptr) {
        return false;
    }
    if (this->parseNextToken(m_LineEnd, m_LineCurrent) == false) {
        return false;
    }
    value.assign(m_WorkField.get(), m_WorkFieldEnd - m_WorkField.get());
    return true;
}

bool CCsvLineParser::atEnd() const {
    return m_LineCurrent == m_LineEnd;
}

bool CCsvLineParser::parseNextToken(const char* end, const char*& current) {
    m_WorkFieldEnd = m_WorkField.get();

    if (current == end) {
        // Allow one empty token at the end of a line
        if (!m_SeparatorAfterLastField) {
            LOG_ERROR(<< "Trying to read too many fields from record:" << core_t::LINE_ENDING
                      << *m_Line);
            return false;
        }
        m_SeparatorAfterLastField = false;
        return true;
    }

    bool insideQuotes(false);
    do {
        if (insideQuotes) {
            if (*current == QUOTE) {
                // We need to look at the character after the quote
                ++current;
                if (current == end) {
                    m_SeparatorAfterLastField = false;
                    return true;
                }

                // The quoting state needs to be reversed UNLESS there are two
                // adjacent quotes
                if (*current != QUOTE) {
                    insideQuotes = false;

                    // Cater for the case where the character after the quote is
                    // the separator
                    if (*current == m_Separator) {
                        ++current;
                        m_SeparatorAfterLastField = true;
                        return true;
                    }
                }
            }

            *(m_WorkFieldEnd++) = *current;
        } else {
            if (*current == m_Separator) {
                ++current;
                m_SeparatorAfterLastField = true;
                return true;
            }

            if (*current == QUOTE) {
                // We're not currently inside quotes so a quote puts us inside
                // quotes regardless of the next character, and we never want to
                // include this quote in the field value
                insideQuotes = true;
            } else {
                *(m_WorkFieldEnd++) = *current;
            }
        }
    } while (++current != end);

    m_SeparatorAfterLastField = false;

    // Inconsistency if the last character of the string is an unmatched quote
    if (insideQuotes) {
        LOG_ERROR(<< "Unmatched final quote in record:" << core_t::LINE_ENDING << *m_Line);
        return false;
    }

    return true;
}

void CCsvLineParser::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCsvLineParser");
    mem->addItem("m_WorkField", m_WorkFieldCapacity);
}

std::size_t CCsvLineParser::memoryUsage() const {
    std::size_t mem = 0;
    mem += m_WorkFieldCapacity;
    return mem;
}
}
}
