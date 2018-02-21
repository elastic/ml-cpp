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
#include <core/CStat.h>

namespace ml
{
namespace core
{

CStat::CStat(void) : m_Value(uint64_t(0))
{
}

void CStat::increment(void)
{
    m_Value.fetch_add(1);
}

void CStat::increment(uint64_t value)
{
    m_Value.fetch_add(value);
}

void CStat::decrement(void)
{
    m_Value.fetch_sub(1);
}

void CStat::set(uint64_t value)
{
    m_Value.store(value);
}

uint64_t CStat::value(void) const
{
    return m_Value;
}

} // core

} // ml

