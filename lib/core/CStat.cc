/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStat.h>

namespace ml
{
namespace core
{

CStat::CStat() : m_Value(uint64_t(0))
{
}

void CStat::increment()
{
    m_Value.fetch_add(1);
}

void CStat::increment(uint64_t value)
{
    m_Value.fetch_add(value);
}

void CStat::decrement()
{
    m_Value.fetch_sub(1);
}

void CStat::set(uint64_t value)
{
    m_Value.store(value);
}

uint64_t CStat::value() const
{
    return m_Value;
}

} // core

} // ml

