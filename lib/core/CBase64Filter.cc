/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBase64Filter.h>

namespace ml
{
namespace core
{

CBase64Encoder::CBase64Encoder() : m_Buffer(4096)
{
}

CBase64Encoder::~CBase64Encoder()
{
}

CBase64Decoder::CBase64Decoder() : m_BufferIn(4096), m_BufferOut(4096), m_Eos(false)
{
}

CBase64Decoder::~CBase64Decoder()
{
}

} // core
} // ml
