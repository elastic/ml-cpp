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
#include <core/CStrPTime.h>


namespace ml
{
namespace core
{


char *CStrPTime::strPTime(const char *buf,
                          const char *format,
                          struct tm *tm)

{
    return ::strptime(buf, format, tm);
}


}
}

