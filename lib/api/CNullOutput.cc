/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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
#include <api/CNullOutput.h>


namespace ml
{
namespace api
{

bool CNullOutput::fieldNames(const TStrVec &/*fieldNames*/,
                             const TStrVec &/*extraFieldNames*/)
{
    return true;
}

const COutputHandler::TStrVec &CNullOutput::fieldNames() const
{
    return EMPTY_FIELD_NAMES;
}

bool CNullOutput::writeRow(const TStrStrUMap &/*dataRowFields*/,
                           const TStrStrUMap &/*overrideDataRowFields*/)
{
    return true;
}

}
}

