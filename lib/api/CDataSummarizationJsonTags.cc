/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataSummarizationJsonTags.h>

namespace ml {
namespace api {
// clang-format off
const std::string CDataSummarizationJsonTags::JSON_COMPRESSED_DATA_SUMMARIZATION_TAG{"compressed_data_summarization"};
const std::string CDataSummarizationJsonTags::JSON_DATA_SUMMARIZATION_TAG{"data_summarization"};
const std::string CDataSummarizationJsonTags::JSON_NUM_COLUMNS_TAG{"num_columns"};
const std::string CDataSummarizationJsonTags::JSON_COLUMN_NAMES_TAG{"column_names"};
const std::string CDataSummarizationJsonTags::JSON_COLUMN_IS_CATEGORICAL_TAG{"column_is_categorical"};
const std::string CDataSummarizationJsonTags::JSON_CATEGORICAL_COLUMN_VALUES_TAG{"categorical_column_values"};
const std::string CDataSummarizationJsonTags::JSON_ENCODING_NAME_INDEX_MAP_TAG{"encodings_indices"};
const std::string CDataSummarizationJsonTags::JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG{"encoding_name"};
const std::string CDataSummarizationJsonTags::JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG{"encoding_index"};
const std::string CDataSummarizationJsonTags::JSON_ENCODINGS_TAG{"encodings"};
const std::string CDataSummarizationJsonTags::JSON_DATA_TAG{"data"};
// clang-format on
}
}
