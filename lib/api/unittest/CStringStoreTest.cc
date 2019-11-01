/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>
#include <model/CStringStore.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>

#include "CMockDataAdder.h"
#include "CMockSearcher.h"

#include <boost/test/unit_test.hpp>

#include <sstream>
