/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "CMockDataProcessor.h"

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>
