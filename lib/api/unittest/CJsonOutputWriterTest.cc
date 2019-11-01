/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CScopedRapidJsonPoolAllocator.h>
#include <core/CSmallVector.h>
#include <core/CTimeUtils.h>

#include <maths/CTools.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CStringStore.h>
#include <model/ModelTypes.h>

#include <api/CJsonOutputWriter.h>

#include <test/BoostTestCloseAbsolute.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>
