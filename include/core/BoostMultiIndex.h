/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_BoostMultiIndex_h
#define INCLUDED_ml_core_BoostMultiIndex_h

//! \brief
//! Boost headers + extra MACRO for boost multi-index objects

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>

//! Define a member function that returns a const reference
#define BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(Class,Type,MemberFunName) \
::boost::multi_index::const_mem_fun< Class, const Type &, &Class::MemberFunName >

#endif // INCLUDED_ml_core_BoostMultiIndex_h

