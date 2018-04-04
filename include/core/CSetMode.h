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
#ifndef INCLUDED_ml_core_CSetMode_h
#define INCLUDED_ml_core_CSetMode_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {

//! \brief
//! Portable wrapper for the Windows _setmode(fd, mode) function.
//!
//! DESCRIPTION:\n
//! Set the input streams translation mode.
//! Input streams can be in various text modes or binary.
//! In binary mode end-of-line translation does not take place
//! and ascii character 26 (end-of-transmission) is ignored.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because it is
//! a Windows specific function. There is no need for an equivalent
//! call on *nix as *nix input streams don't interperet ascii
//! character code 26 as end of transmission (the character send by
//! pressing Ctrl D on Windows).
//!
class CORE_EXPORT CSetMode : private CNonInstantiatable {
public:
    static int setMode(int fd, int mode);
    static int setBinaryMode(int fd);
};
}
}

#endif // INCLUDED_ml_core_CSetMode_h
