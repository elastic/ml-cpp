@echo off
rem
rem Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
rem or more contributor license agreements. Licensed under the Elastic License
rem 2.0 and the following additional limitation. Functionality enabled by the
rem files subject to the Elastic License 2.0 may only be used in production when
rem invoked by an Elasticsearch process with a license key installed that permits
rem use of machine learning features. You may not use this file except in
rem compliance with the Elastic License 2.0 and the foregoing additional
rem limitation.
rem

rem Set up a build environment, to ensure repeatable builds

rem Initialize the Visual Studio command prompt environment variables
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64

rem Set %CPP_SRC_HOME% to be an absolute path to this script's location, as
rem different builds will come from different repositories and go to different
rem staging areas
set CPP_SRC_HOME=%~dp0

rem Assume the drive letter where our 3rd party dependencies are installed under
rem \usr\local is the current drive at the time this script is run
set ROOT=%CD:~0,2%

set PATH=C:\Program Files\CMake\bin;%CPP_SRC_HOME%\build\distribution\platform\windows-x86_64\bin;%PATH%

set INCLUDE=
set LIBPATH=
