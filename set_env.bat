@echo off
REM
REM Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
REM or more contributor license agreements. Licensed under the Elastic License
REM 2.0 and the following additional limitation. Functionality enabled by the
REM files subject to the Elastic License 2.0 may only be used in production when
REM invoked by an Elasticsearch process with a license key installed that permits
REM use of machine learning features. You may not use this file except in
REM compliance with the Elastic License 2.0 and the foregoing additional
REM limitation.
REM

REM Set up a build environment, to ensure repeatable builds

REM Initialize the Visual Studio command prompt environment variables
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64

REM Set %CPP_SRC_HOME% to be an absolute path to this script's location, as
REM different builds will come from different repositories and go to different
REM staging areas
SET CPP_SRC_HOME=%~dp0

REM Logical filesystem root
SET ROOT=%CD:~0,2%

SET PATH=%ROOT%/PROGRA~1/CMake/bin;%CPP_SRC_HOME%/build/distribution/platform/windows-x86_64/bin;%PATH%
		
SET BOOST_ROOT=%ROOT%/usr/local
