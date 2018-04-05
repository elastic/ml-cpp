@echo off
rem
rem ELASTICSEARCH CONFIDENTIAL
rem
rem Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
rem
rem Notice: this software, and all information contained
rem therein, is the exclusive property of Elasticsearch BV
rem and its licensors, if any, and is protected under applicable
rem domestic and foreign law, and international treaties.
rem
rem Reproduction, republication or distribution without the
rem express written consent of Elasticsearch BV is
rem strictly prohibited.
rem

rem The Windows part of ML C++ CI:
rem
rem 1. Build and unit test the Windows version of the C++
rem 2. Upload the build to the artifacts directory on S3 that
rem    subsequent Java builds will download the C++ components from

setlocal enableextensions

rem Change directory to the top level of the repo
cd %~dp0
cd ..

rem Ensure 3rd party dependencies are installed
powershell.exe -ExecutionPolicy RemoteSigned -File dev-tools\download_windows_deps.ps1 || exit /b %ERRORLEVEL%

rem Run the build and unit tests
set ML_KEEP_GOING=1
call .\gradlew.bat --info clean buildZip buildZipSymbols check || exit /b %ERRORLEVEL%

rem Upload the artifacts to S3
call .\gradlew.bat --info -b upload.gradle upload || exit /b %ERRORLEVEL%

endlocal

