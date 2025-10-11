@ECHO OFF

REM Command file for Sphinx documentation

set SPHINXOPTS=
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto help

if "%1"=="clean" (
    rmdir /S /Q %BUILDDIR%
    goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
EXIT /B %ERRORLEVEL%
