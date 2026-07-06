@ECHO OFF
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.Sphinx was not found. Install the 'docs' extra: pip install -e ".[docs]"
	exit /b 1
)
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end
:end
