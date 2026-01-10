@echo off
echo ============================================================
echo CLEANING UP C++ BUILD FILES
echo ============================================================
echo.

REM Remove C++ build directory
if exist "cpp\build" (
    echo Removing cpp\build...
    rmdir /s /q cpp\build
    echo ✓ Removed cpp\build
)

REM Remove any compiled .pyd files in python folder
if exist "python\fast_orderbook.pyd" (
    echo Removing python\fast_orderbook.pyd...
    del python\fast_orderbook.pyd
    echo ✓ Removed fast_orderbook.pyd
)

REM Remove any .so files (Linux)
if exist "python\fast_orderbook.so" (
    del python\fast_orderbook.so
    echo ✓ Removed fast_orderbook.so
)

REM Remove CMake cache files in cpp folder
if exist "cpp\CMakeCache.txt" (
    del cpp\CMakeCache.txt
    echo ✓ Removed CMakeCache.txt
)

if exist "cpp\CMakeFiles" (
    rmdir /s /q cpp\CMakeFiles
    echo ✓ Removed CMakeFiles
)

echo.
echo ============================================================
echo CLEANUP COMPLETE
echo ============================================================
echo.
echo C++ build files removed. Project is now Python-only.
echo.
pause