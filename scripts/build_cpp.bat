@echo off
echo ============================================================
echo Building C++ Order Book
echo ============================================================
echo.

REM Create build directory
if not exist "cpp\build" mkdir cpp\build
cd cpp\build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed
    cd ..\..
    exit /b 1
)

echo.
echo Building...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed
    cd ..\..
    exit /b 1
)

echo.
echo Installing...
cmake --install .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Installation failed
    cd ..\..
    exit /b 1
)

cd ..\..

echo.
echo ============================================================
echo Build Complete!
echo ============================================================
echo.
echo Testing import...
python -c "import python.fast_orderbook; print('✓ C++ module imported successfully!')"

echo.
echo To use: import fast_orderbook
echo.