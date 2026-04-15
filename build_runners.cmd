@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "CPU_SRC=%ROOT%compute_node\performance_metrics\conv2d_runners\cpu\windows"
set "CPU_BUILD=%CPU_SRC%\build"
set "CUDA_SRC=%ROOT%compute_node\performance_metrics\conv2d_runners\cuda"
set "CUDA_BUILD=%CUDA_SRC%\build"

set CPU_OK=0
set CUDA_OK=0

echo.
echo === Finding Visual Studio ===

set "VSDEVCMD="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALL=%%i"
    )
)
if defined VSINSTALL (
    set "VSDEVCMD=!VSINSTALL!\Common7\Tools\VsDevCmd.bat"
)

if not exist "%VSDEVCMD%" (
    echo [ERROR] Visual Studio not found.
    goto :try_cuda
)

echo Found VS: %VSDEVCMD%
call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul 2>&1

echo.
echo === Building CPU Runner ===

if not exist "%CPU_BUILD%" mkdir "%CPU_BUILD%"
pushd "%CPU_BUILD%"
cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:fmvm_cpu_windows.exe ..\fmvm_cpu_windows.cpp
set CPU_EXIT=%ERRORLEVEL%
popd

if %CPU_EXIT% equ 0 (
    echo [OK] CPU runner built
    set CPU_OK=1
) else (
    echo [FAIL] CPU runner failed
)

:try_cuda
echo.
echo === Building CUDA Runner ===

where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [SKIP] nvcc not in PATH
    goto :done
)

if not exist "%CUDA_BUILD%" mkdir "%CUDA_BUILD%"
pushd "%CUDA_BUILD%"
nvcc -std=c++17 -O3 -o fmvm_cuda_runner.exe ..\fmvm_cuda_runner.cu
set CUDA_EXIT=%ERRORLEVEL%
popd

if %CUDA_EXIT% equ 0 (
    echo [OK] CUDA runner built
    set CUDA_OK=1
) else (
    echo [FAIL] CUDA runner failed
)

:done
echo.
echo === SUMMARY ===
if %CPU_OK% equ 1 (echo CPU:  OK) else (echo CPU:  FAILED)
if %CUDA_OK% equ 1 (echo CUDA: OK) else (echo CUDA: FAILED)
echo.

endlocal
