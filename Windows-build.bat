@echo off
echo ==========================================
echo    SuperWeb Cluster - Compiler
echo ==========================================

echo.
echo [1/3] Looking for Visual Studio 
set "VSDEVCMD="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find Common7\Tools\VsDevCmd.bat`) do set "VSDEVCMD=%%i"
)
if not defined VSDEVCMD set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
call "%VSDEVCMD%" -arch=x64 -host_arch=x64

echo.
echo [2/3] CPU 
cd /d D:\Code\Py\SuperWeb\compute_node\performance_metrics\conv2d_runners\cpu\windows
if not exist build mkdir build
cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:build\fmvm_cpu_windows.exe fmvm_cpu_windows.cpp

echo.
echo [3/3]  CUDA 
cd /d D:\Code\Py\SuperWeb\compute_node\performance_metrics\conv2d_runners\cuda
if not exist build mkdir build
nvcc fmvm_cuda_runner.cu -O3 --use_fast_math -std=c++17 -o build\fmvm_cuda_runner.exe

echo.
echo ==========================================
echo Finished
pause
