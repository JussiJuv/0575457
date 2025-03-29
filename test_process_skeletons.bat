@echo off
setlocal enabledelayedexpansion

rem ***** Please change the following paths to match your own directories *****
rem OPENPOSE_PATH should be your OpenPose installation directory
rem BASE_INPUT should be the file where the testing folders are located
rem BASE_OUTPUT should be where you want to the skeletons to be stored at after extracting, recommended to use 0575457/testing_skeleton
rem ***** Remember to change paths in config.py aswell *****
set OPENPOSE_PATH=C:\openpose
set BASE_INPUT=C:\koulua\MVDIA\Project\project_3split\testing
set BASE_OUTPUT=C:\koulua\MVDIA\Project\project_3split\testing_skeleton

rem Verify and run OpenPose
cd /d "%OPENPOSE_PATH%"
for /L %%x in (1,1,32) do (
    echo Processing class %%x...
    set INPUT_DIR=%BASE_INPUT%\%%x
    set CLASS_OUTPUT=%BASE_OUTPUT%\%%x\json
    if not exist "!CLASS_OUTPUT!" mkdir "!CLASS_OUTPUT!"
    if exist "!INPUT_DIR!" (
        bin\OpenPoseDemo.exe ^
          --image_dir "!INPUT_DIR!" ^
          --write_json "!CLASS_OUTPUT!" ^
          --model_pose BODY_25 ^
          --net_resolution 320x176 ^
          --number_people_max 1 ^
          --disable_blending ^
          --render_pose 0
    )
)
echo Testing skeletons processed!
pause