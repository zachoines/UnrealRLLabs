Setting up python debugging in Unreal Engine

0. Created a new Unreal Engine Plugin called "DebugpyPythonPackage", installed it to UnrealRLLabs project

1. Ran the following commands in widows command prompt:
    a. "C:/Program Files/Epic Games/UE_5.4/Engine/Binaries/ThirdParty/Python3/Win64/python.exe" -m ensurepip
    b. "C:/Program Files/Epic Games/UE_5.4/Engine/Binaries/ThirdParty/Python3/Win64/python.exe" -m pip install --target="C:\Users\zachoines\Documents\Unreal\UnrealRLLabs\Plugins\DebugpyPythonPackage\Content\Python" debugpy

2. Then in the unreal engine Python (REFL) command line
    a. import unreal 
    b. import debugpy
    c. import debugpy_unreal
    d. debugpy_unreal.start()

3. This then freezes the Unreal Engine, which is now waiting for remote debug session. So, iN visual studio I navigated to my project directory at "UnrealRLLabs", then setup my launch.json as below:

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Unreal RL Labs Python",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "C:\\Program Files\\Epic Games\\UE_5.4\\Engine\\Plugins\\Experimental\\PythonScriptPlugin\\Content\\Python",
                    "remoteRoot": "C:\\Program Files\\Epic Games\\UE_5.4\\Engine\\Plugins\\Experimental\\PythonScriptPlugin\\Content\\Python"
                },
                {
                    "localRoot": "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python",
                    "remoteRoot": "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python"
                }
            ],
            "redirectOutput": true,
            "justMyCode": false,
        }
    ]
}",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "C:\\Program Files\\Epic Games\\UE_5.4\\Engine\\Plugins\\Experimental\\PythonScriptPlugin\\Content\\Python",
                    "remoteRoot": "C:\\Program Files\\Epic Games\\UE_5.4\\Engine\\Plugins\\Experimental\\PythonScriptPlugin\\Content\\Python"
                },
                {
                    "localRoot": "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python",
                    "remoteRoot": "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python"
                }
            ],
            "redirectOutput": true,
            "justMyCode": false,
        }
    ]
}

4. Once this launch.json was setup I hit "Debug Unreal RL Labs Python" to start debug mode.

5. Then, going back to the editor to the same Python (REFL) command line instance, I entered.
    a. from Train import *
    b. main()

