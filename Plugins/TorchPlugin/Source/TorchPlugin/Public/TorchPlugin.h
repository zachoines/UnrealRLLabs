#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "ITorchWrapper.h"
#include <Windows.h>

#ifdef TORCHPLUGIN_EXPORTS
#define TORCHPLUGIN_API __declspec(dllexport)
#else
#define TORCHPLUGIN_API __declspec(dllimport)
#endif

class TORCHPLUGIN_API FTorchPlugin : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /** Initialize the plugin and load the DLL */
    void Init();

    /** Run a test using the TorchWrapper */
    void RunAgentTest();

    /** Get the TorchWrapper instance */
    ITorchWrapper* GetTorchWrapper();

private:
    /** Import the TorchWrapper DLL */
    // bool ImportDLL(FString FolderName, FString DLLName);

    /** Import the necessary methods from the DLL */
    bool ImportMethods();

private:
    ITorchWrapper* TorchWrapperInstance = nullptr;
    void* v_dllHandle = nullptr;
    HMODULE hModule;

    // Function pointers to the DLL methods
    ITorchWrapper* (*m_funcCreateInstance)() = nullptr;
    void (*m_funcDestroyInstance)(ITorchWrapper*) = nullptr;
};
