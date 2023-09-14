#include "TorchPlugin.h"

#define LOCTEXT_NAMESPACE "FTorchPlugin"

void FTorchPlugin::StartupModule()
{
    // This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
    Init();
    int test = 0;
}

void FTorchPlugin::ShutdownModule()
{
    // This function may be called during shutdown to clean up your module.
    if (TorchWrapperInstance)
    {
        m_funcDestroyInstance(TorchWrapperInstance);
    }
}

bool FTorchPlugin::ImportMethods()
{
    // Load the DLL
    void* dllHandle = nullptr;
    try
    {
        dllHandle = FPlatformProcess::GetDllHandle(TEXT("TorchWrapper.dll"));
    }
    catch (const std::exception& e)
    {
        UE_LOG(LogTemp, Error, TEXT("Exception while loading DLL: %s"), *FString(e.what()));
    }
    catch (...)
    {
        UE_LOG(LogTemp, Error, TEXT("Unknown exception while loading DLL"));
    }

   /* void* dllHandle = nullptr;
    hModule = LoadLibrary(TEXT("TorchWrapper.dll"));*/
    if (!dllHandle)
    {
        int32 errorCode = FPlatformMisc::GetLastError();
        UE_LOG(LogTemp, Error, TEXT("Failed to load TorchWrapper.dll. Error code: %d"), errorCode);
        return false;
    }

    // Get the function pointers
    m_funcCreateInstance = (ITorchWrapper * (*)())FPlatformProcess::GetDllExport(dllHandle, TEXT("CreateTorchWrapperInstance"));
    if (!m_funcCreateInstance)
    {
        int32 errorCode = FPlatformMisc::GetLastError();
        UE_LOG(LogTemp, Error, TEXT("Failed to get CreateTorchWrapperInstance from TorchWrapper.dll. Error code: %d"), errorCode);
        return false;
    }

    m_funcDestroyInstance = (void(*)(ITorchWrapper*))FPlatformProcess::GetDllExport(dllHandle, TEXT("DestroyTorchWrapperInstance"));
    if (!m_funcDestroyInstance)
    {
        int32 errorCode = FPlatformMisc::GetLastError();
        UE_LOG(LogTemp, Error, TEXT("Failed to get DestroyTorchWrapperInstance from TorchWrapper.dll. Error code: %d"), errorCode);
        return false;
    }

    return true;
}


void FTorchPlugin::Init()
{
    if (ImportMethods())
    {
        TorchWrapperInstance = m_funcCreateInstance();
        if (!TorchWrapperInstance)
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create TorchWrapper instance."));
        }
    }
}

ITorchWrapper* FTorchPlugin::GetTorchWrapper()
{
    return TorchWrapperInstance;
}

void FTorchPlugin::RunAgentTest()
{
    if (TorchWrapperInstance)
    {
        // Initialize and use the TorchWrapper functions
        bool success = TorchWrapperInstance->Init(4, 2);
        if (success)
        {
            UE_LOG(LogTemp, Warning, TEXT("Agent initialized successfully."));

            std::vector<std::vector<float>> states = { {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f} };
            auto actions = TorchWrapperInstance->GetActions(states);

            for (const auto& action : actions)
            {
                FString actionStr = "Action: ";
                for (float a : action)
                {
                    actionStr += FString::Printf(TEXT("%f "), a);
                }
                UE_LOG(LogTemp, Warning, TEXT("%s"), *actionStr);
            }

            TorchWrapperInstance->Train();
            UE_LOG(LogTemp, Warning, TEXT("Training completed."));
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to initialize the agent."));
        }
    }
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FTorchPlugin, TorchPlugin)
