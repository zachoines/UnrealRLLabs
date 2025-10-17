// Copyright Epic Games, Inc. All Rights Reserved.

#include "UnrealRLLabs.h"
#include "Modules/ModuleManager.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "UObject/UObjectGlobals.h"
#include "UnrealRLLabs.h"

// Internal contact filter registration API
#include "Physics/EnvContactFilter.h"

DEFINE_LOG_CATEGORY(LOG_UNREALRLLABS);

class FUnrealRLLabsModule : public FDefaultGameModuleImpl
{
public:
    virtual void StartupModule() override
    {
        PostWorldInitHandle = FWorldDelegates::OnPostWorldInitialization.AddRaw(this, &FUnrealRLLabsModule::OnPostWorldInit);
        WorldCleanupHandle = FWorldDelegates::OnWorldCleanup.AddRaw(this, &FUnrealRLLabsModule::OnWorldCleanup);
    }

    virtual void ShutdownModule() override
    {
        if (PostWorldInitHandle.IsValid())
        {
            FWorldDelegates::OnPostWorldInitialization.Remove(PostWorldInitHandle);
            PostWorldInitHandle.Reset();
        }
        if (WorldCleanupHandle.IsValid())
        {
            FWorldDelegates::OnWorldCleanup.Remove(WorldCleanupHandle);
            WorldCleanupHandle.Reset();
        }
    }

private:
    FDelegateHandle PostWorldInitHandle;
    FDelegateHandle WorldCleanupHandle;

    void OnPostWorldInit(UWorld* World, const UWorld::InitializationValues IV)
    {
        if (World && World->IsGameWorld())
        {
            FEnvContactFilter::Register(World);
            UE_LOG(LOG_UNREALRLLABS, Log, TEXT("Registered Env contact filter for world %s"), *World->GetName());
        }
    }

    void OnWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources)
    {
        if (World && World->IsGameWorld())
        {
            FEnvContactFilter::Unregister(World);
            UE_LOG(LOG_UNREALRLLABS, Log, TEXT("Unregistered Env contact filter for world %s"), *World->GetName());
        }
    }
};

IMPLEMENT_PRIMARY_GAME_MODULE(FUnrealRLLabsModule, UnrealRLLabs, "UnrealRLLabs");
