#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"

// JSON and file utilities.
#include "Misc/FileHelper.h"
#include "Dom/JsonValue.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

// Local classes.
#include "BaseEnvironment.h"
#include "RLRunner.h"
#include "TerraShiftEnvironment.h"
#include "SharedMemoryAgentCommunicator.h"
#include "EnvironmentConfig.h"

#include "Engine/TextureRenderTarget2D.h"
#include "UnrealRLLabsGameModeBase.generated.h"

UCLASS()
class UNREALRLLABS_API AUnrealRLLabsGameModeBase : public AGameModeBase
{
    GENERATED_BODY()

public:
    AUnrealRLLabsGameModeBase();

    virtual void BeginDestroy() override;
    virtual void BeginPlay() override;
    virtual void InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage) override;

    /**
     * Helper to create a set of environment spawn locations arranged in a 3D grid pattern.
     */
    TArray<FVector> CreateGridLocations(int32 NumEnvironments, FVector Offset);

private:
    /**
     * A pointer to our RL runner responsible for stepping/training.
     */
    UPROPERTY()
    ARLRunner* Runner;

    /**
     * Array of init-params for each environment instance we will spawn.
     */
    TArray<FBaseInitParams*> InitParamsArray;

    /**
     * Our loaded JSON config, wrapped by UEnvironmentConfig.
     */
    UPROPERTY()
    UEnvironmentConfig* EnvConfig;
};
