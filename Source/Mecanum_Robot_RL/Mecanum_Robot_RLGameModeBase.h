#pragma once
#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"

// JSON parsing 
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Misc/FileHelper.h"

// Local classes
#include "Public/BaseEnvironment.h"
#include "Public/RLRunner.h"
#include "Mecanum_Robot_RL.h"
#include "MultiAgentCubeEnvironment.h"
#include "TerraShiftEnvironment.h"
#include "RLRunner.h"
#include "Public/SharedMemoryAgentCommunicator.h"
#include "Public/RLTypes.h"
#include "Mecanum_Robot_RLGameModeBase.generated.h"

UCLASS()
class AMecanum_Robot_RLGameModeBase : public AGameModeBase
{
    GENERATED_BODY()

public:
    AMecanum_Robot_RLGameModeBase();

    virtual void BeginDestroy() override;
    virtual void BeginPlay() override;
    virtual void InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage) override;

    TArray<FVector> CreateGridLocations(int32 NumEnvironments, FVector Offset);

private:

    ARLRunner* Runner;
    TArray<FBaseInitParams*> InitParamsArray;
    FTrainParams TrainParams;

    bool ReadJsonConfig(const FString& FilePath, FTrainParams& OutTrainParams);
};
