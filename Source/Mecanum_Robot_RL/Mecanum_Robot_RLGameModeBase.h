#pragma once

#include "CoreMinimal.h"

#include "GameFramework/GameModeBase.h"
#include "Public/BaseEnvironment.h"
#include "Public/RLRunner.h"
#include "Mecanum_Robot_RL.h"
#include "CubeEnvironment.h"
#include "RLRunner.h"
// #include "TorchPlugin/Public/TorchPlugin.h"
#include "Public/SharedMemoryAgentCommunicator.h"
#include "Mecanum_Robot_RLGameModeBase.generated.h"

UCLASS()
class AMecanum_Robot_RLGameModeBase : public AGameModeBase
{
    GENERATED_BODY()

public:
    AMecanum_Robot_RLGameModeBase();

    virtual void BeginDestroy() override;
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;
    virtual void InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage) override;

    TArray<FVector> CreateGridLocations(int32 NumEnvironments, FVector Offset);

private:

    ARLRunner* Runner;
    TArray<FBaseInitParams*> InitParamsArray;
    USharedMemoryAgentCommunicator* AgentComm;
};
