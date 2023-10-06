#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "MaterialShared.h"
#include "RLTypes.h"
#include "CubeEnvironment.generated.h"

// Derived struct for initialization parameters specific to CubeEnvironment
USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FCubeEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector GroundPlaneSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ControlledCubeSize;
};

UCLASS()
class MECANUM_ROBOT_RL_API ACubeEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    

    // Sets default values for this actor's properties
    ACubeEnvironment();

    // The cube that this agent controls
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* ControlledCube;

    // The plane that this agent operates on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* GroundPlane;
    
    // The goal the agent move towards
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* GoalObject;

    // Override the InitEnv function to accept the derived struct
    virtual void InitEnv(FBaseInitParams *Params) override;

    // Reset the environment and return the initial state
    virtual FState ResetEnv() override;

    // Perform a step in the environment using the given action
    virtual TTuple<bool, float, FState> Step(FAction Action) override;

private:
    FCubeEnvironmentInitParams* CubeParams = nullptr;

    FVector GoalLocation;

    // Calculates the reward for the current state
    virtual float Reward() override;

    // Creates random location on environments ground plane
    FVector GenerateRandomLocationOnPlane();

    // Creates random spawn location for the cube
    FVector GenerateRandomLocationCube();

    float MaxAngularSpeed = 90.0f;
    float MaxLinearSpeed = 1.0f;
};
