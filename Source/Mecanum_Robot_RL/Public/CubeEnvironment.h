#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "MaterialShared.h"
#include "RLTypes.h"
#include "CubeEnvironment.generated.h"


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
    virtual FState ResetEnv(int NumAgents) override;

    // Update actors in environment with provided actions
    virtual void Act(FAction Action) override;

    // Returns the public view of the state. Called after Update 
    virtual FState State() override;

    // Returns done conditon
    virtual bool Done() override;

    // Returns truncation conditon
    virtual bool Trunc() override;

    // Calculate the reward for the current state
    virtual float Reward() override;

private:
    FCubeEnvironmentInitParams* CubeParams = nullptr;

    // Movement Speed
    float MaxAngularSpeed = 720.0f;
    float MaxLinearSpeed = 400.0f;
    int maxStepsPerEpisode = 128;

    // Internally held state
    int currentUpdate;
    bool CubeNearGoal;
    bool CubeOffGroundPlane;
    float CubeDistToGoal;
    float GoalRadius;
    FVector GroundPlaneSize;
    FVector CubeSize;
    FVector GroundPlaneCenter;
    FTransform GroundPlaneTransform;
    FTransform InverseGroundPlaneTransform;
    FVector CubeLocationRelativeToGround;
    FVector GoalLocationRelativeToGround;
    FVector CubeWorldLocation;
    FVector GoalWorldLocation;
    FRotator CubeWorldRotation;

    // Determines if the Cube goes beyoud bounds of environment
    bool IsCubeOffGroundPlane();

    // Creates random location on environments ground plane
    FVector GenerateRandomLocationOnPlane();

    // Creates random spawn location for the cube
    FVector GenerateRandomLocationCube();
};
