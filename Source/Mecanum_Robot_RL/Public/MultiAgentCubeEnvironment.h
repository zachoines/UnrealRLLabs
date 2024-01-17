#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "MaterialShared.h"
#include "RLTypes.h"
#include "MultiAgentCubeEnvironment.generated.h"


// Derived struct for initialization parameters specific to CubeEnvironment
USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FMultiAgentCubeEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector GroundPlaneSize = FVector::One() * 5.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ControlledCubeSize = FVector::One() * 0.25;
};


UCLASS()
class MECANUM_ROBOT_RL_API AMultiAgentCubeEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:

    // Sets default values for this actor's properties
    AMultiAgentCubeEnvironment();

    // The cube that this agent controls
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* ControlledCube;

    // The goal the agent move towards
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* GoalObject;

    // The cube that this agent controls
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> ControlledCubes;

    // The plane that this agent operates on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* GroundPlane;

    // The goal the agent move towards
    UPROPERTY(EditAnywhere)
    TArray<AStaticMeshActor*> GoalObjects;

    // Override the InitEnv function to accept the derived struct
    virtual void InitEnv(FBaseInitParams* Params) override;

    // Reset the environment and return the initial state
    virtual FState ResetEnv(int NumAgents) override;

    // Update actors in environment with provided actions
    virtual void Act(FAction Action) override;

    // Updates the internally held state of the environment.
    virtual void Update() override;

    // Returns the public view of the state. Called after Update 
    virtual FState State() override;

    // Returns done conditonz
    virtual bool Done() override;

    // Returns truncation conditon
    virtual bool Trunc() override;

    // Calculate the reward for the current state
    virtual float Reward() override;

    void setCurrentAgents(int NumAgents);

private:
    FMultiAgentCubeEnvironmentInitParams* MultiAgentCubeParams = nullptr;
    float GoalRadius;
    FVector GroundPlaneSize;
    FVector CubeSize;
    FVector GroundPlaneCenter;
    
    // Constants
    const int GridSize = 15;
    const int MaxSteps = 64;
    const float AgentVisibility = 3;
    const float MaxAgents = 10;

    // State Variables
    int CurrentStep;
    int CurrentAgents;
    FTransform GroundPlaneTransform;
    FTransform InverseGroundPlaneTransform;
    TArray<TArray<FVector>> GridCenterPoints;

    AStaticMeshActor* InitializeCube();
    AStaticMeshActor* InitializeGoalObject();
    AStaticMeshActor* InitializeCube(const FLinearColor& Color);
    AStaticMeshActor* InitializeGoalObject(const FLinearColor& Color);
    const TArray<FLinearColor> Colors = {
        FLinearColor(1.0f, 0.0f, 0.0f),
        FLinearColor(0.0f, 1.0f, 0.0f),
        FLinearColor(0.0f, 0.0f, 1.0f),
        FLinearColor(1.0f, 1.0f, 0.0f),
        FLinearColor(1.0f, 0.0f, 1.0f),
        FLinearColor(0.0f, 1.0f, 1.0f),
        FLinearColor(1.0f, 0.5f, 0.0f),
        FLinearColor(0.5f, 0.0f, 1.0f),
        FLinearColor(1.0f, 0.0f, 0.5f),
        FLinearColor(0.5f, 1.0f, 0.0f) 
    };
    
    TMap<FIntPoint, TArray<AStaticMeshActor*>> UsedLocations;
    TMap<AStaticMeshActor*, FIntPoint> ActorToLocationMap;
    TMap<int, TPair<FIntPoint, FIntPoint>> AgentGoalPositions;

    void AssignRandomGridLocations();
    FIntPoint GenerateRandomLocation();
    FVector GetWorldLocationFromGridIndex(FIntPoint GridIndex);

    void MoveAgent(int AgentIndex, FIntPoint Location);
    void MoveGoal(int AgentIndex, FIntPoint Location);
    void AgentGoalReset(int AgentIndex);
    void GoalReset(int AgentIndex);
    void AgentReset(int AgentIndex);
    bool AgentGoalReached(int AgentIndex);
    bool AgentHasCollided(int AgentIndex);
    bool AgentOutOfBounds(int AgentIndex);
    TArray<float> AgentGetState(int AgentIndex);
    int Get1DIndexFromPoint(const FIntPoint& point, int gridSize);
    float GridDistance(const FIntPoint& Point1, const FIntPoint& Point2);
};

