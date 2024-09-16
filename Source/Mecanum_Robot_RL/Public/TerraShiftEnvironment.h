#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "RLTypes.h"
#include "TerraShift/Column.h"
#include "TerraShift/GridObject.h"  // Include the GridObject class
#include "TerraShiftEnvironment.generated.h"

// Struct for initialization parameters specific to TerraShiftEnvironment
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float GroundPlaneSize = 2.0f; // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float MaxColumnHeight = 10.0f; // cm

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = { 0.1f, 0.1f, 0.1f }; // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = 0.2f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnMass = 0.01f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnAccelConstant = 0.2f; // Acceleration constant for columns

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int GridSize = 40; // Size of the grid

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxSteps = 1024; // Maximum steps per episode

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int NumGoals = 3; // Number of goals for agents

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float SpawnDelay = 1.0f; // Delay between each GridObject spawn

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxAgents = 10; // Maximum number of agents
};

// Array of colors for goals
const TArray<FLinearColor> GoalColors = {
    FLinearColor(1.0f, 0.0f, 0.0f),
    FLinearColor(0.0f, 1.0f, 0.0f),
    FLinearColor(0.0f, 0.0f, 1.0f),
    FLinearColor(1.0f, 1.0f, 0.0f),
    FLinearColor(1.0f, 0.0f, 1.0f),
    FLinearColor(0.0f, 1.0f, 1.0f),
    FLinearColor(1.0f, 0.5f, 0.0f),
    FLinearColor(0.5f, 0.0f, 1.0f),
    FLinearColor(1.0f, 0.0f, 0.5f),
    FLinearColor(0.5f, 1.0f, 0.0f),
};

UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    ATerraShiftEnvironment();

    // The root component for organizing everything in this environment
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    USceneComponent* TerraShiftRoot;

    // The platform that agents operate on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* Platform;

    // The objects controlled by agents, now using the GridObject class
    UPROPERTY(EditAnywhere)
    TArray<AGridObject*> Objects;

    // The columns controlled by GridObjects, now using the Column class
    UPROPERTY(EditAnywhere)
    TArray<AColumn*> Columns;

    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PostTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;

private:
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;
    int MaxAgents;
    int CurrentStep;
    int CurrentAgents;
    TArray<int> LastColumnIndexArray;
    TArray<FVector> GoalPositionArray;
    TArray<int32> AgentGoalIndices;
    TArray<FVector> GridCenterPoints;

    // Function to spawn the platform in the environment
    AStaticMeshActor* SpawnPlatform(FVector Location, FVector Size);

    // Function to "spawn" grid object with a delay
    void SetSpawnGridObject(int AgentIndex, float Delay, FVector Location);

    // Function to select the next column for an agent based on direction
    int SelectColumn(int AgentIndex, int Direction) const;

    // Function to get the current state of an agent
    TArray<float> AgentGetState(int AgentIndex);

    // Helper function to convert a 2D grid point to a 1D index
    int Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const;

    // Helper function to check if the object has moved off the platform
    bool ObjectOffPlatform(int AgentIndex) const;

    // Helper function to check if the object has reached the wrong goal
    bool ObjectReachedWrongGoal(int AgentIndex) const;

    // Function to map a value between two ranges
    float Map(float x, float in_min, float in_max, float out_min, float out_max) const;

    // Sets number of currently active GridObjects to random locations
    void SetActiveGridObjects(int NumAgents);
};
