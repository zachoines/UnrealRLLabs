// ATerraShiftEnvironment.h

#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "RLTypes.h"
#include "TerraShift/Column.h"
#include "TerraShift/GridObject.h"
#include "TerraShift/MorletWavelets2D.h"
#include "TerraShiftEnvironment.generated.h"

// Struct for initialization parameters specific to TerraShiftEnvironment
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float GroundPlaneSize = 2.0f; // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float MaxColumnHeight =2.0f; // cm

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnHeight = 1.0f; // Relative scaler

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = FVector(0.1f, 0.1f, 0.1f); // m

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = 0.2f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ColumnMass = 0.01f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int GridSize = 20; // Size of the grid

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxSteps = 1024; // Maximum steps per episode

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int NumGoals = 3; // Number of goals for agents

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float SpawnDelay = 1.0f; // Delay between each GridObject spawn

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxAgents = 10; // Maximum number of agents

    // Agent wave parameter ranges
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D AmplitudeRange = FVector2D(5.0f, 10.0f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WaveOrientationRange = FVector2D(0.0f, 2 * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WavenumberRange = FVector2D(0.2f, 0.5f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D PhaseRange = FVector2D(0.0f, 2 * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D SigmaRange = FVector2D(2.0f, 10.0f);

    // Agent movement parameter ranges
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Movement Parameters")
    FVector2D VelocityRange = FVector2D(-3.0f, 3.0f);
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
    virtual ~ATerraShiftEnvironment();

    // The root component for organizing everything in this environment
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    USceneComponent* TerraShiftRoot;

    // The platform that agents operate on
    UPROPERTY(EditAnywhere)
    AStaticMeshActor* Platform;

    // The objects controlled by agents
    UPROPERTY(EditAnywhere)
    TArray<AGridObject*> Objects;

    // The columns in the environment
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
    TArray<AgentParameters> AgentParametersArray;

    // Morlet Wavelets simulator
    MorletWavelets2D* WaveSimulator;

    // Function to spawn the platform in the environment
    AStaticMeshActor* SpawnPlatform(FVector Location, FVector Size);

    // Function to "spawn" grid object with a delay
    void SetSpawnGridObject(int AgentIndex, float Delay, FVector Location);

    // Function to get the current state of an agent
    TArray<float> AgentGetState(int AgentIndex);

    // Helper function to convert a 2D grid point to a 1D index
    int Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const;

    // Helper function to check if the object has moved off the platform
    bool ObjectOffPlatform(int AgentIndex) const;

    // Function to map a value between two ranges
    float Map(float x, float in_min, float in_max, float out_min, float out_max) const;

    // Sets number of currently active GridObjects to random locations
    void SetActiveGridObjects(int NumAgents);

    // Helper function to convert grid position to world position
    FVector GridPositionToWorldPosition(FVector2D GridPosition);
};