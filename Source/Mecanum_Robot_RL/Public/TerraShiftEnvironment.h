#pragma once

#include "CoreMinimal.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Materials/Material.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/StaticMesh.h"
#include "BaseEnvironment.h"
#include "Engine/StaticMeshActor.h"
#include "TimerManager.h"

#include "RLTypes.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/MorletWavelets2D.h"
#include "TerraShiftEnvironment.generated.h"

// Environment initialization parameters
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams {
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float PlatformSize = 1.0f; // meters

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float MaxColumnHeight = 1.0f; // meters, maximum height of the columns

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = FVector(0.10f, 0.10f, 0.10f); // meters, size of the grid objects

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = 0.2f; // kg

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int GridSize = 100; // Number of cells along one side of the grid

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxSteps = 1024; // Maximum steps per episode

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int NumGoals = 3; // Number of goals for agents

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float SpawnDelay = 1.0f; // Delay between each GridObject spawn in seconds

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxAgents = 10; // Maximum number of agents

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float EffectiveRadius = 1;

    // Agent wave parameter ranges
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D AmplitudeRange = FVector2D(0.5f, 1.0f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WaveOrientationRange = FVector2D(0.0f, 2 * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WavenumberRange = FVector2D(0.2f, 0.5f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D PhaseRange = FVector2D(0.0f, 2 * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D SigmaRange = FVector2D(0.5f, 5.0f);

    // Agent movement parameter ranges
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Movement Parameters")
    FVector2D VelocityRange = FVector2D(-0.3f, 0.3f); // Adjusted for meters per second
};

UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment {
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

    // The Grid
    UPROPERTY(EditAnywhere)
    AGrid* Grid;

    // Grid Object Manager
    UPROPERTY(EditAnywhere)
    AGridObjectManager* GridObjectManager;

    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PostTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;

    // Array of colors for the goals
    TArray<FLinearColor> GoalColors;

private:
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;
    int MaxAgents;
    int CurrentStep;
    int CurrentAgents;
    float CellSize;
    bool Intialized;
    TArray<int32> AgentGoalIndices;
    TArray<FVector> GoalPositionArray;
    TArray<FVector> GridCenterPoints;
    TArray<AgentParameters> AgentParametersArray;
    FVector PlatformWorldSize;
    FVector PlatformCenter;

    // Active columns currently having physics enabled
    TSet<int32> ActiveColumns;

    // Morlet Wavelets simulator
    MorletWavelets2D* WaveSimulator;

    // Initializes properties for action and observation space
    void SetupActionAndObservationSpace();

    // Function to spawn the platform in the environment
    AStaticMeshActor* SpawnPlatform(FVector Location);

    // Helper function to generate random positions on the grid for spawning particles
    FVector GenerateRandomGridLocation() const;

    // Helper function to generate random goal indices for edge-bound positions
    FIntPoint GenerateRandomGoalIndex() const;

    // Function to get the current state of an agent
    TArray<float> AgentGetState(int AgentIndex);

    // Determines if object has fallen off the platform
    bool ObjectOffPlatform(int AgentIndex);

    // Helper function to convert a 2D grid point to a 1D index
    int Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const;

    // Helper function to convert grid position to world position
    FVector GridPositionToWorldPosition(FVector2D GridPosition);

    // Sets the number of currently active GridObjects at random locations
    void SetActiveGridObjects(int NumAgents);

    // Helper function to map values between two ranges
    float Map(float x, float in_min, float in_max, float out_min, float out_max);

    // Updates the internally managed list of grid columns that have collisions enabled
    void UpdateActiveColumns();

    // Function to update column and GridObject colors
    void UpdateColumnGoalObjectColors();

    // Helper function to find the closest column index to a position
    int32 FindClosestColumnIndex(const FVector& Position, const TArray<FVector>& ColumnCenters) const;

    // Override the Tick function
    virtual void Tick(float DeltaTime) override;
};
