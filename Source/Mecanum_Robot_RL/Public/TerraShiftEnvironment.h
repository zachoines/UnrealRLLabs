#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"

#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "TerraShift/StateManager.h"
#include "TerraShift/GoalManager.h"

#include "Materials/Material.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "Components/StaticMeshComponent.h"

#include "TerraShiftEnvironment.generated.h"

/**
 * Environment initialization parameters for TerraShift.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY();
};

/**
 * The TerraShift environment class.
 */
UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    /** Constructor & destructor */
    ATerraShiftEnvironment();
    virtual ~ATerraShiftEnvironment();

    // ---------------------------
    //   Overrides from ABaseEnvironment
    // ---------------------------
    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PostTransition() override;
    virtual void PreStep() override;
    virtual void PreTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;
    virtual void Tick(float DeltaTime) override;

    /** Root component for organizing everything. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    USceneComponent* TerraShiftRoot;

    /** The platform that agents operate on. */
    UPROPERTY(EditAnywhere)
    AMainPlatform* Platform;

    /** The grid structure. */
    UPROPERTY(EditAnywhere)
    AGrid* Grid;

    /** Manages the grid objects. */
    UPROPERTY(EditAnywhere)
    AGridObjectManager* GridObjectManager;

    /** The multi-agent Gaussian wave simulator (for RL wave-agents). */
    UPROPERTY()
    UMultiAgentGaussianWaveHeightMap* WaveSimulator;

    /** The new StateManager for grid-object states & the central-state matrix. */
    UPROPERTY()
    UStateManager* StateManager;

private:
    /** Our environment init params. */
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    /** Basic counters. */
    int CurrentStep;
    bool Initialized;

    /**
     * Current number of wave-sim RL “agents”
     */
    int CurrentAgents;

    /**
     * The number of grid objects actually managed by UStateManager.
     */
    int CurrentGridObjects;

    /** Folder path for environment organization. */
    FString EnvironmentFolderPath;

    // --------------  config-based environment settings --------------
    float PlatformSize;
    float MaxColumnHeight;
    FVector ObjectSize;
    float ObjectMass;
    int GridSize;
    int MaxSteps;
    int NumGoals;
    int MaxAgents;
    float GoalThreshold;

    // -------------- derived geometry --------------
    float CellSize;
    FVector PlatformWorldSize;
    FVector PlatformCenter;

    // -------------- references to new manager --------------
    UPROPERTY()
    AGoalManager* GoalManager;

    // -------------- environment-level reward toggles --------------
    bool bUseVelAlignment;
    bool bUseXYDistanceImprovement;
    bool bUseZAccelerationPenalty;
    bool bUseCradleReward;

    float VelAlign_Scale;
    float VelAlign_Min;
    float VelAlign_Max;

    float DistImprove_Scale;
    float DistImprove_Min;
    float DistImprove_Max;

    float ZAccel_Scale;
    float ZAccel_Min;
    float ZAccel_Max;

    float REACH_GOAL_REWARD;
    float FALL_OFF_PENALTY;
    float STEP_PENALTY;

private:
    /** Spawns the main platform. */
    AMainPlatform* SpawnPlatform(FVector Location);

    /** Legacy => unused now. */
    void UpdateGoal(int32 GoalIndex);

    FVector CalculateGoalPlatformLocation(int32 EdgeIndex);

    /** Helper => clamp values */
    float ThresholdAndClamp(float value, float minVal, float maxVal);
};