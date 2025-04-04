#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"

#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "TerraShift/StateManager.h"
#include "TerraShift/GoalPlatform.h"

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
     * Current number of wave-sim RL “agents” (the user’s RLRunner calls ResetEnv(NumAgents)).
     * But this does NOT necessarily match the number of grid objects, see @CurrentGridObjects.
     */
    int CurrentAgents;

    /**
     * The number of grid objects actually managed by UStateManager.
     * Typically read from state-manager config: `max_grid_objects`
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

    // -------------- goal platforms --------------
    UPROPERTY()
    TArray<AGoalPlatform*> GoalPlatforms;

    UPROPERTY()
    TArray<FLinearColor> GoalColors;

    // -------------- active columns --------------
    TSet<int32> ActiveColumns;

private:
    /** Spawns the main platform. */
    AMainPlatform* SpawnPlatform(FVector Location);

    /** Creates a goal platform for each of the 4 edges **/
    void UpdateGoal(int32 GoalIndex);

    /** Locates the platform for a given edge (0=Top,1=Bottom,2=Left,3=Right). */
    FVector CalculateGoalPlatformLocation(int32 EdgeIndex);

    /** Refresh columns' physics based on proximity. */
    void UpdateActiveColumns();

    /** Color columns based on height, etc. */
    void UpdateColumnColors();

    /** Helper for Reward() => thresholding. */
    float ThresholdAndClamp(float value, float minVal, float maxVal);

    // Reward logic toggles
    static constexpr bool bUseVelAlignment = false;
    static constexpr bool bUseXYDistanceImprovement = true;
    static constexpr bool bUseZAccelerationPenalty = false;
    static constexpr bool bUseCradleReward = false;

    static constexpr float VelAlign_Scale = 0.1f;
    static constexpr float VelAlign_Min = -100.f;
    static constexpr float VelAlign_Max = 100.f;

    static constexpr float DistImprove_Scale = 10.f;
    static constexpr float DistImprove_Min = -1.f;
    static constexpr float DistImprove_Max = 1.f;

    static constexpr float ZAccel_Scale = 0.0001f;
    static constexpr float ZAccel_Min = 0.1f;
    static constexpr float ZAccel_Max = 2000.f;

    static constexpr float REACH_GOAL_REWARD = 1.0f;
    static constexpr float FALL_OFF_PENALTY = -1.0f;
    static constexpr float STEP_PENALTY = -0.0001f;
};
