// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.
// This version incorporates the "Fixed-Slot Reward Structure" parameters.

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
 * Environment initialization parameters specific to TerraShift.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY()
};

/**
 * The main environment class for the TerraShift simulation.
 */
UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    ATerraShiftEnvironment();
    virtual ~ATerraShiftEnvironment() override;

    // --- Overrides from ABaseEnvironment ---
    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PreTransition() override;
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;
    virtual void Tick(float DeltaTime) override;

    // --- Components and Core Actors ---
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    USceneComponent* TerraShiftRoot;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AMainPlatform* Platform;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGrid* Grid;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGridObjectManager* GridObjectManager;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGoalManager* GoalManager;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    UMultiAgentGaussianWaveHeightMap* WaveSimulator;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    UStateManager* StateManager;

private:
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;
    int CurrentStep;
    bool Initialized;
    int CurrentAgents;
    int CurrentGridObjects;
    FString EnvironmentFolderPath;

    // --- Configuration-Derived Settings ---
    float PlatformSize;
    float MaxColumnHeight;
    FVector ObjectSize;
    float ObjectMass;
    int GridSize;
    int MaxSteps;
    int MaxAgents;

    // --- Derived Geometry ---
    float CellSize;
    FVector PlatformWorldSize;
    FVector PlatformCenter;

    // --- Reward Configuration (Loaded in InitEnv) ---

    // Dense Shaping Rewards
    UPROPERTY() bool bUsePotentialShaping;
    UPROPERTY() float PotentialShaping_Scale;
    UPROPERTY() float PotentialShaping_Gamma;
    UPROPERTY() bool bUseVelAlignment;
    UPROPERTY() bool bUseXYDistanceImprovement;
    UPROPERTY() bool bUseZAccelerationPenalty;
    UPROPERTY() float VelAlign_Scale;
    UPROPERTY() float VelAlign_Min;
    UPROPERTY() float VelAlign_Max;
    UPROPERTY() float DistImprove_Scale;
    UPROPERTY() float DistImprove_Min;
    UPROPERTY() float DistImprove_Max;
    UPROPERTY() float ZAccel_Scale;
    UPROPERTY() float ZAccel_Min;
    UPROPERTY() float ZAccel_Max;

    // Stationary penalty (config-gated)
    UPROPERTY() bool bUseStationaryPenalty;
    UPROPERTY() float StationaryPenalty_MinSpeed;
    UPROPERTY() float StationaryPenalty_Drain;

    // Event-based rewards
    UPROPERTY()
    float EventReward_GoalReached;

    UPROPERTY()
    float EventReward_OutOfBounds;

    UPROPERTY()
    float TimeStepPenalty;

    // Termination toggles
    UPROPERTY()
    bool bTerminateOnAllGoalsReached;

    UPROPERTY()
    bool bTerminateOnMaxSteps;

    // Play mode toggles
    UPROPERTY()
    bool bUseDistanceBasedReward;

    UPROPERTY()
    bool bDisableEventRewards;

    // --- Runtime State for Potential Shaping ---
    UPROPERTY()
    TArray<float> PreviousPotential;

private:
    // --- Helper Functions ---
    AMainPlatform* SpawnPlatform(FVector Location);
    float ThresholdAndClamp(float value, float minThreshold, float maxClamp);
    float CalculatePotential(int32 ObjIndex) const;

    // --- Deprecated Overrides (kept for API compatibility) ---
    virtual void PostTransition() override {};
    virtual void PreStep() override {};
};
