// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

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
 * Currently inherits from Base, can add TerraShift-specific params here if needed.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams
{
    GENERATED_USTRUCT_BODY()
    // Add any TerraShift-specific initialization parameters here if needed
};

/**
 * The main environment class for the TerraShift simulation.
 * Manages the platform, grid, objects, agents (via WaveSimulator), state, and rewards.
 */
UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    /** Constructor & destructor */
    ATerraShiftEnvironment();
    virtual ~ATerraShiftEnvironment() override; // Added override specifier

    // --- Overrides from ABaseEnvironment ---
    virtual void InitEnv(FBaseInitParams* Params) override;
    virtual FState ResetEnv(int NumAgents) override;
    virtual void Act(FAction Action) override;
    virtual void PreTransition() override; // Step timing: PreTransition -> PhysicsSim/Act -> Reward -> State -> PostStep -> Done/Trunc
    virtual void PostStep() override;
    virtual FState State() override;
    virtual bool Done() override;
    virtual bool Trunc() override;
    virtual float Reward() override;
    virtual void Tick(float DeltaTime) override;

    // --- Components and Core Actors ---

    /** Root component for organizing all environment actors. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    USceneComponent* TerraShiftRoot;

    /** The main platform actor where objects and goals reside. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AMainPlatform* Platform;

    /** The deformable grid actor composed of columns. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGrid* Grid;

    /** Manages spawning, tracking, and physics of the grid objects. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGridObjectManager* GridObjectManager;

    /** Manages the goal locations and checks if objects reach them. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Actors")
    AGoalManager* GoalManager;

    /** Component simulating wave propagation based on agent actions. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    UMultiAgentGaussianWaveHeightMap* WaveSimulator;

    /** Component responsible for state calculation, object tracking, and episode logic. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift|Components")
    UStateManager* StateManager;


private:
    /** Stored initialization parameters. */
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    /** Tracks the current simulation step within an episode. */
    int CurrentStep;

    /** Flag indicating if InitEnv has successfully completed. */
    bool Initialized;

    /** Current number of active RL agents for this episode. */
    int CurrentAgents;

    /** Current number of grid objects being managed in this episode. */
    int CurrentGridObjects;

    /** Folder path used for potential logging/debugging output. */
    FString EnvironmentFolderPath;

    // --- Configuration-Derived Settings (Loaded in InitEnv) ---
    float PlatformSize;
    float MaxColumnHeight;
    FVector ObjectSize;
    float ObjectMass;
    int GridSize;
    int MaxSteps; // Maximum steps per episode before truncation
    int MaxAgents; // Maximum possible agents environment supports

    // --- Derived Geometry (Calculated in InitEnv) ---
    float CellSize;
    FVector PlatformWorldSize;
    FVector PlatformCenter;

    // --- Reward Configuration (Loaded in InitEnv) ---

    // Potential-Based Shaping
    UPROPERTY()
    bool bUsePotentialShaping; // If true, adds potential-based reward shaping term

    UPROPERTY()
    float PotentialShaping_Scale; // Scaling factor for the potential reward

    UPROPERTY()
    float PotentialShaping_Gamma; // Agent's discount factor (gamma), needed for strict potential shaping

    // Other Shaping Rewards
    UPROPERTY()
    bool bUseVelAlignment;
    UPROPERTY()
    bool bUseXYDistanceImprovement;
    UPROPERTY()
    bool bUseZAccelerationPenalty;
    UPROPERTY()
    bool bUseAlignedDistanceShaping;

    // Scales and Limits for Shaping Rewards
    UPROPERTY()
    float VelAlign_Scale;
    UPROPERTY()
    float VelAlign_Min; // Clamping values (if needed, currently unused in cpp)
    UPROPERTY()
    float VelAlign_Max;

    UPROPERTY()
    float DistImprove_Scale;
    UPROPERTY()
    float DistImprove_Min; // Clamping values for delta distance
    UPROPERTY()
    float DistImprove_Max;

    UPROPERTY()
    float ZAccel_Scale;
    UPROPERTY()
    float ZAccel_Min; // Threshold for Z acceleration penalty
    UPROPERTY()
    float ZAccel_Max; // Clamping value for Z acceleration penalty

    // Terminal/Step Rewards
    UPROPERTY()
    float REACH_GOAL_REWARD;
    UPROPERTY()
    float FALL_OFF_PENALTY;
    UPROPERTY()
    float STEP_PENALTY;

    // --- Runtime State for Potential Shaping ---

    /** Stores the potential value calculated in the *previous* step for each object. Used for potential shaping calculation. */
    UPROPERTY()
    TArray<float> PreviousPotential;

private:
    // --- Helper Functions ---

    /** Spawns and initializes the main platform actor during InitEnv. */
    AMainPlatform* SpawnPlatform(FVector Location);

    /** Helper function to threshold and clamp a float value. Used in Z-acceleration penalty. */
    float ThresholdAndClamp(float value, float minThreshold, float maxClamp);

    /** Calculates the potential function Phi(s) for a given object's state. Used for potential shaping. */
    float CalculatePotential(int32 ObjIndex) const; // Added const as it doesn't modify member vars

    // --- Deprecated/Unused Overrides (Kept for compatibility/completeness if needed) ---
    // These are optional lifecycle hooks from BaseEnvironment that might not be needed for TerraShift's logic flow.
    // If not used, their implementations in the .cpp can be empty.
    virtual void PostTransition() override {}; // Example: Empty implementation if not used
    virtual void PreStep() override {}; // Example: Empty implementation if not used

};