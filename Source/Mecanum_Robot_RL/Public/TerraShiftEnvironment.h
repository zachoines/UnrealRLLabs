#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"

#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/MultiAgentFractalWave3D.h"
#include "TerraShift/GoalPlatform.h"

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
 * The TerraShift environment class managing the simulation.
 */
UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment
{
    GENERATED_BODY()

public:
    /** Constructor */
    ATerraShiftEnvironment();

    /** Destructor */
    virtual ~ATerraShiftEnvironment();

    // Overrides from ABaseEnvironment

    /** Initializes the environment with the given parameters. */
    virtual void InitEnv(FBaseInitParams* Params) override;

    /** Resets the environment and returns the initial state. */
    virtual FState ResetEnv(int NumAgents) override;

    /** Applies the given action to the environment. */
    virtual void Act(FAction Action) override;

    /** Called after each transition to update the environment. */
    virtual void PostTransition() override;

    /** Called before each step to update internal counters or state. */
    virtual void PreStep() override;

    /** Called before each transition to update the environment. */
    virtual void PreTransition() override;

    /** Called after each step to update internal counters or state. */
    virtual void PostStep() override;

    /** Returns the current state of the environment. */
    virtual FState State() override;

    /** Checks if the episode is done. */
    virtual bool Done() override;

    /** Checks if the episode should be truncated. */
    virtual bool Trunc() override;

    /** Returns the cumulative reward since the last call. */
    virtual float Reward() override;

    /** Tick function called every frame. */
    virtual void Tick(float DeltaTime) override;

    /** The root component for organizing everything in this environment. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TerraShift")
    USceneComponent* TerraShiftRoot;

    /** The platform that agents operate on. */
    UPROPERTY(EditAnywhere)
    AMainPlatform* Platform;

    /** The grid structure in the environment. */
    UPROPERTY(EditAnywhere)
    AGrid* Grid;

    /** Manages the grid objects in the environment. */
    UPROPERTY(EditAnywhere)
    AGridObjectManager* GridObjectManager;

    /** Manages the generate of waves for grid. */
    UPROPERTY()
    UMultiAgentFractalWave3D* WaveSimulator;

private:
    /** Initialization parameters specific to TerraShift. */
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    /** Current simulation step. */
    int CurrentStep;

    /** Number of active agents. */
    int CurrentAgents;

    /** Size of a single cell in the grid. */
    float CellSize;

    /** Flag to check if the environment is initialized. */
    bool Initialized;

    /** Size of the platform in meters. */
    float PlatformSize;

    /** Maximum height of the columns in meters. */
    float MaxColumnHeight;

    /** Size of the grid objects in meters. */
    FVector ObjectSize;

    /** Mass of the grid objects in kilograms. */
    float ObjectMass;

    /** Number of cells along one side of the grid. */
    int GridSize;

    /** Maximum steps per episode. */
    int MaxSteps;

    /** Number of goals for agents (set between 1 - 4). */
    int NumGoals;

    /** Delay between each GridObject spawn in seconds. */
    float SpawnDelay;

    /** Maximum number of agents. */
    int MaxAgents;

    /** Threshold distance to consider a goal reached. */
    float GoalThreshold;

    /** Indices of goals assigned to agents. */
    TArray<int32> AgentGoalIndices;

    /** Center points of grid cells. */
    TArray<FVector> GridCenterPoints;

    /** Size of the platform in world units. */
    FVector PlatformWorldSize;

    /** Center position of the platform. */
    FVector PlatformCenter;

    /** Folder path for organizing actors in the World Outliner. */
    FString EnvironmentFolderPath;

    /** Set of active columns with physics enabled. */
    TSet<int32> ActiveColumns;

    /** Flags to track if agents have active grid objects. */
    TArray<bool> AgentHasActiveGridObject;

    /** Flags to track if grid objects have reached their goals. */
    TArray<bool> GridObjectHasReachedGoal;

    /** Flags to track if grid objects have fallen off grid. */
    TArray<bool> GridObjectFallenOffGrid;

    /** Flags to track rewards granted for reaching goal and falling of grid events. */
    TArray<bool> GridObjectShouldCollectEventReward;

    /** Flags to track if grid objects should be respawned. */
    TArray<bool> GridObjectShouldRespawn;

    /** Tracks elapsed time til respawn. */
    TArray<float> GridObjectRespawnTimer;

    /** Time delays for respawn */
    TArray<float> GridObjectRespawnDelays;

    /** Array of goal platforms. */
    TArray<AGoalPlatform*> GoalPlatforms;

    /** Colors used for goals. */
    TArray<FLinearColor> GoalColors;

    /** Stores previous velocities for each agent's GridObject for acceleration computation. */
    TArray<FVector> PreviousObjectVelocities;

    /** Stores previous accelerations for each agent's GridObject for acceleration computation. */
    TArray<FVector> PreviousObjectAcceleration;

    /** Stores the previous distances of each agent's GridObject to its assigned goal. */
    TArray<float> PreviousDistances;

    /** Stores the previous positions of each agent's GridObject to its assigned goal. */
    TArray<FVector> PreviousPositions;

    /** Stores current velocities for each agent's GridObject for acceleration computation. */
    TArray<FVector> CurrentObjectVelocities;

    /** Stores current accelerations for each agent's GridObject for acceleration computation. */
    TArray<FVector> CurrentObjectAcceleration;

    /** Stores the previous distances of each agent's GridObject to its assigned goal. */
    TArray<float> CurrentDistances;

    /** Stores the current positions of each agent's GridObject to its assigned goal. */
    TArray<FVector> CurrentPositions;

    /** Stores the last DeltaTime from the simulation loop. */
    float LastDeltaTime;

    /**
     * Spawns the main platform in the environment.
     * @param Location The location to spawn the platform.
     * @return A pointer to the spawned platform.
     */
    AMainPlatform* SpawnPlatform(FVector Location);

    /**
     * Generates a random world position on the grid for spawning grid objects.
     * @return A random world position on the grid.
     */
    FVector GenerateRandomGridLocation() const;


    FMatrix2D ComputeCollisionDistanceMatrix() const;

    /**
     * Retrieves the current state of a specific agent.
     * @param AgentIndex Index of the agent.
     * @return An array representing the agent's state.
     */
    TArray<float> AgentGetState(int AgentIndex);

    /**
     * Maps a value from one range to another.
     * @param x The value to map.
     * @param in_min Input range minimum.
     * @param in_max Input range maximum.
     * @param out_min Output range minimum.
     * @param out_max Output range maximum.
     * @return The mapped value.
     */
    float Map(float x, float in_min, float in_max, float out_min, float out_max);

    /** Updates the list of active grid columns based on proximity to grid objects. */
    void UpdateActiveColumns();

    /** Updates the colors of grid columns and grid objects. */
    void UpdateColumnGoalObjectColors();

    /** Checks for grid objects that need to be respawned. */
    void UpdateGridObjectFlags();

    /** Update information like relative acceleration */
    void UpdateObjectStats();

    /** Respawns Grid Objects based on respawn flag*/
    void RespawnGridObjects();

    /**
     * Calculates the location of a goal platform based on the edge index.
     * @param EdgeIndex Index representing the edge (0 - Top, 1 - Bottom, 2 - Left, 3 - Right).
     * @return The calculated location for the goal platform.
     */
    FVector CalculateGoalPlatformLocation(int32 EdgeIndex);

    /**
     * Creates or updates a goal platform.
     * @param GoalIndex Index of the goal to update.
     */
    void UpdateGoal(int32 GoalIndex);

    /**
     * Converts a grid position to a world position.
     * @param GridPosition The grid position to convert.
     * @return The corresponding world position.
     */
    FVector GridPositionToWorldPosition(FVector2D GridPosition) const;

    /**
     * Helper for Reward() function. That thresholds input to 0 for anyhing below min value. 
     * Clamp to MaxValue
     * @param value Value to threshold and clamp
     * @param minValue Min input value before input is thresholded
     * @param maxValue Max output to clamp to
     * @return Value clamped and thresholded.
     */
    float ThresholdAndClamp(float value, float minVal, float maxVal);

    // Toggles to enable/disable sub-rewards
    static constexpr bool bUseVelAlignment = true;
    static constexpr bool bUseXYDistanceImprovement = true;
    static constexpr bool bUseZAccelerationPenalty = false;
    static constexpr bool bUseCradleReward = false;

    // Velocity-to-goal constants
    static constexpr float VelAlign_Scale = 0.1f;   // main multiplie
    static constexpr float VelAlign_Min = -100.0f;   // clamp “away” speed
    static constexpr float VelAlign_Max = 100.0f;   // clamp “toward” speed

    // Distance-improvement constants (XY only)
    static constexpr float DistImprove_Scale = 10.0; // main multiplier
    static constexpr float DistImprove_Min = -1.0f;   // clamp negative delta
    static constexpr float DistImprove_Max = 1.0f;   // clamp positive delta

    // Z-acceleration penalty (gravity is 980cm.t^2 )
    static constexpr float ZAccel_Scale = 0.0001f;
    static constexpr float ZAccel_Min = .1;
    static constexpr float ZAccel_Max = 2000.0f;

    // Event-based
    static constexpr float REACH_GOAL_REWARD = 1.0f;
    static constexpr float FALL_OFF_PENALTY = -1.0f;
    static constexpr float STEP_PENALTY = -0.0001f;

};
