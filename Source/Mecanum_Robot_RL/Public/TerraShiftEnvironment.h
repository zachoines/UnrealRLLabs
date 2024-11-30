#pragma once

#include "CoreMinimal.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"

#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/MorletWavelets2D.h"
#include "TerraShift/GoalPlatform.h"

#include "TerraShiftEnvironment.generated.h"

/**
 * Environment initialization parameters for TerraShift.
 */
USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTerraShiftEnvironmentInitParams : public FBaseInitParams {
    GENERATED_USTRUCT_BODY();

    /** Size of the platform in meters. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float PlatformSize = 1.0f;

    /** Maximum height of the columns in meters. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float MaxColumnHeight = 1.0f;

    /** Size of the grid objects in meters. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector ObjectSize = FVector(0.10f, 0.10f, 0.10f);

    /** Mass of the grid objects in kilograms. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float ObjectMass = 0.2f;

    /** Number of cells along one side of the grid. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int GridSize = 50;

    /** Maximum steps per episode. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxSteps = 1024;

    /** Number of goals for agents (set between 1 - 4). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int NumGoals = 4;

    /** Delay between each GridObject spawn in seconds. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float SpawnDelay = 1.0f;

    /** Delay before respawning a GridObject. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float RespawnDelay = 1.0f;

    /** Maximum number of agents. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxAgents = 10;

    /** Threshold distance to consider a goal reached. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    float GoalThreshold = 0.1f;

    // Agent wave parameter ranges

    /** Amplitude range (0 allows for null waves). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D AmplitudeRange = FVector2D(0.0f, 10.0f);

    /** Orientation angle in radians. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WaveOrientationRange = FVector2D(0.0f, 2 * PI);

    /** Wavenumber (0 allows for no oscillations). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D WavenumberRange = FVector2D(0.0f, 1.5f);

    /** Initial phase offset in radians. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D PhaseRange = FVector2D(0.0f, 2 * PI);

    /** Spread of Gaussian envelope; small values for high localization. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D SigmaRange = FVector2D(0.01f, 15.0f);

    /** Phase velocity for each agent, affecting frequency. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Wave Parameters")
    FVector2D PhaseVelocityRange = FVector2D(0.0f, 5.0f);

    // Agent movement parameters

    /** Velocity in meters per second for X and Y directions. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent Movement Parameters")
    FVector2D VelocityRange = FVector2D(-10.0f, 10.0f);
};

/**
 * The TerraShift environment class managing the simulation.
 */
UCLASS()
class UNREALRLLABS_API ATerraShiftEnvironment : public ABaseEnvironment {
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

private:
    /** Initialization parameters specific to TerraShift. */
    FTerraShiftEnvironmentInitParams* TerraShiftParams = nullptr;

    /** Maximum number of agents allowed. */
    int MaxAgents;

    /** Current simulation step. */
    int CurrentStep;

    /** Number of active agents. */
    int CurrentAgents;

    /** Size of a single cell in the grid. */
    float CellSize;

    /** Flag to check if the environment is initialized. */
    bool Initialized;

    /** Indices of goals assigned to agents. */
    TArray<int32> AgentGoalIndices;

    /** Center points of grid cells. */
    TArray<FVector> GridCenterPoints;

    /** Parameters for each agent's wave. */
    TArray<AgentParameters> AgentParametersArray;

    /** Size of the platform in world units. */
    FVector PlatformWorldSize;

    /** Center position of the platform. */
    FVector PlatformCenter;

    /** Folder path for organizing actors in the World Outliner. */
    FString EnvironmentFolderPath;

    /** Set of active columns with physics enabled. */
    TSet<int32> ActiveColumns;

    /** Simulator for Morlet wavelets. */
    MorletWavelets2D* WaveSimulator;

    /** Flags to track if agents have active grid objects. */
    TArray<bool> AgentHasActiveGridObject;

    /** Flags to track if grid objects have reached their goals. */
    TArray<bool> GridObjectHasReachedGoal;

    /** Buffer to accumulate rewards. */
    float RewardBuffer;

    /** Array of goal platforms. */
    TArray<AGoalPlatform*> GoalPlatforms;

    /** Locations of goal platforms relative to the platform. */
    TArray<FVector> GoalPlatformLocations;

    /** Colors used for goals. */
    TArray<FLinearColor> GoalColors;

    /** Initializes properties for action and observation space. */
    void SetupActionAndObservationSpace();

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

    /**
     * Retrieves the current state of a specific agent.
     * @param AgentIndex Index of the agent.
     * @return An array representing the agent's state.
     */
    TArray<float> AgentGetState(int AgentIndex);

    /**
     * Activates grid objects for the specified number of agents.
     * @param NumAgents Number of agents to activate grid objects for.
     */
    void SetActiveGridObjects(int NumAgents);

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
    void CheckAndRespawnGridObjects();

    /**
     * Respawns a grid object for a specific agent.
     * @param AgentIndex Index of the agent whose grid object needs to be respawned.
     */
    void RespawnGridObject(int32 AgentIndex);

    /**
     * Called when a grid object is spawned.
     * @param Index Index of the grid object.
     * @param NewGridObject Pointer to the newly spawned grid object.
     */
    UFUNCTION()
    void OnGridObjectSpawned(int32 Index, AGridObject* NewGridObject);

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
};