#pragma once

#include "CoreMinimal.h"
#include "Templates/UniquePtr.h"
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GoalManager.h"
#include "TerraShift/Matrix2D.h"
#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "HeightMapGenerator.h"
#include "TerraShift/OccupancyGrid.h"
#include "Engine/World.h"
#include "Camera/CameraActor.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "StateManager.generated.h"
struct FHeightMapGPUAsyncState
{
    FHeightMapGPUDispatchHandle DispatchHandle;
    TArray<float> FrontState;
    TArray<float> BackState;
    bool bFrontValid = false;
    int32 StateW = 0;
    int32 StateH = 0;
    TArray<FVector3f> ColumnCentersScratch;
    TArray<FVector3f> ColumnRadiiScratch;
    TArray<FVector3f> ObjCentersScratch;
    TArray<FVector3f> ObjRadiiScratch;
    int32 BufferedNumObjects = 0;

    void EnsureDimensions(int32 InW, int32 InH)
    {
        if (StateW != InW || StateH != InH)
        {
            StateW = InW;
            StateH = InH;
            const int32 Count = StateW * StateH;
            FrontState.SetNumZeroed(Count);
            BackState.SetNumZeroed(Count);
            bFrontValid = false;
        }
        BufferedNumObjects = 0;
    }
};

/**
 * NEW: Enum to track the persistent state of each object slot throughout an episode.
 */
    UENUM(BlueprintType)
    enum class EObjectSlotState : uint8
{
    Empty,        // Slot is inactive, waiting for an object to be spawned
    Active,       // Object is currently active and in play
    GoalReached,  // Object successfully reached its goal this episode
    OutOfBounds   // Object fell out of bounds this episode
};

/**
 * UStateManager:
 *
 * - Tracks a certain number of "grid objects"
 * - Uses an OccupancyGrid to place random goals and objects without overlap
 * - Manages toggles for removing or keeping objects on goals/OOB, or "respawning" them
 * - Builds NxN height-based "central state" + optionally a goals-occupancy channel
 * - Colors columns for visualization
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UStateManager : public UObject
{
    GENERATED_BODY()

public:
    // ----------------------------------------------------------------
    //  Configuration & References
    // ----------------------------------------------------------------

    UFUNCTION(BlueprintCallable)
    void LoadConfig(UEnvironmentConfig* Config);

    UFUNCTION(BlueprintCallable)
    void SetReferences(
        AMainPlatform* InPlatform,
        AGridObjectManager* InObjMgr,
        AGrid* InGrid,
        UMultiAgentGaussianWaveHeightMap* InWaveSim,
        AGoalManager* InGoalManager
    );

    // ----------------------------------------------------------------
    //  Main Lifecycle
    // ----------------------------------------------------------------

    UFUNCTION(BlueprintCallable)
    void Reset(int32 NumObjects, int32 CurrentAgents);

    UFUNCTION(BlueprintCallable)
    void UpdateGridObjectFlags();

    UFUNCTION(BlueprintCallable)
    void UpdateObjectStats(float DeltaTime);

    UFUNCTION(BlueprintCallable)
    void RespawnGridObjects();

    UFUNCTION(BlueprintCallable)
    bool AllGridObjectsHandled() const;

    // ----------------------------------------------------------------
    //  Build Central State
    // ----------------------------------------------------------------

    UFUNCTION(BlueprintCallable)
    void BuildCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentState(int32 AgentIndex) const;

    UFUNCTION(BlueprintCallable)
    void UpdateGridColumnsColors();

    // Optimization: Toggle column collision based on nearby GridObjects
    UFUNCTION(BlueprintCallable)
    void UpdateColumnCollisionBasedOnOccupancy();

    // ----------------------------------------------------------------
    //  Accessors: data for environment's reward or logic
    // ----------------------------------------------------------------

    UFUNCTION(BlueprintCallable)
    int32 GetMaxGridObjects() const;

    // NEW Accessor for the persistent slot state
    UFUNCTION(BlueprintCallable)
    EObjectSlotState GetObjectSlotState(int32 ObjIndex) const;

    // Legacy accessors now derive from the new state enum
    UFUNCTION(BlueprintCallable)
    bool GetHasActive(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    bool GetHasReachedGoal(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    bool GetHasFallenOff(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    bool GetShouldCollectReward(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    void SetShouldCollectReward(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    bool GetShouldRespawn(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    int32 GetGoalIndex(int32 ObjIndex);

    UFUNCTION(BlueprintCallable)
    FVector GetCurrentVelocity(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    FVector GetPreviousVelocity(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    float GetCurrentDistance(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    float GetPreviousDistance(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    FVector GetCurrentPosition(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    FVector GetPreviousPosition(int32 ObjIndex) const;

    // Config accessors
    UFUNCTION(BlueprintCallable)
    bool GetRemoveObjectsOnGoal() const;

private:
    // ----------------------------------------------------------------
    //  Helpers
    // ----------------------------------------------------------------

    FVector GetColumnTopWorldLocation(int32 GridX, int32 GridY) const;
    void SetupOverheadCamera();
    TArray<float> CaptureOverheadImage() const;

private:
    // ----------------------------------------------------------------
    //  References
    // ----------------------------------------------------------------

    UPROPERTY()
    AMainPlatform* Platform = nullptr;

    UPROPERTY()
    AGridObjectManager* ObjectMgr = nullptr;

    UPROPERTY()
    AGrid* Grid = nullptr;

    UPROPERTY()
    UMultiAgentGaussianWaveHeightMap* WaveSim = nullptr;

    UPROPERTY()
    AGoalManager* GoalManager = nullptr;

    UPROPERTY()
    UOccupancyGrid* OccupancyGrid = nullptr;

    // ----------------------------------------------------------------
    //  Config & Toggles
    // ----------------------------------------------------------------
    // (All config properties remain the same)
    UPROPERTY() int32 MaxGridObjects = 8;
    UPROPERTY() float MarginXY = 1.5f;
    UPROPERTY() float MinZ = -4.f;
    UPROPERTY() float MaxZ = 100.f;
    UPROPERTY() int32 MarginCells = 4;
    // Deprecated: ObjectScale replaced by ObjectRadius-driven scaling
    // Removed from code paths to avoid confusion.
    UPROPERTY(Transient) float SpawnPaddingZ = 0.1f; // extra Z clearance on spawn
    UPROPERTY() float SpawnSeparationRadius = -1.f;   // if >0 overrides ObjectRadius for occupancy spacing
    UPROPERTY() float ObjectMass = 0.1f;
    UPROPERTY() float MaxColumnHeight = 4.f;
    UPROPERTY() float BaseRespawnDelay = 0.25f;
    UPROPERTY() bool bUseRandomGoals = true;
    UPROPERTY() bool bRespawnOnGoal;
    UPROPERTY() bool bRespawnOnOOB;
    UPROPERTY() bool bTerminateOnAllGoalsReached;
    UPROPERTY() bool bTerminateOnMaxSteps;
    UPROPERTY() bool bRemoveObjectsOnGoal = true;
    // If true, do not set bShouldCollect repeatedly each frame while an object remains at goal
    UPROPERTY() bool bSuppressPerStepGoalReward = false;
    UPROPERTY() float GoalRadius = 1.f;
    UPROPERTY() float GoalCollectRadius = 6.f;
    UPROPERTY() float ObjectRadius = 1.f;
    UPROPERTY() float ObjectUnscaledSize = 50.f;
    // Heightmap rendering biases (grid-local Z adjustments)
    UPROPERTY() float ColumnZBias = 0.f;
    UPROPERTY() float ObjectZBias = 0.f;
    UPROPERTY() TArray<FLinearColor> GoalColors;
    UPROPERTY() TArray<FLinearColor> GridObjectColors;
    UPROPERTY() bool bIncludeHeightMapInState;
    UPROPERTY() bool bIncludeOverheadImageInState;
    UPROPERTY() int32 StateHeightMapResolutionH;
    UPROPERTY() int32 StateHeightMapResolutionW;
    UPROPERTY() int32 StateOverheadImageResX;
    UPROPERTY() int32 StateOverheadImageResY;
    UPROPERTY() bool bIncludeGridObjectSequenceInState;
    UPROPERTY() int32 MaxGridObjectsForState;
    UPROPERTY() int32 GridObjectFeatureSize;

    // ------------------------------
    //  Optional Optimizations
    // ------------------------------
    // Enable/disable ROI-based collision toggling for columns
    UPROPERTY() bool bEnableColumnCollisionOptimization = false;
    // Radius in grid cells around each active GridObject to keep column collision enabled
    UPROPERTY() int32 ColumnCollisionRadiusCells = 2;
    // Hybrid height map is enabled when bEnableColumnCollisionOptimization is true.

    // ----------------------------------------------------------------
    //  Geometry
    // ----------------------------------------------------------------
    UPROPERTY() int32 GridSize = 50;
    UPROPERTY() float CellSize = 1.f;
    UPROPERTY() FVector PlatformWorldSize;
    UPROPERTY() FVector PlatformCenter;

    // ----------------------------------------------------------------
    //  NxN Height State
    // ----------------------------------------------------------------
    UPROPERTY() FMatrix2D PreviousHeight;
    UPROPERTY() FMatrix2D CurrentHeight;
    unsigned long Step = 0;
    TUniquePtr<FHeightMapGPUAsyncState> HeightMapGPUAsync;

    // Track previously enabled column cells to avoid redundant toggling
    UPROPERTY() TSet<int32> PrevEnabledColumnCells;

    // ----------------------------------------------------------------
    //  Object States
    // ----------------------------------------------------------------

    /** UPDATED: Tracks the persistent state of each object slot for the entire episode. */
    UPROPERTY()
    TArray<EObjectSlotState> ObjectSlotStates;

    // These flags remain for single-step logic
    UPROPERTY() TArray<bool> bShouldCollect;
    UPROPERTY() TArray<bool> bShouldResp;

    /** For each object => which "goal index" it is assigned to. */
    UPROPERTY() TArray<int32> ObjectGoalIndices;

    /** velocities, distances, etc. for each object */
    UPROPERTY() TArray<FVector> PrevVel;
    UPROPERTY() TArray<FVector> CurrVel;
    UPROPERTY() TArray<FVector> PrevAcc;
    UPROPERTY() TArray<FVector> CurrAcc;
    UPROPERTY() TArray<float>  PrevDist;
    UPROPERTY() TArray<float>  CurrDist;
    UPROPERTY() TArray<FVector> PrevPos;
    UPROPERTY() TArray<FVector> CurrPos;

    // Respawn timers
    UPROPERTY() TArray<float> RespawnTimer;
    UPROPERTY() TArray<float> RespawnDelays;

    // ----------------------------------------------------------------
    //  Overhead Camera
    // ----------------------------------------------------------------
    UPROPERTY() float OverheadCamDistance = 100.f;
    UPROPERTY() float OverheadCamFOV = 70.f;
    UPROPERTY() int32 OverheadCamResX = 50;
    UPROPERTY() int32 OverheadCamResY = 50;
    UPROPERTY() class ASceneCapture2D* OverheadCaptureActor = nullptr;
    UPROPERTY() class UTextureRenderTarget2D* OverheadRenderTarget = nullptr;
};
