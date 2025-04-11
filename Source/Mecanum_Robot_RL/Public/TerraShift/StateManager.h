#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"

#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GoalManager.h"
#include "TerraShift/Matrix2D.h"
#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "TerraShift/OccupancyGrid.h"

#include "Engine/World.h"
#include "Camera/CameraActor.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Kismet/KismetRenderingLibrary.h"

#include "StateManager.generated.h"

/**
 * UStateManager:
 *
 *  - Tracks a certain number of “grid objects”
 *  - Uses an OccupancyGrid to place random goals and objects without overlap
 *  - Manages toggles for removing or keeping objects on goals/OOB, or "respawning" them
 *  - Builds NxN height-based "central state" + optionally a goals-occupancy channel
 *  - Colors columns for visualization
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UStateManager : public UObject
{
    GENERATED_BODY()

public:
    // ----------------------------------------------------------------
    //  Configuration & References
    // ----------------------------------------------------------------

    /**
     * Loads config data for the StateManager (like toggles, radii, colors).
     */
    UFUNCTION(BlueprintCallable)
    void LoadConfig(UEnvironmentConfig* Config);

    /**
     * Sets all references (Platform, Grid, ObjectMgr, WaveSim, GoalManager).
     * Also initializes OccupancyGrid from the current grid dimension.
     */
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

    /**
     * Called at environment reset. Resets grid, objects, wave, and occupancy.
     * Spawns random or stationary goals. Prepares arrays for N objects.
     */
    UFUNCTION(BlueprintCallable)
    void Reset(int32 NumObjects, int32 CurrentAgents);

    /**
     * Check if any object is out of bounds or reached goal => sets flags.
     * If we remove an object from the scene, also remove it from OccupancyGrid.
     */
    UFUNCTION(BlueprintCallable)
    void UpdateGridObjectFlags();

    /**
     * Updates velocity, acceleration, distance for active objects.
     * Then updates OccupancyGrid->UpdateObjectPosition(...) so occupant location is tracked.
     */
    UFUNCTION(BlueprintCallable)
    void UpdateObjectStats(float DeltaTime);

    /**
     * For objects flagged to respawn => remove old occupant => add new occupant in free cell.
     * Then spawns the actual AGridObject at that location.
     */
    UFUNCTION(BlueprintCallable)
    void RespawnGridObjects();

    /**
     * Returns true if the environment is "done" under the toggles.
     * E.g. if remove-on-goal or remove-on-oob => done once no active or respawning objects remain.
     */
    UFUNCTION(BlueprintCallable)
    bool AllGridObjectsHandled() const;

    // ----------------------------------------------------------------
    //  Build Central State
    // ----------------------------------------------------------------

    /**
     * For each NxN cell, do a line trace to find the column top => store in CurrentHeight.
     * Also captures overhead camera. Step++.
     */
    UFUNCTION(BlueprintCallable)
    void BuildCentralState();

    /**
     * Gathers the NxN height (and delta-height) plus overhead camera data,
     * plus an optional "Goals" occupancy channel, into a single float array.
     */
    UFUNCTION(BlueprintCallable)
    TArray<float> GetCentralState();

    /**
     * For multi-agent wave sim => returns wave agent states (unchanged).
     */
    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentState(int32 AgentIndex) const;

    /**
     * Colors each column by height, then occupant-based color for "GridObjects",
     * then occupant-based color for "Goals".
     */
    UFUNCTION(BlueprintCallable)
    void UpdateGridColumnsColors();

    // ----------------------------------------------------------------
    //  Accessors: data for environment's reward or logic
    // ----------------------------------------------------------------

    UFUNCTION(BlueprintCallable)
    int32 GetMaxGridObjects() const;

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
    int32 GetGoalIndex(int32 ObjIndex) const;

    // For distance-based or velocity-based rewards
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

private:
    // ----------------------------------------------------------------
    //  Helpers
    // ----------------------------------------------------------------

    /** Spawns stationary AGoalPlatforms at edges. If you want them in Occupancy, add them after. */
    void SpawnStationaryGoalPlatforms();

    /**
     *  Top surface of a column (X,Y) => world location.
     */
    FVector GetColumnTopWorldLocation(int32 GridX, int32 GridY) const;

    /**
     * Sets up the overhead capture camera (unrelated to occupancy).
     */
    void SetupOverheadCamera();

    /**
     * Reads the overhead camera's texture => returns flattened R,G,B channels.
     */
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

    /** The new class that unifies random placement & layering of occupant objects (goals, etc). */
    UPROPERTY()
    UOccupancyGrid* OccupancyGrid = nullptr;

    // ----------------------------------------------------------------
    //  Config & Toggles
    // ----------------------------------------------------------------

    UPROPERTY()
    int32 MaxGridObjects = 8;

    UPROPERTY()
    float MarginXY = 1.5f;

    UPROPERTY()
    float MinZ = -4.f;

    UPROPERTY()
    float MaxZ = 100.f;

    UPROPERTY()
    int32 MarginCells = 4;

    UPROPERTY()
    float ObjectScale = 0.1f;

    UPROPERTY()
    float ObjectMass = 0.1f;

    UPROPERTY()
    float MaxColumnHeight = 4.f;

    UPROPERTY()
    float BaseRespawnDelay = 0.25f;

    // toggles
    UPROPERTY()
    bool bUseRandomGoals = true;

    UPROPERTY()
    bool bRemoveGridObjectOnGoalReached = false;

    UPROPERTY()
    bool bRemoveGridObjectOnOOB = false;

    UPROPERTY()
    bool bRespawnGridObjectOnGoalReached = false;

    UPROPERTY()
    float GoalRadius = 1.f;

    UPROPERTY()
    float ObjectRadius = 1.f;

    UPROPERTY()
    float ObjectUnscaledSize = 50.f;

    UPROPERTY()
    TArray<FLinearColor> GoalColors;

    UPROPERTY()
    TArray<FLinearColor> GridObjectColors;

    // ----------------------------------------------------------------
    //  Geometry
    // ----------------------------------------------------------------

    UPROPERTY()
    int32 GridSize = 50;

    UPROPERTY()
    float CellSize = 1.f;

    UPROPERTY()
    FVector PlatformWorldSize;

    UPROPERTY()
    FVector PlatformCenter;

    // ----------------------------------------------------------------
    //  NxN Height State
    // ----------------------------------------------------------------

    UPROPERTY()
    FMatrix2D PreviousHeight;

    UPROPERTY()
    FMatrix2D CurrentHeight;

    unsigned long Step = 0;

    // ----------------------------------------------------------------
    //  Object States
    // ----------------------------------------------------------------

    UPROPERTY()
    TArray<bool> bHasActive;

    UPROPERTY()
    TArray<bool> bHasReached;

    UPROPERTY()
    TArray<bool> bFallenOff;

    UPROPERTY()
    TArray<bool> bShouldCollect;

    UPROPERTY()
    TArray<bool> bShouldResp;

    /** For each object => which "goal index" it is assigned to. */
    UPROPERTY()
    TArray<int32> ObjectGoalIndices;

    /** velocities, distances, etc. for each object */
    UPROPERTY()
    TArray<FVector> PrevVel;

    UPROPERTY()
    TArray<FVector> CurrVel;

    UPROPERTY()
    TArray<FVector> PrevAcc;

    UPROPERTY()
    TArray<FVector> CurrAcc;

    UPROPERTY()
    TArray<float>  PrevDist;

    UPROPERTY()
    TArray<float>  CurrDist;

    UPROPERTY()
    TArray<FVector> PrevPos;

    UPROPERTY()
    TArray<FVector> CurrPos;

    // Respawn timers
    UPROPERTY()
    TArray<float> RespawnTimer;

    UPROPERTY()
    TArray<float> RespawnDelays;

    // ----------------------------------------------------------------
    //  Overhead Camera
    // ----------------------------------------------------------------

    UPROPERTY()
    float OverheadCamDistance = 100.f;

    UPROPERTY()
    float OverheadCamFOV = 70.f;

    UPROPERTY()
    int32 OverheadCamResX = 50;

    UPROPERTY()
    int32 OverheadCamResY = 50;

    UPROPERTY()
    class ASceneCapture2D* OverheadCaptureActor = nullptr;

    UPROPERTY()
    class UTextureRenderTarget2D* OverheadRenderTarget = nullptr;
};
