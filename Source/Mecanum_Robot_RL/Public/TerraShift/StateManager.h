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

#include "Engine/World.h"
#include "Camera/CameraActor.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Kismet/KismetRenderingLibrary.h"

#include "StateManager.generated.h"

/**
 * UStateManager:
 *
 *  - Tracks a certain number of “grid objects.”
 *  - Manages spawning either stationary or random goals (via AGoalManager).
 *  - Has toggles for removing or keeping objects on goal or OOB, or "respawning" them.
 *  - Colors columns & objects differently depending on user config.
 *  - Provides a function for "AllGridObjectsHandled" to see if environment is "done."
 *
 * The new version:
 *  - Reset(NumObjects, CurrentAgents) includes resetting the grid, object manager, wave sim
 *  - We store velocity/positions in PrevVel/CurrVel, etc.
 *  - Accessors for the environment to query velocity/distance, etc.
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UStateManager : public UObject
{
    GENERATED_BODY()

public:
    // ---------------------------------------------------------
    //  Setup / References
    // ---------------------------------------------------------

    /**
     * Loads the StateManager config data from JSON.
     */
    UFUNCTION(BlueprintCallable)
    void LoadConfig(UEnvironmentConfig* Config);

    /**
     * Set references for everything we need:
     *   - The scaled platform (for coordinate transforms)
     *   - The grid
     *   - The object manager
     *   - The wave sim
     *   - The GoalManager
     */
    UFUNCTION(BlueprintCallable)
    void SetReferences(
        AMainPlatform* InPlatform,
        AGridObjectManager* InObjMgr,
        AGrid* InGrid,
        UMultiAgentGaussianWaveHeightMap* InWaveSim,
        AGoalManager* InGoalManager
    );

    // ---------------------------------------------------------
    //  Reset
    // ---------------------------------------------------------

    /**
     * Called at environment reset.
     * 1) grid->ResetGrid()
     * 2) objectManager->ResetGridObjects()
     * 3) waveSim->Reset(CurrentAgents)
     * 4) Clears & initializes local arrays
     * 5) Possibly spawns stationary goals or picks random columns
     */
    UFUNCTION(BlueprintCallable)
    void Reset(int32 NumObjects, int32 CurrentAgents);

    // ---------------------------------------------------------
    //  Primary Loops
    // ---------------------------------------------------------

    /** Check if any object is OOB or reached goal => set flags. */
    UFUNCTION(BlueprintCallable)
    void UpdateGridObjectFlags();

    /** Update velocity, acceleration, distance, etc. for each active object. */
    UFUNCTION(BlueprintCallable)
    void UpdateObjectStats(float DeltaTime);

    /** Respawn objects flagged for "respawn" if their timers are done. */
    UFUNCTION(BlueprintCallable)
    void RespawnGridObjects();

    /**
     * Returns true if environment is "done" under toggles:
     *   - If (bRemoveGridObjectOnGoalReached || bRemoveGridObjectOnOOB) => done once no active/resp
     *   - Else if (bUseRandomGoals && !bRespawnGridObjectOnGoalReached && !bRemoveGridObjectOnOOB) => done if all reached
     *   - Otherwise => false
     */
    UFUNCTION(BlueprintCallable)
    bool AllGridObjectsHandled() const;

    // ---------------------------------------------------------
    //  Build NxN central state
    // ---------------------------------------------------------
    UFUNCTION(BlueprintCallable)
    void BuildCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetCentralState();

    /** If multi-agent wave sim, returns wave agent states. */
    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentState(int32 AgentIndex) const;

    // ---------------------------------------------------------
    //  Column Coloring
    // ---------------------------------------------------------
    /**
     * If bUseRandomGoals => color columns in radius with that goal’s color.
     * Otherwise => black-white by height
     */
    UFUNCTION(BlueprintCallable)
    void UpdateGridColumnsColors();

    // ---------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------
    UFUNCTION(BlueprintCallable)
    int32 GetMaxGridObjects() const;

    // For environment reward logic:
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

    /** For environment logic (i.e. step penalty if not active) */
    UFUNCTION(BlueprintCallable)
    bool GetShouldRespawn(int32 ObjIndex) const;

    /** Additional info for reward calculations. */
    UFUNCTION(BlueprintCallable)
    FVector GetCurrentVelocity(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    FVector GetPreviousVelocity(int32 ObjIndex) const;

    UFUNCTION(BlueprintCallable)
    float GetCurrentDistance(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    float GetPreviousDistance(int32 ObjIndex) const;

    /** Possibly used for debugging or indexing. */
    UFUNCTION(BlueprintCallable)
    int32 GetGoalIndex(int32 ObjIndex) const;

private:
    void SpawnStationaryGoalPlatforms();
    FVector GetColumnTopWorldLocation(int32 GridX, int32 GridY) const;
    FVector GenerateRandomGridLocation() const;

    /** Overhead camera usage. */
    void SetupOverheadCamera();
    TArray<float> CaptureOverheadImage() const;

private:
    // references
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

    // config-based
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
    float ColumnRadius = 1.f;

    UPROPERTY()
    TArray<FLinearColor> GoalColors;
    UPROPERTY()
    TArray<FLinearColor> GridObjectColors;

    // dimension
    UPROPERTY()
    int32 GridSize = 50;
    UPROPERTY()
    float CellSize = 1.f;
    UPROPERTY()
    FVector PlatformWorldSize;
    UPROPERTY()
    FVector PlatformCenter;

    // NxN for building central state
    UPROPERTY()
    FMatrix2D PreviousHeight;
    UPROPERTY()
    FMatrix2D CurrentHeight;
    unsigned long Step = 0;

    // object states
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

    UPROPERTY()
    TArray<int32> ObjectGoalIndices;

    // Velocity, distance, etc. for each object
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

    UPROPERTY()
    TArray<float> RespawnTimer;
    UPROPERTY()
    TArray<float> RespawnDelays;

    // overhead camera
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