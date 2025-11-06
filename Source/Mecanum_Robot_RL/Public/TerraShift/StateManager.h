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

/** Tracks the persistent state of each object slot throughout an episode. */
UENUM(BlueprintType)
enum class EObjectSlotState : uint8
{
    Empty,        // Slot is inactive and waiting for spawn.
    Active,       // Object is currently active and in play.
    GoalReached,  // Object successfully reached its goal this episode.
    OutOfBounds   // Object fell out of bounds this episode.
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
    // Configuration and references.
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

    // Main lifecycle.
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

    // Central state generation.
    UFUNCTION(BlueprintCallable)
    void BuildCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentState(int32 AgentIndex) const;

    UFUNCTION(BlueprintCallable)
    void UpdateGridColumnsColors();

    // Accessors.

    UFUNCTION(BlueprintCallable)
    int32 GetMaxGridObjects() const;

    // Accessor for the persistent slot state.
    UFUNCTION(BlueprintCallable)
    EObjectSlotState GetObjectSlotState(int32 ObjIndex) const;

    // Legacy-style accessors derived from the slot state.
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
    // Helpers.
    FVector GetColumnTopWorldLocation(int32 GridX, int32 GridY) const;
    void SetupOverheadCamera();
    TArray<float> CaptureOverheadImage() const;

private:
    // References.
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

    // Configuration and toggles.
    // (All config properties remain the same)
    UPROPERTY() int32 MaxGridObjects = 8;
    UPROPERTY() float MarginXY = 1.5f;
    UPROPERTY() float MinZ = -4.f;
    UPROPERTY() float MaxZ = 100.f;
    UPROPERTY() int32 MarginCells = 4;
    UPROPERTY() float ObjectScale = 0.1f;
    UPROPERTY() float ObjectMass = 0.1f;
    UPROPERTY() float MaxColumnHeight = 4.f;
    UPROPERTY() float BaseRespawnDelay = 0.25f;
    UPROPERTY() bool bUseRandomGoals = true;
    UPROPERTY() bool bRespawnOnGoal;
    UPROPERTY() bool bRespawnOnOOB;
    UPROPERTY() bool bTerminateOnAllGoalsReached;
    UPROPERTY() bool bTerminateOnMaxSteps;
    UPROPERTY() bool bRemoveObjectsOnGoal;
    UPROPERTY() float GoalRadius = 1.f;
    UPROPERTY() float GoalCollectRadius = 6.f;
    UPROPERTY() float ObjectRadius = 1.f;
    UPROPERTY() float ObjectUnscaledSize = 50.f;
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

    // Geometry.
    UPROPERTY() int32 GridSize = 50;
    UPROPERTY() float CellSize = 1.f;
    UPROPERTY() FVector PlatformWorldSize;
    UPROPERTY() FVector PlatformCenter;

    // NxN height state.
    UPROPERTY() FMatrix2D PreviousHeight;
    UPROPERTY() FMatrix2D CurrentHeight;
    unsigned long Step = 0;

    // Object states.

    /** Tracks the persistent state of each object slot for the entire episode. */
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

    // Overhead camera.
    UPROPERTY() float OverheadCamDistance = 100.f;
    UPROPERTY() float OverheadCamFOV = 70.f;
    UPROPERTY() int32 OverheadCamResX = 50;
    UPROPERTY() int32 OverheadCamResY = 50;
    UPROPERTY() class ASceneCapture2D* OverheadCaptureActor = nullptr;
    UPROPERTY() class UTextureRenderTarget2D* OverheadRenderTarget = nullptr;
};
