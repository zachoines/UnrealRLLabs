#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/MainPlatform.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GoalPlatform.h"
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
 *  - Tracks a certain number of “grid objects” (balls)
 *  - Manages velocities, accelerations, distances in local coords
 *  - Builds NxN x 10 central state: height, velocity, accel, direction
 *  - Reads from "StateManager" config block
 *  - If bUseRaycastForHeight => we do line traces per cell to find object top
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UStateManager : public UObject
{
    GENERATED_BODY()

public:
    /**
     * Initialize from the "StateManager" sub-block in JSON config.
     */
    UFUNCTION(BlueprintCallable)
    void InitializeFromConfig(UEnvironmentConfig* SMConfig);

    /**
     * Set references:
     *  - The scaled platform
     *  - The scaled grid
     *  - The object manager
     *  - The array of AGoalPlatform
     *  - The wave sim
     */
    UFUNCTION(BlueprintCallable)
    void SetReferences(
        AMainPlatform* InPlatform,
        AGridObjectManager* InObjMgr,
        AGrid* InGrid,
        const TArray<AGoalPlatform*>& InGoalPlatforms,
        UMultiAgentGaussianWaveHeightMap* InWaveSim
    );

    /**
     * Allocate arrays, set them to "respawn", etc.
     * @param NumObjects => how many "grid objects" we are actively using
     */
    UFUNCTION(BlueprintCallable)
    void Reset(int32 NumObjects);

    // ------------------------------------------------
    //   Accessor for config-based max # of objects
    // ------------------------------------------------
    UFUNCTION(BlueprintCallable)
    int32 GetMaxGridObjects() const;

    // ------------------------------------------------
    //   Object Logic
    // ------------------------------------------------

    /** Check if any object is out of bounds, reached goal, etc. */
    UFUNCTION(BlueprintCallable)
    void UpdateGridObjectFlags();

    /** Update velocity, acceleration, distances, etc. in local coords. */
    UFUNCTION(BlueprintCallable)
    void UpdateObjectStats(float DeltaTime);

    /** Respawn any objects flagged for “respawn” if their timer > delay. */
    UFUNCTION(BlueprintCallable)
    void RespawnGridObjects();

    /** Returns true if no objects are active or respawning. */
    UFUNCTION(BlueprintCallable)
    bool AllGridObjectsHandled() const;

    // ------------------------------------------------
    //   Build NxN x 10 central state
    // ------------------------------------------------
    UFUNCTION(BlueprintCallable)
    void BuildCentralState();

    UFUNCTION(BlueprintCallable)
    TArray<float> GetCentralState();

    /** Returns wave-sim agent’s 9-float state (orientation, sigma, etc.) from the UMultiAgentGaussianWaveHeightMap. */
    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentState(int32 AgentIndex) const;

    // ------------------------------------------------
    //   Accessors
    // ------------------------------------------------
    UFUNCTION(BlueprintCallable)
    bool GetHasActive(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetHasActive(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    bool GetHasReachedGoal(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetHasReachedGoal(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    bool GetHasFallenOff(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetHasFallenOff(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    bool GetShouldCollectReward(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetShouldCollectReward(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    bool GetShouldRespawn(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetShouldRespawn(int32 ObjIndex, bool bVal);

    UFUNCTION(BlueprintCallable)
    int32 GetGoalIndex(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetGoalIndex(int32 ObjIndex, int32 Goal);

    // velocity, accel, distance, position
    UFUNCTION(BlueprintCallable)
    FVector GetCurrentVelocity(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetCurrentVelocity(int32 ObjIndex, FVector val);

    UFUNCTION(BlueprintCallable)
    FVector GetPreviousVelocity(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetPreviousVelocity(int32 ObjIndex, FVector val);

    UFUNCTION(BlueprintCallable)
    FVector GetCurrentAcceleration(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetCurrentAcceleration(int32 ObjIndex, FVector val);

    UFUNCTION(BlueprintCallable)
    FVector GetPreviousAcceleration(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetPreviousAcceleration(int32 ObjIndex, FVector val);

    UFUNCTION(BlueprintCallable)
    float GetCurrentDistance(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetCurrentDistance(int32 ObjIndex, float val);

    UFUNCTION(BlueprintCallable)
    float GetPreviousDistance(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetPreviousDistance(int32 ObjIndex, float val);

    UFUNCTION(BlueprintCallable)
    FVector GetCurrentPosition(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetCurrentPosition(int32 ObjIndex, FVector val);

    UFUNCTION(BlueprintCallable)
    FVector GetPreviousPosition(int32 ObjIndex) const;
    UFUNCTION(BlueprintCallable)
    void SetPreviousPosition(int32 ObjIndex, FVector val);

private:
    void LoadConfig(UEnvironmentConfig* Config);

    /** Attempt to generate a random location inside the grid that is collision-free. */
    FVector GenerateRandomGridLocation() const;

    FMatrix2D ComputeCollisionDistanceMatrix() const;
    float RaycastColumnTopWorld(const FVector& CellWorldCenter, float waveVal) const;

private:
    UPROPERTY()
    AMainPlatform* Platform;

    UPROPERTY()
    AGridObjectManager* ObjectMgr;

    UPROPERTY()
    AGrid* Grid;

    UPROPERTY()
    TArray<AGoalPlatform*> GoalPlatforms;

    UPROPERTY()
    UMultiAgentGaussianWaveHeightMap* WaveSim;

    // config-based
    UPROPERTY()
    int32 MaxGridObjects;

    UPROPERTY()
    float GoalThreshold;
    UPROPERTY()
    float MarginXY;
    UPROPERTY()
    float MinZ;
    UPROPERTY()
    float MaxZ;
    UPROPERTY()
    float SpawnCollisionRadius;
    UPROPERTY()
    int32 MarginCells;
    UPROPERTY()
    float BoundingSphereScale;

    UPROPERTY()
    float ObjectScale;
    UPROPERTY()
    float ObjectMass;
    UPROPERTY()
    float MaxColumnHeight;

    UPROPERTY()
    bool bUseRaycastForHeight;

    /** base respawn delay for all objects. */
    UPROPERTY()
    float BaseRespawnDelay;

    // dimension
    UPROPERTY()
    int32 GridSize;

    UPROPERTY()
    float CellSize;

    UPROPERTY()
    FVector PlatformWorldSize;

    UPROPERTY()
    FVector PlatformCenter;

    UPROPERTY()
    FMatrix2D SecondOrderDeltaHeight;
    FMatrix2D PreviousDeltaHeight;
    FMatrix2D CurrentDeltaHeight;
    FMatrix2D PreviousHeight;
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

    UPROPERTY()
    float OverheadCamDistance;

    UPROPERTY()
    float OverheadCamFOV;

    UPROPERTY()
    int32 OverheadCamResX;

    UPROPERTY()
    int32 OverheadCamResY;

    // We store the capture actor + render target
    UPROPERTY()
    class ASceneCapture2D* OverheadCaptureActor;

    UPROPERTY()
    class UTextureRenderTarget2D* OverheadRenderTarget;

    // Helper that spawns the overhead camera if not yet spawned
    void SetupOverheadCamera();

    // Helper to read the overhead camera’s RT data, flatten it, return as float array
    TArray<float> CaptureOverheadImage() const;
};
