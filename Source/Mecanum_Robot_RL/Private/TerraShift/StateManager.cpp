// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.
// This version incorporates the "Fixed-Slot Reward Structure" state management system.

#include "TerraShift/StateManager.h"
#include "TerraShift/OccupancyGrid.h"
#include "TerraShift/GridObject.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GoalManager.h"
#include "TerraShift/GoalPlatform.h"
#include "TerraShift/Column.h"
#include "HeightMapGenerator.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"

// ------------------------------------------
//   LoadConfig
// ------------------------------------------
void UStateManager::LoadConfig(UEnvironmentConfig* Config)
{
    if (!Config)
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::LoadConfig => null config!"));
        return;
    }

    MaxGridObjects = Config->GetOrDefaultInt(TEXT("MaxGridObjects"), MaxGridObjects);
    MarginXY = Config->GetOrDefaultNumber(TEXT("MarginXY"), MarginXY);
    MinZ = Config->GetOrDefaultNumber(TEXT("MinZ"), MinZ);
    MaxZ = Config->GetOrDefaultNumber(TEXT("MaxZ"), MaxZ);
    MarginCells = Config->GetOrDefaultInt(TEXT("MarginCells"), MarginCells);
    ObjectMass = Config->GetOrDefaultNumber(TEXT("ObjectMass"), ObjectMass);
    ColumnZBias = Config->GetOrDefaultNumber(TEXT("ColumnZBias"), ColumnZBias);
    ObjectZBias = Config->GetOrDefaultNumber(TEXT("ObjectZBias"), ObjectZBias);
    MaxColumnHeight = Config->GetOrDefaultNumber(TEXT("MaxColumnHeight"), MaxColumnHeight);
    BaseRespawnDelay = Config->GetOrDefaultNumber(TEXT("BaseRespawnDelay"), BaseRespawnDelay);

    OverheadCamDistance = Config->GetOrDefaultNumber(TEXT("OverheadCameraDistance"), OverheadCamDistance);
    OverheadCamFOV = Config->GetOrDefaultNumber(TEXT("OverheadCameraFOV"), OverheadCamFOV);
    OverheadCamResX = Config->GetOrDefaultInt(TEXT("OverheadCameraResX"), OverheadCamResX);
    OverheadCamResY = Config->GetOrDefaultInt(TEXT("OverheadCameraResY"), OverheadCamResY);

    bUseRandomGoals = Config->GetOrDefaultBool(TEXT("bUseRandomGoals"), true);
    bRespawnOnGoal = Config->GetOrDefaultBool(TEXT("bRespawnOnGoal"), true);
    bRespawnOnOOB = Config->GetOrDefaultBool(TEXT("bRespawnOnOOB"), true);
    bTerminateOnAllGoalsReached = Config->GetOrDefaultBool(TEXT("bTerminateOnAllGoalsReached"), false);
    bTerminateOnMaxSteps = Config->GetOrDefaultBool(TEXT("bTerminateOnMaxSteps"), true);
    bRemoveObjectsOnGoal = Config->GetOrDefaultBool(TEXT("bRemoveObjectsOnGoal"), true);
    bSuppressPerStepGoalReward = Config->GetOrDefaultBool(TEXT("bSuppressPerStepGoalReward"), false);

    GoalRadius = Config->GetOrDefaultNumber(TEXT("GoalRadius"), GoalRadius);
    GoalCollectRadius = Config->GetOrDefaultNumber(TEXT("GoalCollectRadius"), GoalCollectRadius);
    ObjectRadius = Config->GetOrDefaultNumber(TEXT("ObjectRadius"), ObjectRadius);
    SpawnPaddingZ = Config->GetOrDefaultNumber(TEXT("SpawnPaddingZ"), SpawnPaddingZ);
    SpawnSeparationRadius = Config->GetOrDefaultNumber(TEXT("SpawnSeparationRadius"), SpawnSeparationRadius);
    GridSize = Config->GetOrDefaultInt(TEXT("GridSize"), GridSize);

    // State Representation Config
    bIncludeHeightMapInState = Config->GetOrDefaultBool(TEXT("bIncludeHeightMapInState"), true);
    bIncludeOverheadImageInState = Config->GetOrDefaultBool(TEXT("bIncludeOverheadImageInState"), true);

    int32 DefaultResH = GridSize > 0 ? GridSize : 25;
    int32 DefaultResW = GridSize > 0 ? GridSize : 25;

    StateHeightMapResolutionH = Config->GetOrDefaultInt(TEXT("StateHeightMapResolutionH"), DefaultResH);
    StateHeightMapResolutionW = Config->GetOrDefaultInt(TEXT("StateHeightMapResolutionW"), DefaultResW);
    StateOverheadImageResX = Config->GetOrDefaultInt(TEXT("StateOverheadImageResX"), OverheadCamResX);
    StateOverheadImageResY = Config->GetOrDefaultInt(TEXT("StateOverheadImageResY"), OverheadCamResY);

    bIncludeGridObjectSequenceInState = Config->GetOrDefaultBool(TEXT("bIncludeGridObjectSequenceInState"), false);
    MaxGridObjectsForState = Config->GetOrDefaultInt(TEXT("MaxGridObjectsForState"), MaxGridObjects);
    GridObjectFeatureSize = Config->GetOrDefaultInt(TEXT("GridObjectFeatureSize"), 9);

    // Optional optimizations
    // Backwards-compat: accept legacy key bEnableColumnCollisionOptimization
    bool CollisionROI = false;
    if (Config->HasPath(TEXT("bRestrictColumnPhysicsToRadius")))
    {
        CollisionROI = Config->GetOrDefaultBool(TEXT("bRestrictColumnPhysicsToRadius"), false);
    }
    else
    {
        CollisionROI = Config->GetOrDefaultBool(TEXT("bEnableColumnCollisionOptimization"), false);
    }
    bRestrictColumnPhysicsToRadius = CollisionROI;
    ColumnCollisionRadiusCells = Config->GetOrDefaultInt(TEXT("ColumnCollisionRadiusCells"), 2);
    // Decoupled shader-based heightmap toggle (defaults to collision ROI when unspecified)
    bEnabledShaderBasedHeightmap = Config->GetOrDefaultBool(TEXT("bEnabledShaderBasedHeightmap"), bRestrictColumnPhysicsToRadius);
    // Restrict movement/update of columns to ROI around objects
    bRestrictColumnMovementToRadius = Config->GetOrDefaultBool(TEXT("bRestrictColumnMovementToRadius"), false);

    if (bRestrictColumnPhysicsToRadius && !bEnabledShaderBasedHeightmap)
    {
        UE_LOG(LogTemp, Warning, TEXT("StateManager::LoadConfig => bRestrictColumnPhysicsToRadius=true while shader heightmap disabled. Using CPU analytic heightmap (no line traces)."));
    }

    if (Config->HasPath(TEXT("GoalColors")))
    {
        TArray<UEnvironmentConfig*> colorArr = Config->Get(TEXT("GoalColors"))->AsArrayOfConfigs();
        GoalColors.Empty();
        for (auto* c : colorArr)
        {
            TArray<float> vals = c->AsArrayOfNumbers();
            if (vals.Num() >= 3)
            {
                FLinearColor col(vals[0], vals[1], vals[2], (vals.Num() >= 4 ? vals[3] : 1.f));
                GoalColors.Add(col);
            }
        }
    }
    if (Config->HasPath(TEXT("GridObjectColors")))
    {
        TArray<UEnvironmentConfig*> gridColArr = Config->Get(TEXT("GridObjectColors"))->AsArrayOfConfigs();
        GridObjectColors.Empty();
        for (auto* g : gridColArr)
        {
            TArray<float> vals = g->AsArrayOfNumbers();
            if (vals.Num() >= 3)
            {
                FLinearColor col(vals[0], vals[1], vals[2], (vals.Num() >= 4 ? vals[3] : 1.f));
                GridObjectColors.Add(col);
            }
        }
    }
}

// ------------------------------------------
//   SetReferences
// ------------------------------------------
void UStateManager::SetReferences(
    AMainPlatform* InPlatform,
    AGridObjectManager* InObjMgr,
    AGrid* InGrid,
    UMultiAgentGaussianWaveHeightMap* InWaveSim,
    AGoalManager* InGoalManager
)
{
    Platform = InPlatform;
    ObjectMgr = InObjMgr;
    Grid = InGrid;
    WaveSim = InWaveSim;
    GoalManager = InGoalManager;
    if (ObjectMgr)
    {
        ObjectMgr->SetSpawnPaddingZ(SpawnPaddingZ);
    }

    checkf(Platform, TEXT("StateManager::SetReferences => Platform is null!"));
    checkf(ObjectMgr, TEXT("StateManager::SetReferences => ObjectMgr is null!"));
    checkf(Grid, TEXT("StateManager::SetReferences => Grid is null!"));
    checkf(WaveSim, TEXT("StateManager::SetReferences => WaveSim is null!"));
    checkf(GoalManager, TEXT("StateManager::SetReferences => GoalManager is null!"));

    GridSize = (Grid->GetTotalColumns() > 0)
        ? FMath::FloorToInt(FMath::Sqrt((float)Grid->GetTotalColumns()))
        : 50;

    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
        * 2.f
        * Platform->GetActorScale3D();

    PlatformCenter = Platform->GetActorLocation();
    CellSize = (GridSize > 0) ? (PlatformWorldSize.X / (float)GridSize) : 1.f;

    if (!OccupancyGrid)
    {
        OccupancyGrid = NewObject<UOccupancyGrid>(this);
    }
    OccupancyGrid->InitGrid(GridSize, PlatformWorldSize.X, PlatformCenter);

    SetupOverheadCamera();
}

// ------------------------------------------
//   Reset
// ------------------------------------------
void UStateManager::Reset(int32 NumObjects, int32 CurrentAgents)
{
    checkf(Platform, TEXT("StateManager::Reset => Platform is null!"));
    checkf(ObjectMgr, TEXT("StateManager::Reset => ObjectMgr is null!"));
    checkf(Grid, TEXT("StateManager::Reset => Grid is null!"));
    checkf(WaveSim, TEXT("StateManager::Reset => WaveSim is null!"));
    checkf(GoalManager, TEXT("StateManager::Reset => GoalManager is null!"));
    checkf(OccupancyGrid, TEXT("StateManager::Reset => OccupancyGrid is null!"));

    // 1) Reset environment sub-components
    Grid->ResetGrid();
    ObjectMgr->ResetGridObjects();
    WaveSim->Reset(CurrentAgents);

    // 2) Allocate local arrays for 'NumObjects'
    ObjectSlotStates.SetNum(NumObjects);
    bShouldCollect.SetNum(NumObjects);
    bShouldResp.SetNum(NumObjects);
    ObjectGoalIndices.SetNum(NumObjects);
    PrevVel.SetNum(NumObjects);
    CurrVel.SetNum(NumObjects);
    PrevAcc.SetNum(NumObjects);
    CurrAcc.SetNum(NumObjects);
    PrevDist.SetNum(NumObjects);
    CurrDist.SetNum(NumObjects);
    PrevPos.SetNum(NumObjects);
    CurrPos.SetNum(NumObjects);
    RespawnTimer.SetNum(NumObjects);
    RespawnDelays.SetNum(NumObjects);

    // Initialize per-object states
    for (int32 i = 0; i < NumObjects; i++)
    {
        ObjectSlotStates[i] = EObjectSlotState::Empty;
        bShouldCollect[i] = false;
        bShouldResp[i] = true; // Start by marking for respawn

        ObjectGoalIndices[i] = -1;

        PrevVel[i] = FVector::ZeroVector;
        CurrVel[i] = FVector::ZeroVector;
        PrevAcc[i] = FVector::ZeroVector;
        CurrAcc[i] = FVector::ZeroVector;
        PrevDist[i] = -1.f;
        CurrDist[i] = -1.f;
        PrevPos[i] = FVector::ZeroVector;
        CurrPos[i] = FVector::ZeroVector;

        RespawnTimer[i] = 0.f;
        // Stagger initial spawns so each object appears BaseRespawnDelay apart
        RespawnDelays[i] = BaseRespawnDelay * static_cast<float>(i);
    }

    // 3) Reset NxN height arrays & step counter
    int32 StateH = (bIncludeHeightMapInState && StateHeightMapResolutionH > 0) ? StateHeightMapResolutionH : GridSize;
    int32 StateW = (bIncludeHeightMapInState && StateHeightMapResolutionW > 0) ? StateHeightMapResolutionW : GridSize;

    PreviousHeight = FMatrix2D(StateH, StateW, 0.f);
    CurrentHeight = FMatrix2D(StateH, StateW, 0.f);
    Step = 0;
    HeightMapGPUAsync.Reset();
    // Clear previous enabled column cache
    PrevEnabledColumnCells.Empty();

    // 4) Clear occupancy
    OccupancyGrid->ResetGrid();

    // 5) Place random goals and gather them for GoalManager
    TArray<AActor*> newGoalActors;
    TArray<FVector> newGoalOffsets;

    const FName GoalsLayerName(TEXT("Goals"));

    if (bUseRandomGoals)
    {
        const int32 DesiredGoalCount = GoalColors.Num();
        if (DesiredGoalCount <= 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("StateManager::Reset => GoalColors array is empty. No random goals will be spawned."));
        }
        else
        {
            const float OriginalGoalRadius = GoalRadius;
            float WorkingGoalRadius = GoalRadius;
            constexpr float MinimumGoalRadius = 1.0f;
            bool bPlacedAllGoals = false;

            auto TryPlaceGoals = [&](float RadiusToUse) -> bool
            {
                OccupancyGrid->ResetGrid();
                newGoalActors.Reset();
                newGoalOffsets.Reset();

                const float RadiusCells = (CellSize > 0.f) ? (RadiusToUse / CellSize) : RadiusToUse;

                for (int32 GoalIdx = 0; GoalIdx < DesiredGoalCount; ++GoalIdx)
                {
                    int32 PlacedCell = OccupancyGrid->AddObjectToGrid(GoalIdx, GoalsLayerName, RadiusCells, TArray<FName>());
                    if (PlacedCell < 0)
                    {
                        return false;
                    }

                    const int32 gx = PlacedCell / GridSize;
                    const int32 gy = PlacedCell % GridSize;
                    const int32 ColumnIndex = gx * GridSize + gy;

                    if (!Grid->Columns.IsValidIndex(ColumnIndex))
                    {
                        return false;
                    }

                    AColumn* ColumnActor = Grid->Columns[ColumnIndex];
                    if (!ColumnActor || !ColumnActor->ColumnMesh)
                    {
                        return false;
                    }

                    const float ColumnHeight = ColumnActor->ColumnMesh->Bounds.BoxExtent.Z * 2.0f;
                    const float GoalOffsetZ = ColumnHeight + ObjectRadius;
                    newGoalActors.Add(ColumnActor);
                    newGoalOffsets.Add(FVector(0.f, 0.f, GoalOffsetZ));
                }

                return true;
            };

            while (WorkingGoalRadius >= MinimumGoalRadius && !bPlacedAllGoals)
            {
                if (TryPlaceGoals(WorkingGoalRadius))
                {
                    bPlacedAllGoals = true;
                    // Do not persistently mutate GoalRadius; only report if a reduction was needed
                    if (!FMath::IsNearlyEqual(WorkingGoalRadius, OriginalGoalRadius))
                    {
                        UE_LOG(LogTemp, Warning, TEXT("StateManager::Reset => Effective placement GoalRadius reduced from %f to %f to place all %d goals (no persistent change)."), OriginalGoalRadius, WorkingGoalRadius, DesiredGoalCount);
                    }
                }
                else
                {
                    WorkingGoalRadius -= 1.0f;
                }
            }

            if (!bPlacedAllGoals)
            {
                UE_LOG(LogTemp, Fatal, TEXT("StateManager::Reset => Failed to place %d goals even after reducing GoalRadius below %f (original: %f)."), DesiredGoalCount, MinimumGoalRadius, OriginalGoalRadius);
            }
        }
    }

    GoalManager->ResetGoals(newGoalActors, newGoalOffsets);

    const int32 ActualGoalCount = GoalManager ? GoalManager->GetNumGoals() : 0;
    for (int32 i = 0; i < NumObjects; ++i)
    {
        ObjectGoalIndices[i] = (ActualGoalCount > 0) ? (i % ActualGoalCount) : -1;
    }
}

// ------------------------------------------
//   UpdateGridObjectFlags
// ------------------------------------------
void UStateManager::UpdateGridObjectFlags()
{
    float halfX = PlatformWorldSize.X * 0.5f;
    float halfY = PlatformWorldSize.Y * 0.5f;
    float minZLocal = PlatformCenter.Z + MinZ;
    float maxZLocal = PlatformCenter.Z + MaxZ;

    for (int32 i = 0; i < ObjectSlotStates.Num(); i++)
    {
        // Skip objects that are not active, except goal-reached objects for when bRemoveObjectsOnGoal is false
        if (ObjectSlotStates[i] != EObjectSlotState::Active && 
            !(ObjectSlotStates[i] == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal))
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue;

        FVector wPos = Obj->GetObjectLocation();

        // (1) Check goal status and handle state transitions
        int32 gIdx = GetGoalIndex(i);
        if (gIdx >= 0)
        {
            bool bInRadius = GoalManager->IsInRadiusOf(gIdx, wPos, GoalCollectRadius);
            
            if (bInRadius)
            {
                // Object is at goal
                if (ObjectSlotStates[i] == EObjectSlotState::Active)
                {
                    // First time reaching goal - collect reward
                    bShouldCollect[i] = true;
                    bShouldResp[i] = bRespawnOnGoal;
                    
                    // Handle based on bRemoveObjectsOnGoal setting
                    if (bRemoveObjectsOnGoal)
                    {
                        ObjectSlotStates[i] = EObjectSlotState::GoalReached;
                        ObjectMgr->DisableGridObject(i);
                        OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                        continue; // Skip OOB check since object is now disabled
                    }
                    else
                    {
                        // Set to GoalReached for reward collection
                        ObjectSlotStates[i] = EObjectSlotState::GoalReached;
                    }
                } 
                // If already GoalReached, optionally suppress per-step event rewards
                else if (ObjectSlotStates[i] == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal)
                {
                    if (!bSuppressPerStepGoalReward)
                    {
                        bShouldCollect[i] = true;
                    }
                }
                
            }
            else
            {
                // Object is NOT at goal
                if (ObjectSlotStates[i] == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal)
                {
                    // Object moved away from goal, return to Active
                    ObjectSlotStates[i] = EObjectSlotState::Active;
                }
            }
        }

        // (2) OOB Check - for active objects and goal-reached for when bRemoveObjectsOnGoal is false
        if (ObjectSlotStates[i] == EObjectSlotState::Active || 
            (ObjectSlotStates[i] == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal))
        {
            float dx = FMath::Abs(wPos.X - PlatformCenter.X);
            float dy = FMath::Abs(wPos.Y - PlatformCenter.Y);
            float zPos = wPos.Z;
            bool bOOB = (dx > (halfX + MarginXY) || dy > (halfY + MarginXY) || zPos < minZLocal || zPos > maxZLocal);

            if (bOOB)
            {
                ObjectSlotStates[i] = EObjectSlotState::OutOfBounds;
                bShouldCollect[i] = true;
                bShouldResp[i] = bRespawnOnOOB;

                if (!bShouldResp[i])
                {
                    ObjectMgr->DisableGridObject(i);
                    OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                }
            }
        }
    }
}

// ------------------------------------------
//   UpdateObjectStats
// ------------------------------------------
void UStateManager::UpdateObjectStats(float DeltaTime)
{
    for (int32 i = 0; i < MaxGridObjects; i++)
    {
        // Zero out stats for inactive/terminal objects (except goal-reached in for when bRemoveObjectsOnGoal is false)
        if (GetObjectSlotState(i) != EObjectSlotState::Active && 
            !(GetObjectSlotState(i) == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal))
        {
            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f;
            CurrDist[i] = -1.f;
            PrevPos[i] = FVector::ZeroVector;
            CurrPos[i] = FVector::ZeroVector;

            if (bShouldResp.IsValidIndex(i) && bShouldResp[i] && RespawnTimer.IsValidIndex(i))
            {
                RespawnTimer[i] += DeltaTime;
            }
            continue;
        }

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue;

        FVector wVel = Obj->MeshComponent->GetPhysicsLinearVelocity();
        FVector wPos = Obj->GetObjectLocation();
        int32 gIdx = GetGoalIndex(i);
        FVector wGoal = (gIdx >= 0 && GoalManager) ? GoalManager->GetGoalLocation(gIdx) : FVector::ZeroVector;

        FVector locVel = Platform->GetActorTransform().InverseTransformVector(wVel);
        FVector locPos = Platform->GetActorTransform().InverseTransformPosition(wPos);
        FVector locGoal = Platform->GetActorTransform().InverseTransformPosition(wGoal);

        PrevVel[i] = CurrVel[i];
        PrevAcc[i] = CurrAcc[i];
        PrevDist[i] = CurrDist[i];
        PrevPos[i] = CurrPos[i];

        CurrVel[i] = locVel;
        CurrAcc[i] = (DeltaTime > SMALL_NUMBER) ? (locVel - PrevVel[i]) / DeltaTime : FVector::ZeroVector;
        CurrPos[i] = locPos;
        CurrDist[i] = FVector::Dist(locPos, locGoal);

        if (OccupancyGrid && CellSize > 0)
        {
            int32 cellIdx = OccupancyGrid->WorldToGrid(wPos);
            const float SepRadiusWorld = (SpawnSeparationRadius > 0.f) ? SpawnSeparationRadius : ObjectRadius;
            float radiusCells = SepRadiusWorld / CellSize;
            OccupancyGrid->UpdateObjectPosition(i, FName("GridObjects"), cellIdx, radiusCells, TArray<FName>{ FName("Goals") });
        }
    }
}

// ------------------------------------------
//   RespawnGridObjects
// ------------------------------------------
void UStateManager::RespawnGridObjects()
{
    int32 nObjColors = GridObjectColors.Num();
    if (nObjColors == 0)
    {
        GridObjectColors.Add(FLinearColor::White);
        nObjColors = 1;
    }

    for (int32 i = 0; i < MaxGridObjects; i++)
    {
        if (!bShouldResp.IsValidIndex(i) || !bShouldResp[i] || !RespawnTimer.IsValidIndex(i) || !RespawnDelays.IsValidIndex(i)) continue;

        if (RespawnTimer[i] >= RespawnDelays[i])
        {
            if (OccupancyGrid) OccupancyGrid->RemoveObject(i, FName("GridObjects"));

            const float SepRadiusWorld2 = (SpawnSeparationRadius > 0.f) ? SpawnSeparationRadius : ObjectRadius;
            float radiusCells = (CellSize > 0) ? (SepRadiusWorld2 / CellSize) : 1.0f;
            int32 cellIdx = OccupancyGrid ? OccupancyGrid->AddObjectToGrid(i, FName("GridObjects"), radiusCells, TArray<FName>{}) : -1;

            if (cellIdx < 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("No free occupant cell for obj %d, will try again."), i);
                continue;
            }

            int32 gx = cellIdx / GridSize;
            int32 gy = cellIdx % GridSize;
            FVector spawnLoc = GetColumnTopWorldLocation(gx, gy);

            // Match visual mesh radius to ObjectRadius so physics + occupancy + overlays agree
            const float VisualScale = (ObjectUnscaledSize > KINDA_SMALL_NUMBER) ? (ObjectRadius / ObjectUnscaledSize) : 1.0f;
            if (ObjectMgr) ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(VisualScale), ObjectMass);

            AGridObject* newObj = ObjectMgr ? ObjectMgr->GetGridObject(i) : nullptr;
            if (newObj)
            {
                newObj->SetGridObjectColor(GridObjectColors[i % nObjColors]);
            }

            ObjectSlotStates[i] = EObjectSlotState::Active;
            bShouldResp[i] = false;
            RespawnTimer[i] = 0.f;
            // Reset delay so subsequent respawns use the base value
            RespawnDelays[i] = BaseRespawnDelay;

            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f;
            CurrDist[i] = -1.f;
            PrevPos[i] = Platform->GetActorTransform().InverseTransformPosition(spawnLoc);
            CurrPos[i] = PrevPos[i];
        }
    }
}

// ------------------------------------------
//   AllGridObjectsHandled
// ------------------------------------------
bool UStateManager::AllGridObjectsHandled() const
{
    for (int32 i = 0; i < MaxGridObjects; ++i)
    {
        EObjectSlotState state = GetObjectSlotState(i);
        // Objects are still "in play" if they're Active, Empty, or GoalReached for when bRemoveObjectsOnGoal is false
        if (state == EObjectSlotState::Active || 
            state == EObjectSlotState::Empty ||
            (state == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal))
        {
            return false;
        }
    }
    return true;
}

// ------------------------------------------
//   BuildCentralState
// ------------------------------------------
void UStateManager::BuildCentralState()
{
    UWorld* w = Grid ? Grid->GetWorld() : nullptr;
    if (!w || !Grid || !Platform) return;

    int32 PhysicalGridSizeH = GridSize > 0 ? GridSize : 1;
    int32 PhysicalGridSizeW = GridSize > 0 ? GridSize : 1;
    int32 CurrentStateMapH = bIncludeHeightMapInState ? (StateHeightMapResolutionH > 0 ? StateHeightMapResolutionH : PhysicalGridSizeH) : PhysicalGridSizeH;
    int32 CurrentStateMapW = bIncludeHeightMapInState ? (StateHeightMapResolutionW > 0 ? StateHeightMapResolutionW : PhysicalGridSizeW) : PhysicalGridSizeW;

    if (CurrentHeight.GetNumRows() != CurrentStateMapH || CurrentHeight.GetNumColumns() != CurrentStateMapW) {
        PreviousHeight.Resize(CurrentStateMapH, CurrentStateMapW, FMatrix2D::EInitialization::Zero);
        CurrentHeight.Resize(CurrentStateMapH, CurrentStateMapW, FMatrix2D::EInitialization::Zero);
    }

    FMatrix2D HeightTmp(CurrentStateMapH, CurrentStateMapW, 0.f);
    if (bIncludeHeightMapInState)
    {
        FTransform GridTransform = Grid->GetActorTransform();
        const float visMinZ = MinZ;
        const float visMaxZ = MaxZ;

        // If shader-based heightmap is enabled, try GPU heightmap generation first
        if (bEnabledShaderBasedHeightmap && WaveSim)
        {
            FHeightMapGPUAsyncState* AsyncStatePtr = nullptr;
            if (!HeightMapGPUAsync)
            {
                HeightMapGPUAsync = MakeUnique<FHeightMapGPUAsyncState>();
            }
            AsyncStatePtr = HeightMapGPUAsync.Get();

            if (AsyncStatePtr)
            {
                AsyncStatePtr->EnsureDimensions(CurrentStateMapW, CurrentStateMapH);

                bool bHasFreshGPUState = false;
                if (AsyncStatePtr->DispatchHandle.IsActive())
                {
                    if (ResolveHeightMapGPU(AsyncStatePtr->DispatchHandle, AsyncStatePtr->BackState))
                    {
                        Swap(AsyncStatePtr->FrontState, AsyncStatePtr->BackState);
                        AsyncStatePtr->bFrontValid = true;
                        bHasFreshGPUState = true;
                    }
                }

                if (!AsyncStatePtr->DispatchHandle.IsActive())
                {
                    const int32 ColCount = GridSize * GridSize;
                    AsyncStatePtr->ColumnCentersScratch.SetNum(ColCount);
                    AsyncStatePtr->ColumnRadiiScratch.SetNum(ColCount);

                    if (Grid)
                    {
                        for (int32 x = 0; x < GridSize; ++x)
                        {
                            for (int32 y = 0; y < GridSize; ++y)
                            {
                                const int32 idx = y * GridSize + x;
                                const int32 colIndex = x * GridSize + y;
                                if (Grid->Columns.IsValidIndex(colIndex))
                                {
                                    AColumn* Col = Grid->Columns[colIndex];
                                    if (Col && Col->ColumnMesh)
                                    {
                                        const FVector wpos = Col->GetActorLocation();
                                        const FVector local = GridTransform.InverseTransformPosition(wpos);
                                        AsyncStatePtr->ColumnCentersScratch[idx] = FVector3f((float)local.X, (float)local.Y, (float)local.Z);

                                        const FBoxSphereBounds CB = Col->ColumnMesh->Bounds;
                                        AsyncStatePtr->ColumnRadiiScratch[idx] = FVector3f((float)CB.BoxExtent.X, (float)CB.BoxExtent.Y, (float)CB.BoxExtent.Z);
                                    }
                                }
                            }
                        }
                    }

                    AsyncStatePtr->ObjCentersScratch.Reset();
                    AsyncStatePtr->ObjRadiiScratch.Reset();
                    AsyncStatePtr->ObjCentersScratch.Reserve(ObjectSlotStates.Num());
                    AsyncStatePtr->ObjRadiiScratch.Reserve(ObjectSlotStates.Num());

                    int32 NumActiveObjects = 0;
                    if (ObjectMgr)
                    {
                        for (int32 iObj = 0; iObj < ObjectSlotStates.Num(); ++iObj)
                        {
                            if (ObjectSlotStates[iObj] != EObjectSlotState::Active)
                            {
                                continue;
                            }

                            AGridObject* Obj = ObjectMgr->GetGridObject(iObj);
                            if (!Obj || !Obj->MeshComponent || !Obj->IsActive())
                            {
                                continue;
                            }

                            const FVector objLocal = GridTransform.InverseTransformPosition(Obj->GetObjectLocation());
                            const float radius = ObjectRadius;
                            AsyncStatePtr->ObjCentersScratch.Add(FVector3f((float)objLocal.X, (float)objLocal.Y, (float)objLocal.Z));
                            AsyncStatePtr->ObjRadiiScratch.Add(FVector3f(radius, radius, radius));
                            ++NumActiveObjects;
                        }
                    }

                    AsyncStatePtr->BufferedNumObjects = NumActiveObjects;

                    FHeightMapGenParams GP;
                    GP.GridSize = GridSize;
                    GP.StateW = CurrentStateMapW;
                    GP.StateH = CurrentStateMapH;
                    GP.PlatformSize = FVector2D(PlatformWorldSize.X, PlatformWorldSize.Y);
                    GP.CellSize = CellSize;
                    GP.MinZ = visMinZ;
                    GP.MaxZ = visMaxZ;
                    GP.ColZBias = ColumnZBias;
                    GP.ObjZBias = ObjectZBias;
                    GP.NumObjects = NumActiveObjects;
                    if (NumActiveObjects == 0)
                    {
                        AsyncStatePtr->ObjCentersScratch.SetNum(1);
                        AsyncStatePtr->ObjCentersScratch[0] = FVector3f::ZeroVector;
                        AsyncStatePtr->ObjRadiiScratch.SetNum(1);
                        AsyncStatePtr->ObjRadiiScratch[0] = FVector3f::ZeroVector;
                    }

                    if (!DispatchHeightMapGPU(GP,
                                              AsyncStatePtr->ColumnCentersScratch,
                                              AsyncStatePtr->ColumnRadiiScratch,
                                              AsyncStatePtr->ObjCentersScratch,
                                              AsyncStatePtr->ObjRadiiScratch,
                                              AsyncStatePtr->DispatchHandle))
                    {
                        AsyncStatePtr->DispatchHandle.Reset();
                    }
                }

                if (bHasFreshGPUState)
                {
                    const TArray<float>& GPUState = AsyncStatePtr->FrontState;
                    const int32 ExpectedCount = CurrentStateMapW * CurrentStateMapH;
                    if (GPUState.Num() == ExpectedCount)
                    {
                        for (int32 r = 0; r < CurrentStateMapH; ++r)
                        {
                            const int32 rowStart = r * CurrentStateMapW;
                            for (int32 c = 0; c < CurrentStateMapW; ++c)
                            {
                                HeightTmp[r][c] = GPUState[rowStart + c];
                            }
                        }

                        PreviousHeight = CurrentHeight;
                        CurrentHeight = HeightTmp;

                        if (bIncludeOverheadImageInState && OverheadCaptureActor)
                        {
                            OverheadCaptureActor->GetCaptureComponent2D()->CaptureScene();
                        }

                        Step++;
                        return;
                    }

                    UE_LOG(LogTemp, Warning, TEXT("StateManager::BuildCentralState => GPU height map size mismatch (expected %d, got %d)"), ExpectedCount, GPUState.Num());
                    AsyncStatePtr->bFrontValid = false;
                }

                if (AsyncStatePtr->bFrontValid)
                {
                    const TArray<float>& GPUState = AsyncStatePtr->FrontState;
                    const int32 ExpectedCount = CurrentStateMapW * CurrentStateMapH;
                    if (GPUState.Num() == ExpectedCount)
                    {
                        for (int32 r = 0; r < CurrentStateMapH; ++r)
                        {
                            const int32 rowStart = r * CurrentStateMapW;
                            for (int32 c = 0; c < CurrentStateMapW; ++c)
                            {
                                HeightTmp[r][c] = GPUState[rowStart + c];
                            }
                        }

                        PreviousHeight = CurrentHeight;
                        CurrentHeight = HeightTmp;

                        if (bIncludeOverheadImageInState && OverheadCaptureActor)
                        {
                            OverheadCaptureActor->GetCaptureComponent2D()->CaptureScene();
                        }

                        Step++;
                        return;
                    }

                    UE_LOG(LogTemp, Warning, TEXT("StateManager::BuildCentralState => GPU cached state size mismatch (expected %d, got %d)"), ExpectedCount, GPUState.Num());
                    AsyncStatePtr->bFrontValid = false;
                    AsyncStatePtr->FrontState.Reset();
                }
            }

            const float InvRangeZ = (visMaxZ > visMinZ) ? 1.0f / (visMaxZ - visMinZ) : 0.0f;
            const float HalfX = PlatformWorldSize.X * 0.5f;
            const float HalfY = PlatformWorldSize.Y * 0.5f;
            const float SafeCellSize = (CellSize > KINDA_SMALL_NUMBER) ? CellSize : 1.f;

            const int32 BufferedObjects = AsyncStatePtr ? AsyncStatePtr->BufferedNumObjects : 0;
            const int32 ColumnScratchCount = AsyncStatePtr ? AsyncStatePtr->ColumnCentersScratch.Num() : 0;

            for (int32 RowIdx = 0; RowIdx < CurrentStateMapH; ++RowIdx)
            {
                const float normY = (CurrentStateMapH > 1) ? static_cast<float>(RowIdx) / static_cast<float>(CurrentStateMapH - 1) : 0.5f;
                const float ly = (normY - 0.5f) * PlatformWorldSize.Y;
                const float fy = (ly + HalfY) / SafeCellSize - 0.5f;
                const int iy0 = FMath::Clamp((int32)FMath::FloorToFloat(fy), 0, GridSize - 1);
                const int iy1 = FMath::Clamp(iy0 + 1, 0, GridSize - 1);

                for (int32 ColIdx = 0; ColIdx < CurrentStateMapW; ++ColIdx)
                {
                    const float normX = (CurrentStateMapW > 1) ? static_cast<float>(ColIdx) / static_cast<float>(CurrentStateMapW - 1) : 0.5f;
                    const float lx = (normX - 0.5f) * PlatformWorldSize.X;

                    float zLocal = visMinZ;

                    if (AsyncStatePtr && ColumnScratchCount > 0)
                    {
                        const float fx = (lx + HalfX) / SafeCellSize - 0.5f;
                        const int ix0 = FMath::Clamp((int32)FMath::FloorToFloat(fx), 0, GridSize - 1);
                        const int ix1 = FMath::Clamp(ix0 + 1, 0, GridSize - 1);
                        const int candIdx[4] = { iy0 * GridSize + ix0, iy1 * GridSize + ix0, iy0 * GridSize + ix1, iy1 * GridSize + ix1 };

                        for (int candidate = 0; candidate < 4; ++candidate)
                        {
                            const int ci = candIdx[candidate];
                            if (!AsyncStatePtr->ColumnCentersScratch.IsValidIndex(ci) || !AsyncStatePtr->ColumnRadiiScratch.IsValidIndex(ci))
                            {
                                continue;
                            }

                            const FVector3f& cc = AsyncStatePtr->ColumnCentersScratch[ci];
                            const FVector3f& cr = AsyncStatePtr->ColumnRadiiScratch[ci];
                            const float rx = FMath::Max(cr.X, KINDA_SMALL_NUMBER);
                            const float ry = FMath::Max(cr.Y, KINDA_SMALL_NUMBER);
                            const float rz = FMath::Max(cr.Z, KINDA_SMALL_NUMBER);
                            const float dx = lx - cc.X;
                            const float dy = ly - cc.Y;
                            const float nx = dx / rx;
                            const float ny = dy / ry;
                            const float r2 = nx * nx + ny * ny;

                            if (r2 <= 1.0f)
                            {
                                const float candidateZ = (cc.Z + ColumnZBias) + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - r2));
                                zLocal = FMath::Max(zLocal, candidateZ);
                            }
                        }
                    }

                    if (AsyncStatePtr && BufferedObjects > 0)
                    {
                        for (int32 objIdx = 0; objIdx < BufferedObjects; ++objIdx)
                        {
                            if (!AsyncStatePtr->ObjCentersScratch.IsValidIndex(objIdx) || !AsyncStatePtr->ObjRadiiScratch.IsValidIndex(objIdx))
                            {
                                continue;
                            }

                            const FVector3f& oc = AsyncStatePtr->ObjCentersScratch[objIdx];
                            const FVector3f& orad = AsyncStatePtr->ObjRadiiScratch[objIdx];
                            const float rx = FMath::Max(orad.X, KINDA_SMALL_NUMBER);
                            const float ry = FMath::Max(orad.Y, KINDA_SMALL_NUMBER);
                            const float rz = FMath::Max(orad.Z, KINDA_SMALL_NUMBER);
                            const float onx = (lx - oc.X) / rx;
                            const float ony = (ly - oc.Y) / ry;
                            const float or2 = onx * onx + ony * ony;

                            if (or2 <= 1.0f)
                            {
                                const float candidateZ = (oc.Z + ObjectZBias) + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - or2));
                                zLocal = FMath::Max(zLocal, candidateZ);
                            }
                        }
                    }

                    const float zClamped = FMath::Clamp(zLocal, visMinZ, visMaxZ);
                    const float normHeight = (zClamped - visMinZ) * InvRangeZ;
                    HeightTmp[RowIdx][ColIdx] = FMath::Clamp(normHeight * 2.f - 1.f, -1.f, 1.f);
                }
            }

            HeightTmp.Clip(-1.f, 1.f);

            if (AsyncStatePtr)
            {
                const int32 ExpectedCount = CurrentStateMapW * CurrentStateMapH;
                AsyncStatePtr->FrontState.SetNum(ExpectedCount, EAllowShrinking::No);
                for (int32 r = 0; r < CurrentStateMapH; ++r)
                {
                    const int32 rowStart = r * CurrentStateMapW;
                    for (int32 c = 0; c < CurrentStateMapW; ++c)
                    {
                        AsyncStatePtr->FrontState[rowStart + c] = HeightTmp[r][c];
                    }
                }
                AsyncStatePtr->bFrontValid = true;
            }
        }
        else if (bRestrictColumnPhysicsToRadius)
        {
            // Analytic CPU heightmap using column/object geometry; avoids line traces
            // Reuse GridTransform from the parent scope

            FHeightMapGPUAsyncState* AsyncStatePtr = nullptr;
            if (!HeightMapGPUAsync)
            {
                HeightMapGPUAsync = MakeUnique<FHeightMapGPUAsyncState>();
            }
            AsyncStatePtr = HeightMapGPUAsync.Get();
            if (AsyncStatePtr)
            {
                AsyncStatePtr->EnsureDimensions(CurrentStateMapW, CurrentStateMapH);

                const int32 ColCount = GridSize * GridSize;
                AsyncStatePtr->ColumnCentersScratch.SetNum(ColCount);
                AsyncStatePtr->ColumnRadiiScratch.SetNum(ColCount);

                if (Grid)
                {
                    for (int32 x = 0; x < GridSize; ++x)
                    {
                        for (int32 y = 0; y < GridSize; ++y)
                        {
                            const int32 idx = y * GridSize + x;
                            const int32 colIndex = x * GridSize + y;
                            if (Grid->Columns.IsValidIndex(colIndex))
                            {
                                AColumn* Col = Grid->Columns[colIndex];
                                if (Col && Col->ColumnMesh)
                                {
                                    const FVector wpos = Col->GetActorLocation();
                                    const FVector local = GridTransform.InverseTransformPosition(wpos);
                                    AsyncStatePtr->ColumnCentersScratch[idx] = FVector3f((float)local.X, (float)local.Y, (float)local.Z);

                                    const FBoxSphereBounds CB = Col->ColumnMesh->Bounds;
                                    AsyncStatePtr->ColumnRadiiScratch[idx] = FVector3f((float)CB.BoxExtent.X, (float)CB.BoxExtent.Y, (float)CB.BoxExtent.Z);
                                }
                            }
                        }
                    }
                }

                AsyncStatePtr->ObjCentersScratch.Reset();
                AsyncStatePtr->ObjRadiiScratch.Reset();
                AsyncStatePtr->ObjCentersScratch.Reserve(ObjectSlotStates.Num());
                AsyncStatePtr->ObjRadiiScratch.Reserve(ObjectSlotStates.Num());

                int32 NumActiveObjects = 0;
                if (ObjectMgr)
                {
                    for (int32 iObj = 0; iObj < ObjectSlotStates.Num(); ++iObj)
                    {
                        if (ObjectSlotStates[iObj] != EObjectSlotState::Active)
                        {
                            continue;
                        }

                        AGridObject* Obj = ObjectMgr->GetGridObject(iObj);
                        if (!Obj || !Obj->MeshComponent || !Obj->IsActive())
                        {
                            continue;
                        }

                        const FVector objLocal = GridTransform.InverseTransformPosition(Obj->GetObjectLocation());
                        const float radius = ObjectRadius;
                        AsyncStatePtr->ObjCentersScratch.Add(FVector3f((float)objLocal.X, (float)objLocal.Y, (float)objLocal.Z));
                        AsyncStatePtr->ObjRadiiScratch.Add(FVector3f(radius, radius, radius));
                        ++NumActiveObjects;
                    }
                }

                AsyncStatePtr->BufferedNumObjects = NumActiveObjects;

                const float InvRangeZ = (visMaxZ > visMinZ) ? 1.0f / (visMaxZ - visMinZ) : 0.0f;
                const float HalfX = PlatformWorldSize.X * 0.5f;
                const float HalfY = PlatformWorldSize.Y * 0.5f;
                const float SafeCellSize = (CellSize > KINDA_SMALL_NUMBER) ? CellSize : 1.f;

                const int32 BufferedObjects = AsyncStatePtr ? AsyncStatePtr->BufferedNumObjects : 0;
                const int32 ColumnScratchCount = AsyncStatePtr ? AsyncStatePtr->ColumnCentersScratch.Num() : 0;

                for (int32 RowIdx = 0; RowIdx < CurrentStateMapH; ++RowIdx)
                {
                    const float normY = (CurrentStateMapH > 1) ? static_cast<float>(RowIdx) / static_cast<float>(CurrentStateMapH - 1) : 0.5f;
                    const float ly = (normY - 0.5f) * PlatformWorldSize.Y;
                    const float fy = (ly + HalfY) / SafeCellSize - 0.5f;
                    const int iy0 = FMath::Clamp((int32)FMath::FloorToFloat(fy), 0, GridSize - 1);
                    const int iy1 = FMath::Clamp(iy0 + 1, 0, GridSize - 1);

                    for (int32 ColIdx2 = 0; ColIdx2 < CurrentStateMapW; ++ColIdx2)
                    {
                        const float normX = (CurrentStateMapW > 1) ? static_cast<float>(ColIdx2) / static_cast<float>(CurrentStateMapW - 1) : 0.5f;
                        const float lx = (normX - 0.5f) * PlatformWorldSize.X;

                        float zLocal = visMinZ;

                        if (AsyncStatePtr && ColumnScratchCount > 0)
                        {
                            const float fx = (lx + HalfX) / SafeCellSize - 0.5f;
                            const int ix0 = FMath::Clamp((int32)FMath::FloorToFloat(fx), 0, GridSize - 1);
                            const int ix1 = FMath::Clamp(ix0 + 1, 0, GridSize - 1);
                            const int candIdx[4] = { iy0 * GridSize + ix0, iy1 * GridSize + ix0, iy0 * GridSize + ix1, iy1 * GridSize + ix1 };

                            for (int candidate = 0; candidate < 4; ++candidate)
                            {
                                const int ci = candIdx[candidate];
                                if (!AsyncStatePtr->ColumnCentersScratch.IsValidIndex(ci) || !AsyncStatePtr->ColumnRadiiScratch.IsValidIndex(ci))
                                {
                                    continue;
                                }

                                const FVector3f& cc = AsyncStatePtr->ColumnCentersScratch[ci];
                                const FVector3f& cr = AsyncStatePtr->ColumnRadiiScratch[ci];
                                const float rx = FMath::Max(cr.X, KINDA_SMALL_NUMBER);
                                const float ry = FMath::Max(cr.Y, KINDA_SMALL_NUMBER);
                                const float rz = FMath::Max(cr.Z, KINDA_SMALL_NUMBER);
                                const float dx = lx - cc.X;
                                const float dy = ly - cc.Y;
                                const float nx = dx / rx;
                                const float ny = dy / ry;
                                const float r2 = nx * nx + ny * ny;

                                if (r2 <= 1.0f)
                                {
                                    const float candidateZ = (cc.Z + ColumnZBias) + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - r2));
                                    zLocal = FMath::Max(zLocal, candidateZ);
                                }
                            }
                        }

                        if (AsyncStatePtr && BufferedObjects > 0)
                        {
                            for (int32 objIdx = 0; objIdx < BufferedObjects; ++objIdx)
                            {
                                if (!AsyncStatePtr->ObjCentersScratch.IsValidIndex(objIdx) || !AsyncStatePtr->ObjRadiiScratch.IsValidIndex(objIdx))
                                {
                                    continue;
                                }

                                const FVector3f& oc = AsyncStatePtr->ObjCentersScratch[objIdx];
                                const FVector3f& orad = AsyncStatePtr->ObjRadiiScratch[objIdx];
                                const float rx = FMath::Max(orad.X, KINDA_SMALL_NUMBER);
                                const float ry = FMath::Max(orad.Y, KINDA_SMALL_NUMBER);
                                const float rz = FMath::Max(orad.Z, KINDA_SMALL_NUMBER);
                                const float onx = (lx - oc.X) / rx;
                                const float ony = (ly - oc.Y) / ry;
                                const float or2 = onx * onx + ony * ony;

                                if (or2 <= 1.0f)
                                {
                                    const float candidateZ = (oc.Z + ObjectZBias) + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - or2));
                                    zLocal = FMath::Max(zLocal, candidateZ);
                                }
                            }
                        }

                        const float zClamped = FMath::Clamp(zLocal, visMinZ, visMaxZ);
                        const float normHeight = (zClamped - visMinZ) * InvRangeZ;
                        HeightTmp[RowIdx][ColIdx2] = FMath::Clamp(normHeight * 2.f - 1.f, -1.f, 1.f);
                    }
                }

                HeightTmp.Clip(-1.f, 1.f);
            }
        }
        else
        {
            // Original: pure line-trace height map across the full state grid
            const FTransform LocalGridTransform = Grid->GetActorTransform();
            const float LocalVisMinZ = MinZ;
            const float LocalVisMaxZ = MaxZ;
            const float LocalTraceDistUp = FMath::Abs(LocalVisMaxZ);
            const float LocalTraceDistDown = FMath::Abs(LocalVisMinZ);

            for (int32 TraceRowIdx = 0; TraceRowIdx < CurrentStateMapH; ++TraceRowIdx)
            {
                for (int32 TraceColIdx = 0; TraceColIdx < CurrentStateMapW; ++TraceColIdx)
                {
                    const float normX = (CurrentStateMapW > 1) ? static_cast<float>(TraceColIdx) / static_cast<float>(CurrentStateMapW - 1) : 0.5f;
                    const float normY = (CurrentStateMapH > 1) ? static_cast<float>(TraceRowIdx) / static_cast<float>(CurrentStateMapH - 1) : 0.5f;
                    const float lx = (normX - 0.5f) * PlatformWorldSize.X;
                    const float ly = (normY - 0.5f) * PlatformWorldSize.Y;

                    const FVector wStart = LocalGridTransform.TransformPosition(FVector(lx, ly, LocalTraceDistUp));
                    const FVector wEnd = LocalGridTransform.TransformPosition(FVector(lx, ly, -LocalTraceDistDown));

                    FHitResult hit;
                    const bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);

                    const float finalLocalZ = bHit ? LocalGridTransform.InverseTransformPosition(hit.ImpactPoint).Z : 0.f;
                    const float clampedVisZ = FMath::Clamp(finalLocalZ, LocalVisMinZ, LocalVisMaxZ);
                    const float normHeight = (LocalVisMaxZ > LocalVisMinZ) ? (clampedVisZ - LocalVisMinZ) / (LocalVisMaxZ - LocalVisMinZ) : 0.f;
                    HeightTmp[TraceRowIdx][TraceColIdx] = (normHeight * 2.f) - 1.f;
                }
            }
            HeightTmp.Clip(-1.f, 1.f);
        }
        
        PreviousHeight = CurrentHeight;
        CurrentHeight = HeightTmp;

        if (bIncludeOverheadImageInState && OverheadCaptureActor)
        {
            OverheadCaptureActor->GetCaptureComponent2D()->CaptureScene();
        }
        Step++;
    }
}

// ------------------------------------------
//   GetCentralState
// ------------------------------------------
TArray<float> UStateManager::GetCentralState()
{
    TArray<float> outArr;

    if (bIncludeHeightMapInState && CurrentHeight.Num() > 0)
    {
        outArr.Append(CurrentHeight.GetData());
    }

    if (bIncludeOverheadImageInState && OverheadRenderTarget && OverheadCaptureActor)
    {
        TArray<float> overhead = CaptureOverheadImage();
        if (overhead.Num() > 0) outArr.Append(overhead);
    }

    if (bIncludeGridObjectSequenceInState && MaxGridObjectsForState > 0 && GridObjectFeatureSize > 0)
    {
        const float HalfPlatformSizeX = (PlatformWorldSize.X > KINDA_SMALL_NUMBER) ? (PlatformWorldSize.X / 2.0f) : 1.0f;
        const float HalfPlatformSizeY = (PlatformWorldSize.Y > KINDA_SMALL_NUMBER) ? (PlatformWorldSize.Y / 2.0f) : 1.0f;
        const float ObjectZNormRange = MaxZ - MinZ;
        const float MaxVelocityClip = 100.0f;

        for (int32 i = 0; i < MaxGridObjectsForState; ++i)
        {
            // Include active objects and goal-reached objects for when bRemoveObjectsOnGoal is false
            if (GetObjectSlotState(i) == EObjectSlotState::Active || 
                (GetObjectSlotState(i) == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal))
            {
                FVector ObjPosLocal = GetCurrentPosition(i);
                int32 GoalIdx = GetGoalIndex(i);
                FVector GoalPosWorld = (GoalManager && GoalIdx != -1) ? GoalManager->GetGoalLocation(GoalIdx) : FVector::ZeroVector;
                FVector GoalPosLocal = Platform ? Platform->GetActorTransform().InverseTransformPosition(GoalPosWorld) : FVector::ZeroVector;
                FVector ObjVelLocal = GetCurrentVelocity(i);

                float NormObjPosX = FMath::Clamp(ObjPosLocal.X / HalfPlatformSizeX, -1.0f, 1.0f);
                float NormObjPosY = FMath::Clamp(ObjPosLocal.Y / HalfPlatformSizeY, -1.0f, 1.0f);
                float NormObjPosZ = (ObjectZNormRange > KINDA_SMALL_NUMBER) ? FMath::Clamp(((ObjPosLocal.Z - MinZ) / ObjectZNormRange) * 2.0f - 1.0f, -1.0f, 1.0f) : 0.0f;
                float NormGoalPosX = FMath::Clamp(GoalPosLocal.X / HalfPlatformSizeX, -1.0f, 1.0f);
                float NormGoalPosY = FMath::Clamp(GoalPosLocal.Y / HalfPlatformSizeY, -1.0f, 1.0f);
                float NormGoalPosZ = (ObjectZNormRange > KINDA_SMALL_NUMBER) ? FMath::Clamp(((GoalPosLocal.Z - MinZ) / ObjectZNormRange) * 2.0f - 1.0f, -1.0f, 1.0f) : 0.0f;
                float NormVelX = FMath::Clamp(ObjVelLocal.X, -MaxVelocityClip, MaxVelocityClip) / MaxVelocityClip;
                float NormVelY = FMath::Clamp(ObjVelLocal.Y, -MaxVelocityClip, MaxVelocityClip) / MaxVelocityClip;
                float NormVelZ = FMath::Clamp(ObjVelLocal.Z, -MaxVelocityClip, MaxVelocityClip) / MaxVelocityClip;

                outArr.Add(NormObjPosX); outArr.Add(NormObjPosY); outArr.Add(NormObjPosZ);
                outArr.Add(NormGoalPosX); outArr.Add(NormGoalPosY); outArr.Add(NormGoalPosZ);
                outArr.Add(NormVelX); outArr.Add(NormVelY); outArr.Add(NormVelZ);

                for (int32 pad_idx = 9; pad_idx < GridObjectFeatureSize; ++pad_idx)
                {
                    outArr.Add(0.0f);
                }
            }
            else
            {
                for (int32 feat_idx = 0; feat_idx < GridObjectFeatureSize; ++feat_idx)
                {
                    outArr.Add(0.0f);
                }
            }
        }
    }
    return outArr;
}

// ------------------------------------------
//   GetAgentState
// ------------------------------------------
TArray<float> UStateManager::GetAgentState(int32 AgentIndex) const
{
    return WaveSim ? WaveSim->GetAgentState(AgentIndex) : TArray<float>();
}

// ------------------------------------------
//   UpdateGridColumnsColors
// ------------------------------------------
void UStateManager::UpdateGridColumnsColors()
{
    if (!Grid || !OccupancyGrid) return;
    float mn = Grid->GetMinHeight(), mx = Grid->GetMaxHeight();
    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    {
        float h = Grid->GetColumnHeight(c);
        float ratio = (mx > mn) ? FMath::GetMappedRangeValueClamped(FVector2D(mn, mx), FVector2D(0.f, 1.f), h) : 0.5f;
        Grid->SetColumnColor(c, FLinearColor::LerpUsingHSV(FLinearColor::Black, FLinearColor::White, ratio));
    }
    /*if (GridObjectColors.Num() > 0) {
        FMatrix2D objOcc = OccupancyGrid->GetOccupancyMatrix({ FName("GridObjects") }, false);
        for (int32 r = 0; r < objOcc.GetNumRows(); ++r) {
            for (int32 col = 0; col < objOcc.GetNumColumns(); ++col) {
                if (objOcc[r][col] >= 0.f)
                {
                    int32 occupantId = FMath::RoundToInt(objOcc[r][col]);
                    Grid->SetColumnColor(r * GridSize + col, GridObjectColors[occupantId % GridObjectColors.Num()]);
                }
            }
        }
    }*/
    if (bUseRandomGoals && GoalColors.Num() > 0)
    {
        FMatrix2D goalOcc = OccupancyGrid->GetOccupancyMatrix({ FName("Goals") }, false);
        for (int32 r = 0; r < goalOcc.GetNumRows(); ++r) {
            for (int32 col = 0; col < goalOcc.GetNumColumns(); ++col) {
                if (goalOcc[r][col] >= 0.f)
                {
                    int32 occupantId = FMath::RoundToInt(goalOcc[r][col]);
                    Grid->SetColumnColor(r * GridSize + col, GoalColors[occupantId % GoalColors.Num()]);
                }
            }
        }
    }
}

// ------------------------------------------
//   Accessors
// ------------------------------------------
int32 UStateManager::GetMaxGridObjects() const { return MaxGridObjects; }
EObjectSlotState UStateManager::GetObjectSlotState(int32 ObjIndex) const { return ObjectSlotStates.IsValidIndex(ObjIndex) ? ObjectSlotStates[ObjIndex] : EObjectSlotState::Empty; }
bool UStateManager::GetHasActive(int32 ObjIndex) const { return GetObjectSlotState(ObjIndex) == EObjectSlotState::Active; }
bool UStateManager::GetHasReachedGoal(int32 ObjIndex) const { return GetObjectSlotState(ObjIndex) == EObjectSlotState::GoalReached; }
bool UStateManager::GetHasFallenOff(int32 ObjIndex) const { return GetObjectSlotState(ObjIndex) == EObjectSlotState::OutOfBounds; }
bool UStateManager::GetShouldCollectReward(int32 ObjIndex) const { return bShouldCollect.IsValidIndex(ObjIndex) ? bShouldCollect[ObjIndex] : false; }
void UStateManager::SetShouldCollectReward(int32 ObjIndex, bool bVal) { if (bShouldCollect.IsValidIndex(ObjIndex)) bShouldCollect[ObjIndex] = bVal; }
bool UStateManager::GetShouldRespawn(int32 ObjIndex) const { return bShouldResp.IsValidIndex(ObjIndex) ? bShouldResp[ObjIndex] : false; }
int32 UStateManager::GetGoalIndex(int32 ObjIndex)
{
    if (!ObjectGoalIndices.IsValidIndex(ObjIndex))
    {
        return -1;
    }

    int32 CurrentIndex = ObjectGoalIndices[ObjIndex];
    if (!GoalManager)
    {
        return CurrentIndex;
    }

    if (!GoalManager->IsValidGoalIndex(CurrentIndex))
    {
        const int32 NumGoals = GoalManager->GetNumGoals();
        if (NumGoals <= 0)
        {
            ObjectGoalIndices[ObjIndex] = -1;
            return -1;
        }

        CurrentIndex = ObjIndex % NumGoals;
        ObjectGoalIndices[ObjIndex] = CurrentIndex;
    }

    return CurrentIndex;
}
FVector UStateManager::GetCurrentVelocity(int32 ObjIndex) const { return CurrVel.IsValidIndex(ObjIndex) ? CurrVel[ObjIndex] : FVector::ZeroVector; }
FVector UStateManager::GetPreviousVelocity(int32 ObjIndex) const { return PrevVel.IsValidIndex(ObjIndex) ? PrevVel[ObjIndex] : FVector::ZeroVector; }
float UStateManager::GetCurrentDistance(int32 ObjIndex) const { return CurrDist.IsValidIndex(ObjIndex) ? CurrDist[ObjIndex] : -1.f; }
float UStateManager::GetPreviousDistance(int32 ObjIndex) const { return PrevDist.IsValidIndex(ObjIndex) ? PrevDist[ObjIndex] : -1.f; }

bool UStateManager::GetRemoveObjectsOnGoal() const { return bRemoveObjectsOnGoal; }
FVector UStateManager::GetCurrentPosition(int32 ObjIndex) const { return CurrPos.IsValidIndex(ObjIndex) ? CurrPos[ObjIndex] : FVector::ZeroVector; }
FVector UStateManager::GetPreviousPosition(int32 ObjIndex) const { return PrevPos.IsValidIndex(ObjIndex) ? PrevPos[ObjIndex] : FVector::ZeroVector; }

// ------------------------------------------
//   Helper: column top
// ------------------------------------------
FVector UStateManager::GetColumnTopWorldLocation(int32 GridX, int32 GridY) const
{
    if (!Grid) return PlatformCenter + FVector(0, 0, 100.f);
    int32 idx = GridX * GridSize + GridY;
    if (Grid->Columns.IsValidIndex(idx))
    {
        AColumn* col = Grid->Columns[idx];
        if (col && col->ColumnMesh)
        {
            FVector colRootLocation = col->GetActorLocation();
            // Use component world bounds; BoxExtent is already half-height in world units
            const float halfHeightWorld = col->ColumnMesh->Bounds.BoxExtent.Z;
            return FVector(colRootLocation.X, colRootLocation.Y, colRootLocation.Z + halfHeightWorld);
        }
    }
    return PlatformCenter + FVector(0, 0, 100.f);
}

// ------------------------------------------
//   SetupOverheadCamera
// ------------------------------------------
void UStateManager::SetupOverheadCamera()
{
    checkf(Platform, TEXT("SetupOverheadCamera => missing platform."));
    // Do not create the capture actor if overhead image is not part of the state
    if (!bIncludeOverheadImageInState) return;
    UWorld* w = Platform->GetWorld();
    if (!w || OverheadCaptureActor) return;

    FVector spawnLoc = PlatformCenter + FVector(0, 0, OverheadCamDistance);
    FRotator spawnRot = FRotator(-90.f, 0.f, 0.f);
    OverheadCaptureActor = w->SpawnActor<ASceneCapture2D>(spawnLoc, spawnRot, FActorSpawnParameters());

    if (!OverheadCaptureActor) return;

    int32 TargetResX = bIncludeOverheadImageInState && StateOverheadImageResX > 0 ? StateOverheadImageResX : OverheadCamResX;
    int32 TargetResY = bIncludeOverheadImageInState && StateOverheadImageResY > 0 ? StateOverheadImageResY : OverheadCamResY;

    OverheadRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    OverheadRenderTarget->RenderTargetFormat = RTF_RGBA8;
    OverheadRenderTarget->InitAutoFormat(TargetResX, TargetResY);
    OverheadRenderTarget->UpdateResourceImmediate(true);

    USceneCaptureComponent2D* comp = OverheadCaptureActor->GetCaptureComponent2D();
    comp->ProjectionType = ECameraProjectionMode::Perspective;
    comp->FOVAngle = OverheadCamFOV;
    comp->TextureTarget = OverheadRenderTarget;
    comp->CaptureSource = SCS_BaseColor;
    comp->bCaptureEveryFrame = false;
    comp->bCaptureOnMovement = false;
    comp->MaxViewDistanceOverride = OverheadCamDistance * 1.1f;
    comp->SetActive(true);
}

TArray<float> UStateManager::CaptureOverheadImage() const
{
    TArray<float> out;
    if (!OverheadCaptureActor || !OverheadRenderTarget) return out;
    UWorld* w = OverheadCaptureActor->GetWorld();
    if (!w) return out;

    TArray<FColor> pixels;
    if (!UKismetRenderingLibrary::ReadRenderTarget(w, OverheadRenderTarget, pixels) || pixels.Num() == 0) return out;

    out.Reserve(pixels.Num());
    for (const FColor& PixelColor : pixels)
    {
        float R01 = PixelColor.R / 255.0f;
        float G01 = PixelColor.G / 255.0f;
        float B01 = PixelColor.B / 255.0f;
        float luminance01 = 0.299f * R01 + 0.587f * G01 + 0.114f * B01;
        out.Add(luminance01 * 2.0f - 1.0f);
    }
    return out;
}

void UStateManager::ComputeColumnsInRadius(TSet<int32>& OutColumns) const
{
    OutColumns.Reset();
    if (!Grid || !ObjectMgr || !OccupancyGrid || GridSize <= 0) return;

    const int32 R = FMath::Max(0, ColumnCollisionRadiusCells);
    auto AddNeighbors = [&](int32 CenterIdx)
    {
        const int gx = CenterIdx / GridSize;
        const int gy = CenterIdx % GridSize;
        const int minX = FMath::Max(gx - R, 0);
        const int maxX = FMath::Min(gx + R, GridSize - 1);
        const int minY = FMath::Max(gy - R, 0);
        const int maxY = FMath::Min(gy + R, GridSize - 1);
        for (int xx = minX; xx <= maxX; ++xx)
        {
            for (int yy = minY; yy <= maxY; ++yy)
            {
                const float dx = static_cast<float>(xx - gx);
                const float dy = static_cast<float>(yy - gy);
                if (dx * dx + dy * dy <= static_cast<float>(R * R))
                {
                    OutColumns.Add(xx * GridSize + yy);
                }
            }
        }
    };

    // Collect neighbors around all visible grid objects (Active or GoalReached)
    for (int32 i = 0; i < ObjectSlotStates.Num(); ++i)
    {
        const EObjectSlotState SlotState = ObjectSlotStates[i];
        if (!(SlotState == EObjectSlotState::Active || SlotState == EObjectSlotState::GoalReached)) continue;
        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue;
        const FVector wPos = Obj->GetObjectLocation();
        const int32 cell = OccupancyGrid->WorldToGrid(wPos);
        AddNeighbors(cell);
    }
}

void UStateManager::UpdateColumnCollisionBasedOnOccupancy()
{
    if (!bRestrictColumnPhysicsToRadius) return;
    if (!Grid || !ObjectMgr || !OccupancyGrid || GridSize <= 0) return;

    TSet<int32> EnableCells;
    ComputeColumnsInRadius(EnableCells);

    // Enable/disable only where changed from previous frame
    TArray<int32> ToEnable;
    TArray<int32> ToDisable;
    ToEnable.Reserve(EnableCells.Num());
    ToDisable.Reserve(PrevEnabledColumnCells.Num());

    for (int32 idx : EnableCells)
    {
        if (!PrevEnabledColumnCells.Contains(idx))
        {
            ToEnable.Add(idx);
        }
    }
    for (int32 idx : PrevEnabledColumnCells)
    {
        if (!EnableCells.Contains(idx))
        {
            ToDisable.Add(idx);
        }
    }

    for (int32 idx : ToEnable) { Grid->SetColumnCollision(idx, true); }
    for (int32 idx : ToDisable) { Grid->SetColumnCollision(idx, false); }

    // Update cache
    PrevEnabledColumnCells = EnableCells;
}
