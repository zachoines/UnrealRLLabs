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
    ObjectScale = Config->GetOrDefaultNumber(TEXT("ObjectScale"), ObjectScale);
    ObjectMass = Config->GetOrDefaultNumber(TEXT("ObjectMass"), ObjectMass);
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

    GoalRadius = Config->GetOrDefaultNumber(TEXT("GoalRadius"), GoalRadius);
    GoalCollectRadius = Config->GetOrDefaultNumber(TEXT("GoalCollectRadius"), GoalCollectRadius);
    ObjectRadius = Config->GetOrDefaultNumber(TEXT("ObjectRadius"), ObjectRadius);
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
    bEnableColumnCollisionOptimization = Config->GetOrDefaultBool(TEXT("bEnableColumnCollisionOptimization"), false);
    ColumnCollisionRadiusCells = Config->GetOrDefaultInt(TEXT("ColumnCollisionRadiusCells"), 2);

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
    // Clear previous enabled column cache
    PrevEnabledColumnCells.Empty();

    // 4) Clear occupancy
    OccupancyGrid->ResetGrid();

    // 5) Place random goals and gather them for GoalManager
    TArray<AActor*>  newGoalActors;
    TArray<FVector>  newGoalOffsets;

    if (bUseRandomGoals)
    {
        const int32 nGoals = GoalColors.Num();
        for (int32 g = 0; g < nGoals; g++)
        {
            float radiusCells = GoalRadius / CellSize;
            int32 placedCell = OccupancyGrid->AddObjectToGrid(g, FName("Goals"), radiusCells, TArray<FName>());
            if (placedCell < 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("Could not place random goal %d => skipping."), g);
                continue;
            }

            int32 gx = placedCell / GridSize;
            int32 gy = placedCell % GridSize;
            int32 colIndex = gx * GridSize + gy;
            if (!Grid->Columns.IsValidIndex(colIndex)) continue;

            AColumn* col = Grid->Columns[colIndex];
            if (!col || !col->ColumnMesh) continue;

            float full = col->ColumnMesh->Bounds.BoxExtent.Z * 2.0;
            float objWorldRadius = ObjectUnscaledSize * ObjectScale;
            FVector offset(0.f, 0.f, full + objWorldRadius);

            newGoalActors.Add(col);
            newGoalOffsets.Add(offset);
        }
    }

    GoalManager->ResetGoals(newGoalActors, newGoalOffsets);

    // 6) Round-robin "goal index" for each object
    int32 numGoals = GoalColors.Num();
    for (int32 i = 0; i < NumObjects; i++)
    {
        ObjectGoalIndices[i] = (numGoals > 0) ? (i % numGoals) : -1;
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
        int32 gIdx = ObjectGoalIndices[i];
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
                // If already GoalReached, Keep collecting reward
                else if (ObjectSlotStates[i] == EObjectSlotState::GoalReached && !bRemoveObjectsOnGoal)
                {
                    bShouldCollect[i] = true;
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
        int32 gIdx = ObjectGoalIndices[i];
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
            float radiusCells = ObjectRadius / CellSize;
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

            float radiusCells = (CellSize > 0) ? (ObjectRadius / CellSize) : 1.0f;
            int32 cellIdx = OccupancyGrid ? OccupancyGrid->AddObjectToGrid(i, FName("GridObjects"), radiusCells, TArray<FName>{}) : -1;

            if (cellIdx < 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("No free occupant cell for obj %d, will try again."), i);
                continue;
            }

            int32 gx = cellIdx / GridSize;
            int32 gy = cellIdx % GridSize;
            FVector spawnLoc = GetColumnTopWorldLocation(gx, gy);

            if (ObjectMgr) ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(ObjectScale), ObjectMass);

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
        const float traceDistUp = FMath::Abs(visMaxZ);
        const float traceDistDown = FMath::Abs(visMinZ);

        // If hybrid mode is enabled, start from the simulator wave and then overwrite with traces in ROI.
        if (bEnableColumnCollisionOptimization && WaveSim)
        {
            const FMatrix2D& Wave = WaveSim->GetHeightMap();
            const int32 WaveH = Wave.GetNumRows();
            const int32 WaveW = Wave.GetNumColumns();

            // Baseline: resample the wave into HeightTmp
            for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state)
            {
                for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state)
                {
                    // Map state pixel to wave indices (bilinear sample)
                    const float u = (CurrentStateMapW > 1) ? (float)c_state / (float)(CurrentStateMapW - 1) : 0.5f;
                    const float v = (CurrentStateMapH > 1) ? (float)r_state / (float)(CurrentStateMapH - 1) : 0.5f;
                    const float wf = u * (WaveW - 1);
                    const float hf = v * (WaveH - 1);
                    const int32 x0 = FMath::Clamp((int32)FMath::FloorToFloat(hf), 0, WaveH - 1);
                    const int32 x1 = FMath::Clamp(x0 + 1, 0, WaveH - 1);
                    const int32 y0 = FMath::Clamp((int32)FMath::FloorToFloat(wf), 0, WaveW - 1);
                    const int32 y1 = FMath::Clamp(y0 + 1, 0, WaveW - 1);
                    const float dx = hf - (float)x0;
                    const float dy = wf - (float)y0;

                    const float v00 = Wave[x0][y0];
                    const float v01 = Wave[x0][y1];
                    const float v10 = Wave[x1][y0];
                    const float v11 = Wave[x1][y1];
                    const float v0 = FMath::Lerp(v00, v01, dy);
                    const float v1 = FMath::Lerp(v10, v11, dy);
                    const float vInterp = FMath::Lerp(v0, v1, dx);

                    const float clampedVisZ = FMath::Clamp(vInterp, visMinZ, visMaxZ);
                    const float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                    HeightTmp[r_state][c_state] = (norm_height * 2.f) - 1.f;
                }
            }

            // Analytical overlay for column curvature across non-traced positions
            // Compute ellipsoid cap height per pixel within column XY footprint and overlay on baseline
            for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state)
            {
                for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state)
                {
                    const float norm_x = (CurrentStateMapW > 1) ? (float)c_state / (float)(CurrentStateMapW - 1) : 0.5f;
                    const float norm_y = (CurrentStateMapH > 1) ? (float)r_state / (float)(CurrentStateMapH - 1) : 0.5f;
                    const float lx = (norm_x - 0.5f) * PlatformWorldSize.X;
                    const float ly = (norm_y - 0.5f) * PlatformWorldSize.Y;

                    const FVector wStart = GridTransform.TransformPosition(FVector(lx, ly, traceDistUp));
                    const int32 gridCell = OccupancyGrid ? OccupancyGrid->WorldToGrid(wStart) : -1;
                    if (!Grid || !Grid->Columns.IsValidIndex(gridCell))
                    {
                        continue;
                    }
                    AColumn* Col = Grid->Columns[gridCell];
                    if (!Col || !Col->ColumnMesh) continue;

                    // Column center in local frame
                    const FVector colLocal = GridTransform.InverseTransformPosition(Col->GetActorLocation());
                    const float dx = lx - colLocal.X;
                    const float dy = ly - colLocal.Y;

                    // Radii from mesh local bounds and relative scale
                    const FBoxSphereBounds ColumnBounds = Col->ColumnMesh->CalcLocalBounds();
                    const FVector colScale = Col->ColumnMesh->GetRelativeScale3D();
                    const float rx = ColumnBounds.BoxExtent.X * colScale.X;
                    const float ry = ColumnBounds.BoxExtent.Y * colScale.Y;
                    const float rz = ColumnBounds.BoxExtent.Z * colScale.Z;
                    if (rx <= KINDA_SMALL_NUMBER || ry <= KINDA_SMALL_NUMBER || rz <= KINDA_SMALL_NUMBER)
                    {
                        continue;
                    }
                    const float nx = dx / rx;
                    const float ny = dy / ry;
                    const float r2 = nx * nx + ny * ny;
                    if (r2 > 1.0f) continue; // outside ellipsoid XY projection

                    const float zLocalTop = colLocal.Z + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - r2));
                    const float clampedVisZ = FMath::Clamp(zLocalTop, visMinZ, visMaxZ);
                    const float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                    const float h01 = (norm_height * 2.f) - 1.f;
                    if (h01 > HeightTmp[r_state][c_state])
                    {
                        HeightTmp[r_state][c_state] = h01;
                    }
                }
            }

            // Build ROI of grid cells near grid objects for tracing
            TSet<int32> TraceCells;
            const int32 R = FMath::Max(0, ColumnCollisionRadiusCells);
            if (ObjectMgr && OccupancyGrid && R > 0)
            {
                const int32 TotalCols = GridSize * GridSize;
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
                            const float dx = (float)(xx - gx);
                            const float dy = (float)(yy - gy);
                            if (dx * dx + dy * dy <= (float)(R * R))
                            {
                                TraceCells.Add(xx * GridSize + yy);
                            }
                        }
                    }
                };

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

            // Overwrite pixels that fall within TraceCells with line-trace heights
            for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state)
            {
                for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state)
                {
                    // Map state pixel to world XY
                    const float norm_x = (CurrentStateMapW > 1) ? (float)c_state / (float)(CurrentStateMapW - 1) : 0.5f;
                    const float norm_y = (CurrentStateMapH > 1) ? (float)r_state / (float)(CurrentStateMapH - 1) : 0.5f;
                    const float lx = (norm_x - 0.5f) * PlatformWorldSize.X;
                    const float ly = (norm_y - 0.5f) * PlatformWorldSize.Y;

                    const FVector wStart = GridTransform.TransformPosition(FVector(lx, ly, traceDistUp));
                    const FVector wEnd = GridTransform.TransformPosition(FVector(lx, ly, -traceDistDown));

                    // Determine column cell under this pixel
                    const int32 gridCell = OccupancyGrid ? OccupancyGrid->WorldToGrid(wStart) : -1;
                    if (TraceCells.Contains(gridCell))
                    {
                        FHitResult hit;
                        bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);
                        float finalLocalZ;
                        if (bHit)
                        {
                            finalLocalZ = GridTransform.InverseTransformPosition(hit.ImpactPoint).Z;
                        }
                        else
                        {
                            // Fallback: synthesize exact ellipsoid cap height for the column under this pixel
                            float zLocalSynth = 0.f;
                            bool bSynthValid = false;
                            if (Grid && Grid->Columns.IsValidIndex(gridCell))
                            {
                                if (AColumn* Col = Grid->Columns[gridCell])
                                {
                                    if (Col->ColumnMesh)
                                    {
                                        const FVector colLocal = GridTransform.InverseTransformPosition(Col->GetActorLocation());
                                        const float dx = lx - colLocal.X;
                                        const float dy = ly - colLocal.Y;
                                        const FBoxSphereBounds ColumnBounds = Col->ColumnMesh->CalcLocalBounds();
                                        const FVector colScale = Col->ColumnMesh->GetRelativeScale3D();
                                        const float rx = ColumnBounds.BoxExtent.X * colScale.X;
                                        const float ry = ColumnBounds.BoxExtent.Y * colScale.Y;
                                        const float rz = ColumnBounds.BoxExtent.Z * colScale.Z;
                                        if (rx > KINDA_SMALL_NUMBER && ry > KINDA_SMALL_NUMBER && rz > KINDA_SMALL_NUMBER)
                                        {
                                            const float nx = dx / rx;
                                            const float ny = dy / ry;
                                            const float r2 = nx * nx + ny * ny;
                                            if (r2 <= 1.0f)
                                            {
                                                zLocalSynth = colLocal.Z + rz * FMath::Sqrt(FMath::Max(0.f, 1.0f - r2));
                                                bSynthValid = true;
                                            }
                                        }
                                    }
                                }
                            }
                            if (bSynthValid)
                            {
                                finalLocalZ = zLocalSynth;
                            }
                            else
                            {
                                // Ultimate fallback: top center point
                                const int gx = FMath::Clamp(gridCell / GridSize, 0, GridSize - 1);
                                const int gy = FMath::Clamp(gridCell % GridSize, 0, GridSize - 1);
                                const FVector topWorld = GetColumnTopWorldLocation(gx, gy);
                                finalLocalZ = GridTransform.InverseTransformPosition(topWorld).Z;
                            }
                        }
                        const float clampedVisZ = FMath::Clamp(finalLocalZ, visMinZ, visMaxZ);
                        const float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                        HeightTmp[r_state][c_state] = (norm_height * 2.f) - 1.f;
                    }
                }
            }
            HeightTmp.Clip(-1.f, 1.f);
        }
        else
        {
            // Original: pure line-trace height map across the full state grid
            for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state)
            {
                for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state)
                {
                    const float norm_x = (CurrentStateMapW > 1) ? static_cast<float>(c_state) / (CurrentStateMapW - 1) : 0.5f;
                    const float norm_y = (CurrentStateMapH > 1) ? static_cast<float>(r_state) / (CurrentStateMapH - 1) : 0.5f;
                    const float lx = (norm_x - 0.5f) * PlatformWorldSize.X;
                    const float ly = (norm_y - 0.5f) * PlatformWorldSize.Y;

                    const FVector wStart = GridTransform.TransformPosition(FVector(lx, ly, traceDistUp));
                    const FVector wEnd = GridTransform.TransformPosition(FVector(lx, ly, -traceDistDown));

                    FHitResult hit;
                    const bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);

                    const float finalLocalZ = bHit ? GridTransform.InverseTransformPosition(hit.ImpactPoint).Z : 0.f;
                    const float clampedVisZ = FMath::Clamp(finalLocalZ, visMinZ, visMaxZ);
                    const float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                    HeightTmp[r_state][c_state] = (norm_height * 2.f) - 1.f;
                }
            }
            HeightTmp.Clip(-1.f, 1.f);
        }
    }
    PreviousHeight = CurrentHeight;
    CurrentHeight = HeightTmp;

    if (bIncludeOverheadImageInState && OverheadCaptureActor)
    {
        OverheadCaptureActor->GetCaptureComponent2D()->CaptureScene();
    }
    Step++;
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
    if (GridObjectColors.Num() > 0) {
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
    }
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
int32 UStateManager::GetGoalIndex(int32 ObjIndex) const { return ObjectGoalIndices.IsValidIndex(ObjIndex) ? ObjectGoalIndices[ObjIndex] : -1; }
FVector UStateManager::GetCurrentVelocity(int32 ObjIndex) const { return CurrVel.IsValidIndex(ObjIndex) ? CurrVel[ObjIndex] : FVector::ZeroVector; }
FVector UStateManager::GetPreviousVelocity(int32 ObjIndex) const { return PrevVel.IsValidIndex(ObjIndex) ? PrevVel[ObjIndex] : FVector::ZeroVector; }
float UStateManager::GetCurrentDistance(int32 ObjIndex) const { return CurrDist.IsValidIndex(ObjIndex) ? CurrDist[ObjIndex] : -1.f; }
float UStateManager::GetPreviousDistance(int32 ObjIndex) const { return PrevDist.IsValidIndex(ObjIndex) ? PrevDist[ObjIndex] : -1.f; }
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
            float scaledHalfMeshHeight = col->ColumnMesh->Bounds.BoxExtent.Z * col->ColumnMesh->GetComponentScale().Z;
            return FVector(colRootLocation.X, colRootLocation.Y, colRootLocation.Z + (scaledHalfMeshHeight * 2.0));
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

void UStateManager::UpdateColumnCollisionBasedOnOccupancy()
{
    if (!bEnableColumnCollisionOptimization) return;
    if (!Grid || !ObjectMgr || !OccupancyGrid || GridSize <= 0) return;

    TSet<int32> EnableCells;
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
                const float dx = (float)(xx - gx);
                const float dy = (float)(yy - gy);
                if (dx * dx + dy * dy <= (float)(R * R))
                {
                    EnableCells.Add(xx * GridSize + yy);
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
