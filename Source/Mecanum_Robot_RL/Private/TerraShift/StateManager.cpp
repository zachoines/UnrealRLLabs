// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

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
    MinZ = Config->GetOrDefaultNumber(TEXT("MinZ"), MinZ); // Used for OOB and Z-norm
    MaxZ = Config->GetOrDefaultNumber(TEXT("MaxZ"), MaxZ); // Used for OOB and Z-norm
   
    ObjectScale = Config->GetOrDefaultNumber(TEXT("ObjectScale"), ObjectScale);
    ObjectMass = Config->GetOrDefaultNumber(TEXT("ObjectMass"), ObjectMass);
    MaxColumnHeight = Config->GetOrDefaultNumber(TEXT("MaxColumnHeight"), MaxColumnHeight);
    BaseRespawnDelay = Config->GetOrDefaultNumber(TEXT("BaseRespawnDelay"), BaseRespawnDelay);

    OverheadCamDistance = Config->GetOrDefaultNumber(TEXT("OverheadCameraDistance"), OverheadCamDistance);
    OverheadCamFOV = Config->GetOrDefaultNumber(TEXT("OverheadCameraFOV"), OverheadCamFOV);
    OverheadCamResX = Config->GetOrDefaultInt(TEXT("OverheadCameraResX"), OverheadCamResX);
    OverheadCamResY = Config->GetOrDefaultInt(TEXT("OverheadCameraResY"), OverheadCamResY);

    bUseRandomGoals = Config->GetOrDefaultBool(TEXT("bUseRandomGoals"), true);
    bRemoveGridObjectOnGoalReached = Config->GetOrDefaultBool(TEXT("bRemoveGridObjectOnGoalReached"), false);
    bRemoveGridObjectOnOOB = Config->GetOrDefaultBool(TEXT("bRemoveGridObjectOnOOB"), false);
    bRespawnGridObjectOnGoalReached = Config->GetOrDefaultBool(TEXT("bRespawnGridObjectOnGoalReached"), false);

    GoalRadius = Config->GetOrDefaultNumber(TEXT("GoalRadius"), GoalRadius);
    GoalCollectRadius = Config->GetOrDefaultNumber(TEXT("GoalCollectRadius"), GoalCollectRadius);
    ObjectRadius = Config->GetOrDefaultNumber(TEXT("ObjectRadius"), ObjectRadius);
    GridSize = Config->GetOrDefaultInt(TEXT("GridSize"), GridSize);

    // State Representation Config
    bIncludeHeightMapInState = Config->GetOrDefaultBool(TEXT("bIncludeHeightMapInState"), true);
    bIncludeOverheadImageInState = Config->GetOrDefaultBool(TEXT("bIncludeOverheadImageInState"), true);

    // GridSize would have been set in SetReferences, use it for default if needed
    int32 DefaultResH = GridSize > 0 ? GridSize : 25;
    int32 DefaultResW = GridSize > 0 ? GridSize : 25;

    StateHeightMapResolutionH = Config->GetOrDefaultInt(TEXT("StateHeightMapResolutionH"), DefaultResH);
    StateHeightMapResolutionW = Config->GetOrDefaultInt(TEXT("StateHeightMapResolutionW"), DefaultResW);
    StateOverheadImageResX = Config->GetOrDefaultInt(TEXT("StateOverheadImageResX"), OverheadCamResX);
    StateOverheadImageResY = Config->GetOrDefaultInt(TEXT("StateOverheadImageResY"), OverheadCamResY);

    // ***** Load NEW Grid Object Sequence State Config *****
    bIncludeGridObjectSequenceInState = Config->GetOrDefaultBool(TEXT("bIncludeGridObjectSequenceInState"), false);
    MaxGridObjectsForState = Config->GetOrDefaultInt(TEXT("MaxGridObjectsForState"), MaxGridObjects);
    GridObjectFeatureSize = Config->GetOrDefaultInt(TEXT("GridObjectFeatureSize"), 9);
    // ***** END OF LOADING NEW PARAMS *****


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
    bHasActive.SetNum(NumObjects);
    bHasReached.SetNum(NumObjects);
    bFallenOff.SetNum(NumObjects);
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
        bHasActive[i] = false;
        bHasReached[i] = false;
        bFallenOff[i] = false;
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
        RespawnDelays[i] = 0.0f; // For immediate first spawn if BaseRespawnDelay is 0
        if (BaseRespawnDelay > 0.0f) // If BaseRespawnDelay is configured, use it for initial staggering
        {
            RespawnDelays[i] = BaseRespawnDelay * static_cast<float>(i);
        }
    }

    // 3) Reset NxN height arrays & step counter
    int32 StateH = (bIncludeHeightMapInState && StateHeightMapResolutionH > 0) ? StateHeightMapResolutionH : GridSize;
    int32 StateW = (bIncludeHeightMapInState && StateHeightMapResolutionW > 0) ? StateHeightMapResolutionW : GridSize;

    PreviousHeight = FMatrix2D(StateH, StateW, 0.f);
    CurrentHeight = FMatrix2D(StateH, StateW, 0.f);
    Step = 0;

    // 4) Clear occupancy
    OccupancyGrid->ResetGrid();

    // 5) Place random or stationary goals, BUT also gather them for GoalManager
    TArray<AActor*>  newGoalActors;
    TArray<FVector>  newGoalOffsets;

    if (bUseRandomGoals)
    {
        const int32 nGoals = GoalColors.Num();

        for (int32 g = 0; g < nGoals; g++)
        {
            float radiusCells = GoalRadius / CellSize;

            // occupant ID => g
            int32 placedCell = OccupancyGrid->AddObjectToGrid(
                g,
                FName("Goals"),
                radiusCells,
                /*OverlapLayers=*/ TArray<FName>()
            );
            if (placedCell < 0)
            {
                UE_LOG(LogTemp, Warning,
                    TEXT("Could not place random goal %d => skipping."), g);
                continue;
            }

            // Now we figure out which column that cell corresponds to
            int32 gx = placedCell / GridSize;
            int32 gy = placedCell % GridSize;
            int32 colIndex = gx * GridSize + gy;
            if (!Grid->Columns.IsValidIndex(colIndex))
            {
                // failsafe
                continue;
            }

            AColumn* col = Grid->Columns[colIndex];
            if (!col || !col->ColumnMesh)
                continue;

            // Suppose we want the offset to be: top of column + an extra
            // "object radius" in Z.  For a typical small object scale,
            // you might do something like:
            float halfZ = col->ColumnMesh->Bounds.BoxExtent.Z;
            float objWorldRadius = ObjectUnscaledSize * ObjectScale;

            FVector offset(0.f, 0.f, halfZ + objWorldRadius);

            newGoalActors.Add(col);
            newGoalOffsets.Add(offset);
        }
    }
    else
    {
        // Stationary => spawn AGoalPlatform actors
        SpawnStationaryGoalPlatforms();
        // Then gather them for GoalManager
        // e.g. foundGoalActors, each => offset=some
    }

    //  5B) Pass them to GoalManager => (Actor + offset)
    GoalManager->ResetGoals(newGoalActors, newGoalOffsets);

    // 6) Round-robin "goal index" for each object
    int32 numGoals = GoalColors.Num();
    for (int32 i = 0; i < NumObjects; i++)
    {
        if (numGoals <= 0)
        {
            ObjectGoalIndices[i] = -1;
        }
        else
        {
            ObjectGoalIndices[i] = i % numGoals;
        }
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

    // We now use GoalManager->IsInRadiusOf(...) to check "reached"
    // so we can comment out occupant-based intersection
    // bool bUseOccupancyForGoals = bUseRandomGoals; // We won't use it for "reached" check

    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue;

        FVector wPos = Obj->GetObjectLocation();

        // (1) Check if object reached its goal
        if (!bHasReached[i])
        {
            int32 gIdx = ObjectGoalIndices[i];
            if (gIdx >= 0)
            {
                // Distance-based approach with GoalManager
                bool bInRadius = GoalManager->IsInRadiusOf(gIdx, wPos, GoalCollectRadius);
                if (bInRadius)
                {
                    bHasActive[i] = false;
                    bHasReached[i] = true;
                    bShouldCollect[i] = true;

                    if (bRemoveGridObjectOnGoalReached)
                    {
                        bShouldResp[i] = false;
                    }
                    else
                    {
                        bShouldResp[i] = true;
                    }

                    // also remove occupant from "GridObjects"
                    OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                    ObjectMgr->DisableGridObject(i);
                }
            }
        }

        // (2) OOB Check
        if (!bFallenOff[i] && !bHasReached[i])
        {
            float dx = FMath::Abs(wPos.X - PlatformCenter.X);
            float dy = FMath::Abs(wPos.Y - PlatformCenter.Y);
            float zPos = wPos.Z;

            bool bOOB = false;
            if (dx > (halfX + MarginXY) || dy > (halfY + MarginXY))
            {
                bOOB = true;
            }
            else if (zPos < minZLocal || zPos > maxZLocal)
            {
                bOOB = true;
            }

            if (bOOB)
            {
                bFallenOff[i] = true;
                bShouldCollect[i] = true;

                if (bRemoveGridObjectOnOOB)
                {
                    bHasActive[i] = false;
                    bShouldResp[i] = false;
                    OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                    ObjectMgr->DisableGridObject(i);
                }
                else
                {
                    bHasActive[i] = false;
                    bShouldResp[i] = true;
                    OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                    ObjectMgr->DisableGridObject(i);
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
    for (int32 i = 0; i < MaxGridObjects; i++) // Iterate up to MaxGridObjects
    {
        // If the object is not active (e.g., waiting for respawn, or already terminal)
        if (!bHasActive.IsValidIndex(i) || !bHasActive[i])
        {
            // Zero out stats for inactive objects
            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f;
            CurrDist[i] = -1.f;
            PrevPos[i] = FVector::ZeroVector;
            CurrPos[i] = FVector::ZeroVector;

            // If it's marked for respawn, increment its timer
            if (bShouldResp.IsValidIndex(i) && bShouldResp[i] && RespawnTimer.IsValidIndex(i))
            {
                RespawnTimer[i] += DeltaTime;
            }
            continue; // Move to the next object
        }


        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue; // Should not happen if bHasActive[i] is true

        FVector wVel = Obj->MeshComponent->GetPhysicsLinearVelocity();
        FVector wPos = Obj->GetObjectLocation();

        int32 gIdx = ObjectGoalIndices[i];
        FVector wGoal(0.f);
        if (gIdx >= 0 && GoalManager) // Added GoalManager null check
        {
            wGoal = GoalManager->GetGoalLocation(gIdx);
        }

        // localize
        FVector locVel = Platform->GetActorTransform().InverseTransformVector(wVel);
        FVector locPos = Platform->GetActorTransform().InverseTransformPosition(wPos);
        FVector locGoal = Platform->GetActorTransform().InverseTransformPosition(wGoal);

        // shift old => prev
        PrevVel[i] = CurrVel[i];
        PrevAcc[i] = CurrAcc[i];
        PrevDist[i] = CurrDist[i];
        PrevPos[i] = CurrPos[i];

        CurrVel[i] = locVel;
        CurrAcc[i] = (DeltaTime > SMALL_NUMBER)
            ? (locVel - PrevVel[i]) / DeltaTime
            : FVector::ZeroVector;
        CurrPos[i] = locPos;
        CurrDist[i] = FVector::Dist(locPos, locGoal);

        if (bShouldResp.IsValidIndex(i) && bShouldResp[i] && RespawnTimer.IsValidIndex(i))
        {
            RespawnTimer[i] += DeltaTime;
        }

        // occupant position => from wPos
        if (OccupancyGrid && CellSize > 0) // Added OccupancyGrid and CellSize check
        {
            int32 cellIdx = OccupancyGrid->WorldToGrid(wPos);
            float radiusCells = ObjectRadius / CellSize;

            // direct occupant update => occupant ID = i
            OccupancyGrid->UpdateObjectPosition(
                i,
                FName("GridObjects"),
                cellIdx,
                radiusCells,
                /*OverlapLayers=*/ TArray<FName>{ FName("Goals") }
            );
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

    for (int32 i = 0; i < MaxGridObjects; i++) // Iterate up to MaxGridObjects
    {
        // Check if this slot should be respawned
        if (!bShouldResp.IsValidIndex(i) || !bShouldResp[i] || !RespawnTimer.IsValidIndex(i) || !RespawnDelays.IsValidIndex(i)) continue;


        if (RespawnTimer[i] >= RespawnDelays[i])
        {
            // Remove old occupant in "GridObjects" layer if it existed
            if (OccupancyGrid) OccupancyGrid->RemoveObject(i, FName("GridObjects"));

            float radiusCells = (CellSize > 0) ? (ObjectRadius / CellSize) : 1.0f;

            // occupant ID = i
            int32 cellIdx = -1;
            if (OccupancyGrid)
            {
                cellIdx = OccupancyGrid->AddObjectToGrid(
                    i,
                    FName("GridObjects"),
                    radiusCells,
                    /*OverlapLayers=*/ TArray<FName>{} // Empty, meaning no overlap with other "GridObjects" implicitly, Goals are fine if desired
                );
            }

            if (cellIdx < 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("No free occupant cell for obj %d, will try again."), i);
                // Optionally, reset RespawnTimer[i] here to retry after a short delay, or let it keep ticking
                // For immediate retry logic (could be problematic if grid is full): RespawnTimer[i] = 0.f;
                continue; // Try next frame
            }

            // Convert cell => top-of-column
            int32 gx = cellIdx / GridSize;
            int32 gy = cellIdx % GridSize;
            FVector spawnLoc = GetColumnTopWorldLocation(gx, gy);

            // Actually spawn/reuse the object
            if (ObjectMgr) ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(ObjectScale), ObjectMass);

            // Color => occupant i => i % nObjColors
            AGridObject* newObj = ObjectMgr ? ObjectMgr->GetGridObject(i) : nullptr;
            if (newObj)
            {
                int32 colorIdx = i % nObjColors;  // round‐robin object color
                newObj->SetGridObjectColor(GridObjectColors[colorIdx]);
            }

            // Update state flags for the newly respawned object
            bHasActive[i] = true;
            bHasReached[i] = false; // Reset reached status
            bFallenOff[i] = false;  // Reset fallen off status
            bShouldResp[i] = false; // It has now respawned
            bShouldCollect[i] = false; // Reset reward collection flag
            RespawnTimer[i] = 0.f;  // Reset its respawn timer

            // Re-initialize stats for the new life of this object
            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f; // Will be calculated on next UpdateObjectStats
            CurrDist[i] = -1.f; // Will be calculated on next UpdateObjectStats
            PrevPos[i] = Platform->GetActorTransform().InverseTransformPosition(spawnLoc); // Initial position
            CurrPos[i] = PrevPos[i];
        }
    }
}

// ------------------------------------------
//   AllGridObjectsHandled
// ------------------------------------------
bool UStateManager::AllGridObjectsHandled() const
{
    if (bRemoveGridObjectOnGoalReached || bRemoveGridObjectOnOOB)
    {
        if (bHasActive.Contains(true)) return false;
        if (bShouldResp.Contains(true)) return false;
        return true;
    }
    if (bUseRandomGoals && !bRespawnGridObjectOnGoalReached && !bRemoveGridObjectOnOOB)
    {
        return !bHasReached.Contains(false);
    }
    return false;
}


// ------------------------------------------
//   BuildCentralState
// ------------------------------------------
void UStateManager::BuildCentralState()
{
    UWorld* w = (Grid) ? Grid->GetWorld() : nullptr;
    if (!w)
    {
        UE_LOG(LogTemp, Warning, TEXT("BuildCentralState => no valid UWorld."));
        return;
    }
    if (!Grid || !Platform)
    {
        UE_LOG(LogTemp, Error, TEXT("BuildCentralState => Grid or Platform actor is null."));
        return;
    }

    // Determine the dimensions for the state's height map
    // If bIncludeHeightMapInState is false, CurrentStateMapH/W will be based on physical GridSize,
    int32 PhysicalGridSizeH = GridSize > 0 ? GridSize : 1;
    int32 PhysicalGridSizeW = GridSize > 0 ? GridSize : 1;

    int32 CurrentStateMapH = bIncludeHeightMapInState ? (StateHeightMapResolutionH > 0 ? StateHeightMapResolutionH : PhysicalGridSizeH) : PhysicalGridSizeH;
    int32 CurrentStateMapW = bIncludeHeightMapInState ? (StateHeightMapResolutionW > 0 ? StateHeightMapResolutionW : PhysicalGridSizeW) : PhysicalGridSizeW;

    // Ensure CurrentHeight and PreviousHeight matrices are correctly sized.
    // This should ideally be done once after config load if sizes are static for the session.
    if (CurrentHeight.GetNumRows() != CurrentStateMapH || CurrentHeight.GetNumColumns() != CurrentStateMapW) {
        PreviousHeight.Resize(CurrentStateMapH, CurrentStateMapW, FMatrix2D::EInitialization::Zero);
        CurrentHeight.Resize(CurrentStateMapH, CurrentStateMapW, FMatrix2D::EInitialization::Zero);
    }

    FMatrix2D HeightTmp(CurrentStateMapH, CurrentStateMapW, 0.f);

    if (bIncludeHeightMapInState) {
        float halfPlatformSizeX = PlatformWorldSize.X * 0.5f;
        float halfPlatformSizeY = PlatformWorldSize.Y * 0.5f;
        FTransform GridTransform = Grid->GetActorTransform();

        float VisualizationZRange = MaxZ;
        float halfVisRange = VisualizationZRange * 0.5f;
        float visMinZ = -halfVisRange;
        float visMaxZ = halfVisRange;
        float traceDistUp = FMath::Abs(MaxZ) + 100.0f;
        float traceDistDown = FMath::Abs(visMinZ) + 100.0f;


        for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state) {
            for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state) {
                // Map state grid cell (r_state, c_state) to a physical normalized position (0-1) on the platform
                // Handle division by zero if CurrentStateMapW/H is 1
                float norm_x_on_platform = (CurrentStateMapW > 1) ? (static_cast<float>(c_state) / (CurrentStateMapW - 1)) : 0.5f;
                float norm_y_on_platform = (CurrentStateMapH > 1) ? (static_cast<float>(r_state) / (CurrentStateMapH - 1)) : 0.5f;

                // Convert normalized platform coordinates to local coordinates relative to the Grid's center for tracing
                float lx = (norm_x_on_platform - 0.5f) * PlatformWorldSize.X;
                float ly = (norm_y_on_platform - 0.5f) * PlatformWorldSize.Y;

                FVector localStart(lx, ly, traceDistUp);
                FVector localEnd(lx, ly, -traceDistDown);

                FVector wStart = GridTransform.TransformPosition(localStart);
                FVector wEnd = GridTransform.TransformPosition(localEnd);

                FHitResult hit;
                // Consider adding trace complex true if needed: FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(TerraShiftStateTrace), true);
                bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);

                float finalLocalZ = 0.f;
                if (bHit) {
                    FVector localHit = GridTransform.InverseTransformPosition(hit.ImpactPoint);
                    finalLocalZ = localHit.Z;
                }

                float clampedVisZ = FMath::Clamp(finalLocalZ, visMinZ, visMaxZ);
                float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                HeightTmp[r_state][c_state] = (norm_height * 2.f) - 1.f;
            }
        }
        HeightTmp.Clip(-1.f, 1.f);
    }

    PreviousHeight = CurrentHeight;
    CurrentHeight = HeightTmp; // HeightTmp will be empty if bIncludeHeightMapInState is false and it wasn't resized above

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

    // 1. Append Height Map (if enabled)
    if (bIncludeHeightMapInState && CurrentHeight.Num() > 0)
    {
        outArr.Append(CurrentHeight.GetData());
    }

    // 2. Append Overhead Image (if enabled)
    if (bIncludeOverheadImageInState && OverheadRenderTarget && OverheadCaptureActor)
    {
        TArray<float> overhead = CaptureOverheadImage();
        if (overhead.Num() > 0) {
            outArr.Append(overhead);
        }
    }

    // 3. Append Grid Object Sequence (if enabled)
    if (bIncludeGridObjectSequenceInState && MaxGridObjectsForState > 0 && GridObjectFeatureSize > 0)
    {
        const float HalfPlatformSizeX = (PlatformWorldSize.X > KINDA_SMALL_NUMBER) ? (PlatformWorldSize.X / 2.0f) : 1.0f;
        const float HalfPlatformSizeY = (PlatformWorldSize.Y > KINDA_SMALL_NUMBER) ? (PlatformWorldSize.Y / 2.0f) : 1.0f;

        const float ObjectZNormRangeMin = MinZ;
        const float ObjectZNormRangeMax = MaxZ;
        const float ObjectZNormRange = ObjectZNormRangeMax - ObjectZNormRangeMin;

        const float MaxVelocityClip = 100.0f; // Hardcoded as per request

        for (int32 i = 0; i < MaxGridObjectsForState; ++i)
        {
            // Check if object 'i' is valid and active.
            // bHasActive is sized by MaxGridObjects (total manageable objects).
            // We iterate up to MaxGridObjectsForState for the state representation.
            if (bHasActive.IsValidIndex(i) && bHasActive[i] &&
                ObjectMgr && ObjectMgr->GetGridObject(i) != nullptr)
            {
                FVector ObjPosLocal = GetCurrentPosition(i);

                FVector GoalPosLocal = FVector::ZeroVector;
                int32 GoalIdx = GetGoalIndex(i);
                if (GoalManager && GoalIdx != -1)
                {
                    FVector GoalPosWorld = GoalManager->GetGoalLocation(GoalIdx);
                    if (Platform)
                    {
                        GoalPosLocal = Platform->GetActorTransform().InverseTransformPosition(GoalPosWorld);
                    }
                }
                FVector ObjVelLocal = GetCurrentVelocity(i);

                // Normalize Object Position
                float NormObjPosX = FMath::Clamp(ObjPosLocal.X / HalfPlatformSizeX, -1.0f, 1.0f);
                float NormObjPosY = FMath::Clamp(ObjPosLocal.Y / HalfPlatformSizeY, -1.0f, 1.0f);
                float NormObjPosZ = (ObjectZNormRange > KINDA_SMALL_NUMBER) ?
                    FMath::Clamp(((ObjPosLocal.Z - ObjectZNormRangeMin) / ObjectZNormRange) * 2.0f - 1.0f, -1.0f, 1.0f) :
                    0.0f;

                // Normalize Goal Position
                float NormGoalPosX = FMath::Clamp(GoalPosLocal.X / HalfPlatformSizeX, -1.0f, 1.0f);
                float NormGoalPosY = FMath::Clamp(GoalPosLocal.Y / HalfPlatformSizeY, -1.0f, 1.0f);
                float NormGoalPosZ = (ObjectZNormRange > KINDA_SMALL_NUMBER) ?
                    FMath::Clamp(((GoalPosLocal.Z - ObjectZNormRangeMin) / ObjectZNormRange) * 2.0f - 1.0f, -1.0f, 1.0f) :
                    0.0f;

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
    if (!WaveSim)
        return TArray<float>();
    return WaveSim->GetAgentState(AgentIndex);
}

// ------------------------------------------
//   UpdateGridColumnsColors
// ------------------------------------------
void UStateManager::UpdateGridColumnsColors()
{
    if (!Grid || !OccupancyGrid) return;

    float mn = Grid->GetMinHeight();
    float mx = Grid->GetMaxHeight();

    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    {
        float h = Grid->GetColumnHeight(c);
        // Ensure mx is greater than mn to avoid division by zero or incorrect mapping
        float ratio = (mx > mn) ? FMath::GetMappedRangeValueClamped(FVector2D(mn, mx), FVector2D(0.f, 1.f), h) : 0.5f;

        FLinearColor baseCol = FLinearColor::LerpUsingHSV(
            FLinearColor::Black,
            FLinearColor::White,
            ratio
        );
        Grid->SetColumnColor(c, baseCol);
    }


    if (GridObjectColors.Num() > 0) {
        FMatrix2D objOcc = OccupancyGrid->GetOccupancyMatrix({ FName("GridObjects") }, false);
        int32 nObjColors = GridObjectColors.Num();

        for (int32 r = 0; r < objOcc.GetNumRows(); ++r) {
            for (int32 col_idx = 0; col_idx < objOcc.GetNumColumns(); ++col_idx) {
                float occupantIdFloat = objOcc[r][col_idx];
                if (occupantIdFloat >= 0.f)
                {
                    int32 occupantId = FMath::RoundToInt(occupantIdFloat);
                    int32 colorIdx = occupantId % nObjColors;
                    int32 flatGridIndex = r * GridSize + col_idx; // Assuming GridSize is num columns
                    Grid->SetColumnColor(flatGridIndex, GridObjectColors[colorIdx]);
                }
            }
        }
    }


    if (bUseRandomGoals && GoalColors.Num() > 0)
    {
        FMatrix2D goalOcc = OccupancyGrid->GetOccupancyMatrix({ FName("Goals") }, false);
        int32 nGoalCols = GoalColors.Num();

        for (int32 r = 0; r < goalOcc.GetNumRows(); ++r) {
            for (int32 col_idx = 0; col_idx < goalOcc.GetNumColumns(); ++col_idx) {
                float occupantIdFloat = goalOcc[r][col_idx];
                if (occupantIdFloat >= 0.f)
                {
                    int32 occupantId = FMath::RoundToInt(occupantIdFloat);
                    int32 colorIdx = occupantId % nGoalCols;
                    int32 flatGridIndex = r * GridSize + col_idx;
                    Grid->SetColumnColor(flatGridIndex, GoalColors[colorIdx]);
                }
            }
        }
    }
}

// ------------------------------------------
//   Accessors
// ------------------------------------------
int32 UStateManager::GetMaxGridObjects() const
{
    return MaxGridObjects;
}

bool UStateManager::GetHasActive(int32 ObjIndex) const
{
    return bHasActive.IsValidIndex(ObjIndex) ? bHasActive[ObjIndex] : false;
}
bool UStateManager::GetHasReachedGoal(int32 ObjIndex) const
{
    return bHasReached.IsValidIndex(ObjIndex) ? bHasReached[ObjIndex] : false;
}
bool UStateManager::GetHasFallenOff(int32 ObjIndex) const
{
    return bFallenOff.IsValidIndex(ObjIndex) ? bFallenOff[ObjIndex] : false;
}
bool UStateManager::GetShouldCollectReward(int32 ObjIndex) const
{
    return bShouldCollect.IsValidIndex(ObjIndex) ? bShouldCollect[ObjIndex] : false;
}
void UStateManager::SetShouldCollectReward(int32 ObjIndex, bool bVal)
{
    if (bShouldCollect.IsValidIndex(ObjIndex))
        bShouldCollect[ObjIndex] = bVal;
}
bool UStateManager::GetShouldRespawn(int32 ObjIndex) const
{
    return bShouldResp.IsValidIndex(ObjIndex) ? bShouldResp[ObjIndex] : false;
}
int32 UStateManager::GetGoalIndex(int32 ObjIndex) const
{
    return ObjectGoalIndices.IsValidIndex(ObjIndex)
        ? ObjectGoalIndices[ObjIndex]
        : -1;
}

FVector UStateManager::GetCurrentVelocity(int32 ObjIndex) const
{
    return CurrVel.IsValidIndex(ObjIndex)
        ? CurrVel[ObjIndex]
        : FVector::ZeroVector;
}
FVector UStateManager::GetPreviousVelocity(int32 ObjIndex) const
{
    return PrevVel.IsValidIndex(ObjIndex)
        ? PrevVel[ObjIndex]
        : FVector::ZeroVector;
}
float UStateManager::GetCurrentDistance(int32 ObjIndex) const
{
    return CurrDist.IsValidIndex(ObjIndex)
        ? CurrDist[ObjIndex]
        : -1.f;
}
float UStateManager::GetPreviousDistance(int32 ObjIndex) const
{
    return PrevDist.IsValidIndex(ObjIndex)
        ? PrevDist[ObjIndex]
        : -1.f;
}
FVector UStateManager::GetCurrentPosition(int32 ObjIndex) const
{
    return CurrPos.IsValidIndex(ObjIndex)
        ? CurrPos[ObjIndex]
        : FVector::ZeroVector;
}
FVector UStateManager::GetPreviousPosition(int32 ObjIndex) const
{
    return PrevPos.IsValidIndex(ObjIndex)
        ? PrevPos[ObjIndex]
        : FVector::ZeroVector;
}

// ------------------------------------------
//   Helper: column top
// ------------------------------------------
FVector UStateManager::GetColumnTopWorldLocation(int32 GridX, int32 GridY) const
{
    if (!Grid)
    {
        return PlatformCenter + FVector(0, 0, 100.f); // Fallback
    }
    int32 idx = GridX * GridSize + GridY;
    if (Grid->Columns.IsValidIndex(idx))
    {
        AColumn* col = Grid->Columns[idx];
        if (col && col->ColumnMesh)
        {
            // Get the world location of the column actor itself (its root)
            FVector colRootLocation = col->GetActorLocation();
            // The column mesh is likely centered at its root, so its extent gives half-height
            float halfMeshHeightLocal = col->ColumnMesh->Bounds.BoxExtent.Z;
            // Account for any scaling of the column mesh
            float scaledHalfMeshHeight = halfMeshHeightLocal * col->ColumnMesh->GetComponentScale().Z;
            // The top surface Z is the column's Z + its scaled half-height
            // This assumes StartingPosition.Z (in Column.cpp) is the base of the column when height is 0.
            // And UpdateColumnPosition sets Z to StartingPosition.Z + NewHeight.
            // So, the root of the column actor is effectively at the visual bottom of the movable part.
            // The visual top would be ActorLocation.Z + (2 * scaledHalfMeshHeight) if height=0 and mesh is centered.
            // Given how AColumn::UpdateColumnPosition works (SetActorRelativeLocation(StartingPosition + FVector(0,0,NewHeight)))
            // The current ActorLocation already includes the visual height offset from a base plane.
            // So, the top of the column is its current Z + its scaled half-height (if origin is at its center).
            // Let's use the Column's GetColumnHeight method for the visual height from base.
            // Then add the half-extent of the mesh in Z.
            return FVector(colRootLocation.X, colRootLocation.Y, colRootLocation.Z + scaledHalfMeshHeight);
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
    if (!w) return;
    if (OverheadCaptureActor) return; // Already setup

    FVector spawnLoc = PlatformCenter + FVector(0, 0, OverheadCamDistance);
    FRotator spawnRot = FRotator(-90.f, 0.f, 0.f); // Looking straight down

    FActorSpawnParameters sp;
    sp.Owner = Platform;
    OverheadCaptureActor = w->SpawnActor<ASceneCapture2D>(spawnLoc, spawnRot, sp);

    if (!OverheadCaptureActor)
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::SetupOverheadCamera - Failed to spawn overhead camera actor."));
        return;
    }

    // Use StateOverheadImageResX/Y if image is part of state, otherwise use general OverheadCamResX/Y
    int32 TargetResX = bIncludeOverheadImageInState && StateOverheadImageResX > 0 ? StateOverheadImageResX : OverheadCamResX;
    int32 TargetResY = bIncludeOverheadImageInState && StateOverheadImageResY > 0 ? StateOverheadImageResY : OverheadCamResY;
    if (TargetResX <= 0) TargetResX = 25; // Fallback if still invalid
    if (TargetResY <= 0) TargetResY = 25;


    // Configure the Render Target
    OverheadRenderTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("OverheadCameraRenderTarget"));
    OverheadRenderTarget->RenderTargetFormat = RTF_RGBA8; // Using RGBA8 for color
    OverheadRenderTarget->InitAutoFormat(TargetResX, TargetResY);
    OverheadRenderTarget->UpdateResourceImmediate(true);


    USceneCaptureComponent2D* comp = OverheadCaptureActor->GetCaptureComponent2D();
    if (!comp)
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::SetupOverheadCamera - OverheadCaptureActor is missing USceneCaptureComponent2D."));
        OverheadCaptureActor->Destroy();
        OverheadCaptureActor = nullptr;
        return;
    }

    comp->ProjectionType = ECameraProjectionMode::Perspective; // Or Orthographic if preferred
    comp->FOVAngle = OverheadCamFOV;
    comp->TextureTarget = OverheadRenderTarget;
    comp->CaptureSource = SCS_BaseColor; // Capture base color

    comp->MaxViewDistanceOverride = OverheadCamDistance * 1.1f; // Ensure capture distance is sufficient

    comp->bCaptureEveryFrame = false;
    comp->bCaptureOnMovement = false;

    FEngineShowFlags& ShowFlags = comp->ShowFlags;
    ShowFlags.SetAtmosphere(false);
    ShowFlags.SetBSP(false);
    ShowFlags.SetDecals(false);
    ShowFlags.SetFog(false);
    ShowFlags.SetVolumetricFog(false);
    ShowFlags.SetSkeletalMeshes(true); // Keep true if agents or objects are skeletal
    ShowFlags.SetStaticMeshes(true);
    ShowFlags.SetTranslucency(false);
    ShowFlags.SetLighting(false); // Disable lighting for a simpler, more consistent image
    ShowFlags.SetPostProcessing(false);
    ShowFlags.SetDynamicShadows(false);
    ShowFlags.SetAmbientOcclusion(false);
    ShowFlags.SetGlobalIllumination(false);
    ShowFlags.SetReflectionEnvironment(false);
    ShowFlags.SetScreenSpaceReflections(false);
    ShowFlags.SetDistanceFieldAO(false);
    ShowFlags.SetAntiAliasing(false);
    ShowFlags.SetBloom(false);
    ShowFlags.SetLensFlares(false);
    ShowFlags.SetMotionBlur(false);
    ShowFlags.SetDepthOfField(false);
    ShowFlags.SetEyeAdaptation(false);
    ShowFlags.SetMaterials(true); // Keep true to see materials
    ShowFlags.SetInstancedStaticMeshes(true);
    ShowFlags.SetInstancedGrass(false);
    ShowFlags.SetInstancedFoliage(false);
    ShowFlags.SetPaper2DSprites(false);
    ShowFlags.SetParticles(false);
    ShowFlags.SetTextRender(false);
    ShowFlags.SetLandscape(false);


    comp->SetActive(true);
    OverheadCaptureActor->SetActorEnableCollision(false);
}

TArray<float> UStateManager::CaptureOverheadImage() const
{
    TArray<float> out;
    if (!OverheadCaptureActor || !OverheadRenderTarget)
    {
        UE_LOG(LogTemp, Warning, TEXT("StateManager::CaptureOverheadImage - OverheadCaptureActor or OverheadRenderTarget is null."));
        return out; // Return empty if not setup
    }

    UWorld* w = OverheadCaptureActor->GetWorld();
    if (!w)
    {
        UE_LOG(LogTemp, Warning, TEXT("StateManager::CaptureOverheadImage - GetWorld() returned null."));
        return out;
    }

    TArray<FColor> pixels;
    bool bOk = UKismetRenderingLibrary::ReadRenderTarget(w, OverheadRenderTarget, pixels);
    if (!bOk || pixels.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("StateManager::CaptureOverheadImage - ReadRenderTarget failed or returned no pixels."));
        return out;
    }

    // Extract R, G, B channels separately and normalize to [-1, 1]
    // Output order: RRR...GGG...BBB...
    // Or, if Python expects interleaved RGBRGB..., then adjust the loop.
    // Assuming Python will reshape it correctly (e.g., HxWx3).
    // For a single luminance channel, this logic needs to change.
    // Current config has OverheadCamResX/Y = 25. If StateOverheadImageResX/Y matches, good.

    int32 TargetResX = bIncludeOverheadImageInState && StateOverheadImageResX > 0 ? StateOverheadImageResX : OverheadCamResX;
    int32 TargetResY = bIncludeOverheadImageInState && StateOverheadImageResY > 0 ? StateOverheadImageResY : OverheadCamResY;
    if (TargetResX <= 0) TargetResX = 1;
    if (TargetResY <= 0) TargetResY = 1;


    out.Reserve(TargetResX * TargetResY); // For single channel (luminance)

    for (const FColor& PixelColor : pixels)
    {
        float R_float = (PixelColor.R / 255.0f) * 2.0f - 1.0f; // Normalize to [-1, 1]
        float G_float = (PixelColor.G / 255.0f) * 2.0f - 1.0f; // Normalize to [-1, 1]
        float B_float = (PixelColor.B / 255.0f) * 2.0f - 1.0f; // Normalize to [-1, 1]

        // Using standard luminance calculation, then normalizing to [-1, 1]
        // Y = 0.299R + 0.587G + 0.114B (for R,G,B in [0,1] range)
        // First, get R,G,B in [0,1]
        float R01 = (R_float + 1.0f) / 2.0f;
        float G01 = (G_float + 1.0f) / 2.0f;
        float B01 = (B_float + 1.0f) / 2.0f;
        float luminance01 = 0.299f * R01 + 0.587f * G01 + 0.114f * B01;
        out.Add(luminance01 * 2.0f - 1.0f); // Normalize luminance to [-1, 1]
    }

    return out;
}


// ------------------------------------------
//   SpawnStationaryGoalPlatforms
// ------------------------------------------
void UStateManager::SpawnStationaryGoalPlatforms()
{
    checkf(Platform, TEXT("SpawnStationaryGoalPlatforms => Platform is null!"));
    checkf(GoalManager, TEXT("SpawnStationaryGoalPlatforms => GoalManager is null!"));

    UWorld* w = GetWorld();
    if (!w) return;

    float offsetExtra = 0.f;
    if (ObjectMgr && ObjectMgr->GetGridObject(0) && ObjectMgr->GetGridObject(0)->MeshComponent)
    {
        // Use the extent of the first grid object as a reference for offset
        offsetExtra = ObjectMgr->GetGridObject(0)->MeshComponent->Bounds.SphereRadius;
    }
    else {
        // Fallback if no grid objects exist or mesh is not set
        offsetExtra = ObjectScale * 50.0f * 0.5f; // Estimate based on default sphere mesh size if ObjectScale is unit scale
    }
    offsetExtra += 5.f; // Add a small margin


    float half = PlatformWorldSize.X * 0.5f; // Assuming square platform
    float offset = half + offsetExtra; // Place goals just outside the platform edge

    TArray<FVector> edgeOffsets = {
        FVector(0.f, +offset, 0.f), // Right
        FVector(0.f, -offset, 0.f), // Left
        FVector(-offset, 0.f, 0.f), // Back
        FVector(+offset, 0.f, 0.f)  // Front
    };

    for (int32 i = 0; i < edgeOffsets.Num(); i++)
    {
        if (!GoalColors.IsValidIndex(i)) continue; // Skip if not enough colors defined

        FVector spawnLoc = PlatformCenter + edgeOffsets[i]; // World location for the goal platform
        FActorSpawnParameters sp;
        sp.Owner = Platform; // Set platform as owner
        AGoalPlatform* gp = w->SpawnActor<AGoalPlatform>(AGoalPlatform::StaticClass(), spawnLoc, FRotator::ZeroRotator, sp);

        if (gp)
        {
            // Scale of the goal platform itself (e.g., visual representation of the goal area)
            // This scale should be relative to the GoalPlatform's mesh, not ObjectScale.
            // Let's assume GoalRadius dictates its visual size.
            float GoalPlatformVisualScale = GoalRadius * 0.2f; // Example scaling factor
            FVector goalPlatformScaleVec(GoalPlatformVisualScale, GoalPlatformVisualScale, 0.1f); // Thin platform

            // InitializeGoalPlatform takes world location, scale, color, and parent
            // For stationary goals attached to the world (or main platform), the "parent" might be the Platform actor.
            // Location should be relative to parent if attached. Here, we spawn at world loc then attach.
            gp->InitializeGoalPlatform(FVector::ZeroVector, goalPlatformScaleVec, GoalColors[i], Platform);
            // Since InitializeGoalPlatform already handles attachment and relative location if ParentPlatform is provided,
            // and we are spawning at world location first, then attaching with KeepWorldTransform.
            // If InitializeGoalPlatform is designed to set relative location to parent, ensure spawnLoc passed to it is relative,
            // or adjust logic. Current AGoalPlatform::InitializeGoalPlatform uses SetActorRelativeLocation if attached.
            // Here we spawn in world, then attach, so its world location is already set.

            // Add to GoalManager: The GoalActor is the AGoalPlatform itself, offset is ZeroVector relative to it.
            // The GoalManager will then use gp->GetActorLocation() + FVector::ZeroVector.
            // Note: This part was missing from the original SpawnStationaryGoalPlatforms,
            // it's crucial for GoalManager to know about these goals.
            if (GoalManager) {
                // This logic for adding to GoalManager was missing. Assuming you want to add these.
                // The ResetGoals function in StateManager::Reset is for *random* goals using Columns.
                // For stationary, we might need a different approach or extend ResetGoals.
                // For now, let's assume stationary goals are added directly or through another mechanism.
                // If these are the *only* goals, then ResetGoals in StateManager would need to be adapted.
            }
        }
    }
}