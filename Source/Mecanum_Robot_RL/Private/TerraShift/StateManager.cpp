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
    bRemoveGridObjectOnGoalReached = Config->GetOrDefaultBool(TEXT("bRemoveGridObjectOnGoalReached"), false);
    bRemoveGridObjectOnOOB = Config->GetOrDefaultBool(TEXT("bRemoveGridObjectOnOOB"), false);
    bRespawnGridObjectOnGoalReached = Config->GetOrDefaultBool(TEXT("bRespawnGridObjectOnGoalReached"), false);

    GoalRadius = Config->GetOrDefaultNumber(TEXT("GoalRadius"), GoalRadius);
    ObjectRadius = Config->GetOrDefaultNumber(TEXT("ObjectRadius"), ObjectRadius);

    // Read GoalColors
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

    // Read GridObjectColors
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

    // Create or re-init the OccupancyGrid
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
        bShouldResp[i] = true;

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
        RespawnDelays[i] = BaseRespawnDelay * i;
    }

    // 3) Reset NxN height arrays & step counter
    PreviousHeight = FMatrix2D(GridSize, GridSize, 0.f);
    CurrentHeight = FMatrix2D(GridSize, GridSize, 0.f);
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
                bool bInRadius = GoalManager->IsInRadiusOf(gIdx, wPos, GoalRadius);
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
    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
        {
            // zero stats
            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f;
            CurrDist[i] = -1.f;
            PrevPos[i] = FVector::ZeroVector;
            CurrPos[i] = FVector::ZeroVector;

            if (bShouldResp[i])
            {
                RespawnTimer[i] += DeltaTime;
            }
            continue;
        }

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        FVector wVel = Obj->MeshComponent->GetPhysicsLinearVelocity();
        FVector wPos = Obj->GetObjectLocation();

        int32 gIdx = ObjectGoalIndices[i];
        FVector wGoal(0.f);
        if (gIdx >= 0)
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

        if (bShouldResp[i])
        {
            RespawnTimer[i] += DeltaTime;
        }

        // occupant position => from wPos
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

    for (int32 i = 0; i < bShouldResp.Num(); i++)
    {
        if (bShouldResp[i] && RespawnTimer[i] >= RespawnDelays[i])
        {
            // Remove old occupant in "GridObjects" layer
            OccupancyGrid->RemoveObject(i, FName("GridObjects"));

            float radiusCells = ObjectRadius / CellSize;

            // occupant ID = i
            int32 cellIdx = OccupancyGrid->AddObjectToGrid(
                i,
                FName("GridObjects"),
                radiusCells,
                /*OverlapLayers=*/ TArray<FName>{  }
            );
            if (cellIdx < 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("No free occupant cell for obj %d"), i);
                continue;
            }

            // Convert cell => top-of-column
            int32 gx = cellIdx / GridSize;
            int32 gy = cellIdx % GridSize;
            FVector spawnLoc = GetColumnTopWorldLocation(gx, gy);

            // Actually spawn the object
            ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(ObjectScale), ObjectMass);

            // Color => occupant i => i % nObjColors
            AGridObject* newObj = ObjectMgr->GetGridObject(i);
            if (newObj)
            {
                int32 colorIdx = i % nObjColors;  // round‐robin object color
                newObj->SetGridObjectColor(GridObjectColors[colorIdx]);
            }

            bHasActive[i] = true;
            bHasReached[i] = false;
            bFallenOff[i] = false;
            bShouldResp[i] = false;
            RespawnTimer[i] = 0.f;
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
// ------------------------------------------
//   BuildCentralState (Centered Visualization)
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

    // Define the Z range specifically for visualization, centered around 0 local Z
    // Use half of the configured total range
    float VisualizationZRange = MaxZ;
    float halfVisRange = VisualizationZRange * 0.5f;
    float visMinZ = -halfVisRange;
    float visMaxZ = halfVisRange;

    // Keep trace distance large enough, independent of visualization range
    // Use MinZ/MaxZ (OOB limits) or fixed large values to define trace extent
    float traceDistUp = FMath::Abs(MaxZ) + 100.0f; // Little extra space
    float traceDistDown = FMath::Abs(visMinZ) + 100.0f;

    float halfPlatformSize = (GridSize > 0) ? (PlatformWorldSize.X * 0.5f) : 0.f;
    FTransform GridTransform = Grid->GetActorTransform();
    FMatrix2D HeightTmp(GridSize, GridSize, 0.f);

    for (int32 row = 0; row < GridSize; row++)
    {
        for (int32 col = 0; col < GridSize; col++)
        {
            // 1. Calculate XY center in Grid's local space
            float lx = (col + 0.5f) * CellSize - halfPlatformSize;
            float ly = (row + 0.5f) * CellSize - halfPlatformSize;

            // 2. Define trace start/end in Grid's local space (using robust distances)
            FVector localStart(lx, ly, traceDistUp);
            FVector localEnd(lx, ly, -traceDistDown);

            // 3. Transform to world space for trace
            FVector wStart = GridTransform.TransformPosition(localStart);
            FVector wEnd = GridTransform.TransformPosition(localEnd);

            FHitResult hit;
            bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);

            float finalLocalZ = 0.f; // Default to 0 (relative Z) if no hit
            if (bHit)
            {
                // 4. Transform hit point back to Grid's local space
                FVector localHit = GridTransform.InverseTransformPosition(hit.ImpactPoint);
                finalLocalZ = localHit.Z;
            }

            // 5. Clamp the local Z value based on the *visualization* range [visMinZ, visMaxZ]
            float clampedVisZ = FMath::Clamp(finalLocalZ, visMinZ, visMaxZ);

            // 6. Normalize the clamped *visualization* Z value to the range [-1, 1]
            float norm = 0.f;
            // Check if visualization range is valid
            if (visMaxZ > visMinZ) {
                // Map [visMinZ, visMaxZ] to [0, 1]
                norm = (clampedVisZ - visMinZ) / (visMaxZ - visMinZ);
            }
            float mapped = (norm * 2.f) - 1.f; // Map [0, 1] to [-1, 1]
            HeightTmp[row][col] = mapped;
        }
    }

    // Update height maps
    HeightTmp.Clip(-1, 1);
    PreviousHeight = CurrentHeight;
    CurrentHeight = HeightTmp;

    // Capture overhead camera
    if (OverheadCaptureActor)
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
    UWorld* w = (Grid) ? Grid->GetWorld() : nullptr;
    float dt = (w) ? w->GetDeltaSeconds() : 0.f;

    TArray<float> outArr;
    // (1) current height
    outArr.Append(CurrentHeight.Data);

    // (2) delta height
    if (Step > 1 && dt > SMALL_NUMBER)
    {
        FMatrix2D diff = ((CurrentHeight - PreviousHeight) / dt);
        diff.Clip(-1, 1);
        outArr.Append(diff.Data);
    }
    else
    {
        outArr.Append(PreviousHeight.Data);
    }

    // (3) overhead camera
    TArray<float> overhead = CaptureOverheadImage();
    outArr.Append(overhead);

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
    // 1) Color by height
    float mn = Grid->GetMinHeight();
    float mx = Grid->GetMaxHeight();
    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    {
        float h = Grid->GetColumnHeight(c);
        float ratio = FMath::GetMappedRangeValueClamped(
            FVector2D(mn, mx),
            FVector2D(0.f, 1.f),
            h
        );
        FLinearColor baseCol = FLinearColor::LerpUsingHSV(
            FLinearColor::Black,
            FLinearColor::White,
            ratio
        );
        Grid->SetColumnColor(c, baseCol);
    }

    // 2) Overwrite columns if occupied by "GridObjects"
    //{
    //    FMatrix2D objOcc = OccupancyGrid->GetOccupancyMatrix(
    //        { FName("GridObjects") },
    //        /*bUseBinary=*/ false
    //    );
    //    int32 nObjColors = GridObjectColors.Num();
    //    if (nObjColors == 0)
    //    {
    //        GridObjectColors.Add(FLinearColor::White);
    //        nObjColors = 1;
    //    }

    //    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    //    {
    //        int gx = c / GridSize;
    //        int gy = c % GridSize;

    //        float occupantIdFloat = objOcc[gx][gy];
    //        if (occupantIdFloat >= 0.f) // occupant present
    //        {
    //            int32 occupantId = FMath::RoundToInt(occupantIdFloat);
    //            int32 colorIdx = (occupantId >= 0)
    //                ? occupantId % nObjColors
    //                : 0;
    //            Grid->SetColumnColor(c, GridObjectColors[colorIdx]);
    //        }
    //    }
    //}

    // 3) Overlay "Goals"
    if (bUseRandomGoals && GoalColors.Num() > 0 && OccupancyGrid)
    {
        FMatrix2D goalOcc = OccupancyGrid->GetOccupancyMatrix(
            { FName("Goals") },
            /*bUseBinary=*/ false
        );
        int32 nGoalCols = GoalColors.Num();

        for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
        {
            int gx = c / GridSize;
            int gy = c % GridSize;

            float occupantIdFloat = goalOcc[gx][gy];
            if (occupantIdFloat < 0.f)
                continue; // no occupant => keep current color

            int32 occupantId = FMath::RoundToInt(occupantIdFloat);
            int32 colIdx = (occupantId >= 0)
                ? occupantId % nGoalCols
                : 0;
            Grid->SetColumnColor(c, GoalColors[colIdx]);
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
        return PlatformCenter + FVector(0, 0, 100.f);
    }
    int32 idx = GridX * GridSize + GridY;
    if (Grid->Columns.IsValidIndex(idx))
    {
        AColumn* col = Grid->Columns[idx];
        if (col && col->ColumnMesh)
        {
            FVector centerWS = col->ColumnMesh->GetComponentLocation();
            float halfZ = col->ColumnMesh->Bounds.BoxExtent.Z;
            return centerWS + FVector(0.f, 0.f, halfZ);
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
    if (OverheadCaptureActor) return;

    FVector spawnLoc = PlatformCenter + FVector(0, 0, OverheadCamDistance);
    FRotator spawnRot = FRotator(-90.f, 0.f, 0.f);

    FActorSpawnParameters sp;
    sp.Owner = Platform;
    OverheadCaptureActor = w->SpawnActor<ASceneCapture2D>(spawnLoc, spawnRot, sp);
    if (!OverheadCaptureActor)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn overhead camera actor."));
        return;
    }

    OverheadRenderTarget = NewObject<UTextureRenderTarget2D>();
    OverheadRenderTarget->RenderTargetFormat = RTF_RGBA8;
    OverheadRenderTarget->InitAutoFormat(OverheadCamResX, OverheadCamResY);

    USceneCaptureComponent2D* comp = OverheadCaptureActor->GetCaptureComponent2D();
    comp->ProjectionType = ECameraProjectionMode::Perspective;
    comp->FOVAngle = OverheadCamFOV;
    comp->TextureTarget = OverheadRenderTarget;
    comp->CaptureSource = ESceneCaptureSource::SCS_BaseColor;
    comp->MaxViewDistanceOverride = OverheadCamDistance * 1.1f;
    comp->bCaptureEveryFrame = false;
    comp->bCaptureOnMovement = false;
}

// ------------------------------------------
//   CaptureOverheadImage
// ------------------------------------------
TArray<float> UStateManager::CaptureOverheadImage() const
{
    TArray<float> out;
    if (!OverheadCaptureActor || !OverheadRenderTarget)
        return out;

    UWorld* w = OverheadCaptureActor->GetWorld();
    if (!w) return out;

    TArray<FColor> pixels;
    bool bOk = UKismetRenderingLibrary::ReadRenderTarget(w, OverheadRenderTarget, pixels);
    if (!bOk || pixels.Num() == 0)
        return out;

    TArray<float> R; R.Reserve(pixels.Num());
    TArray<float> G; G.Reserve(pixels.Num());
    TArray<float> B; B.Reserve(pixels.Num());

    for (FColor c : pixels)
    {
        R.Add(c.R / 255.f);
        G.Add(c.G / 255.f);
        B.Add(c.B / 255.f);
    }
    out.Append(R);
    out.Append(G);
    out.Append(B);
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
    {
        FActorSpawnParameters sp;
        AGoalPlatform* tempGP = w->SpawnActor<AGoalPlatform>(
            AGoalPlatform::StaticClass(),
            FVector::ZeroVector,
            FRotator::ZeroRotator,
            sp
        );
        if (tempGP)
        {
            FVector bExt = tempGP->MeshComponent->Bounds.BoxExtent;
            offsetExtra = bExt.Y * tempGP->GetActorScale3D().Y + 5.f;
            tempGP->Destroy();
        }
    }

    float half = PlatformWorldSize.X * 0.5f;
    float offset = half + offsetExtra;

    TArray<FVector> edgeOffsets = {
        FVector(0.f, +offset, 0.f),
        FVector(0.f, -offset, 0.f),
        FVector(-offset, 0.f, 0.f),
        FVector(+offset, 0.f, 0.f)
    };

    for (int32 i = 0; i < edgeOffsets.Num(); i++)
    {
        FVector spawnLoc = PlatformCenter + edgeOffsets[i];
        FActorSpawnParameters sp;
        AGoalPlatform* gp = w->SpawnActor<AGoalPlatform>(
            AGoalPlatform::StaticClass(),
            spawnLoc,
            FRotator::ZeroRotator,
            sp
        );
        if (gp && GoalColors.IsValidIndex(i))
        {
            FVector s(ObjectScale, ObjectScale, ObjectScale);
            gp->InitializeGoalPlatform(FVector::ZeroVector, s, GoalColors[i], Platform);
        }
    }
}
