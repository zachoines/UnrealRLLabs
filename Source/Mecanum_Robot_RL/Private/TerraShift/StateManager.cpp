#include "TerraShift/StateManager.h"
#include "TerraShift/GridObject.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GoalManager.h"
#include "TerraShift/GoalPlatform.h"
#include "TerraShift/Column.h"

#include "Kismet/KismetSystemLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"

void UStateManager::LoadConfig(UEnvironmentConfig* Config)
{
    if (!Config)
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::LoadConfig => null config!"));
        return;
    }

    // same usage of GetOrDefault as before
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
    ColumnRadius = Config->GetOrDefaultNumber(TEXT("ColumnRadius"), ColumnRadius);

    // read GoalColors
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

    // read GridObjectColors
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

    // checkf => crash if null
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

    SetupOverheadCamera();
}

void UStateManager::Reset(int32 NumObjects, int32 CurrentAgents)
{
    checkf(Platform, TEXT("StateManager::Reset => Platform is null!"));
    checkf(ObjectMgr, TEXT("StateManager::Reset => ObjectMgr is null!"));
    checkf(Grid, TEXT("StateManager::Reset => Grid is null!"));
    checkf(WaveSim, TEXT("StateManager::Reset => WaveSim is null!"));
    checkf(GoalManager, TEXT("StateManager::Reset => GoalManager is null!"));

    // 0)  reset the main grid, object manager, wave sim
    Grid->ResetGrid();
    ObjectMgr->ResetGridObjects();
    WaveSim->Reset(CurrentAgents);

    // 1) allocate arrays
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

    // NxN => reset
    PreviousHeight = FMatrix2D(GridSize, GridSize, 0.f);
    CurrentHeight = FMatrix2D(GridSize, GridSize, 0.f);
    Step = 0;

    // 3) If random => pick random columns for each goal color
    //    If not random => spawn AGoalPlatforms along edges
    TArray<AActor*> goalActors;
    TArray<FVector> offsets;

    if (bUseRandomGoals)
    {
        int32 colTotal = Grid->GetTotalColumns();
        int32 numGoalCols = GoalColors.Num();

        for (int32 g = 0; g < numGoalCols; g++)
        {
            if (colTotal <= 0) break;
            int32 randIdx = FMath::RandRange(0, colTotal - 1);
            AColumn* col = (Grid->Columns.IsValidIndex(randIdx) ? Grid->Columns[randIdx] : nullptr);
            if (col)
            {
                float halfZ = col->ColumnMesh->Bounds.BoxExtent.Z;
                goalActors.Add(col);
                offsets.Add(FVector(0, 0, halfZ));
            }
        }
    }
    else
    {
        SpawnStationaryGoalPlatforms();

        TArray<AActor*> foundGoalPlatforms;
        for (TObjectIterator<AGoalPlatform> it; it; ++it)
        {
            if (it->GetWorld() == GetWorld())
            {
                foundGoalPlatforms.Add(*it);
            }
        }

        int32 limit = FMath::Min(GoalColors.Num(), foundGoalPlatforms.Num());
        for (int32 i = 0; i < limit; i++)
        {
            AGoalPlatform* gp = Cast<AGoalPlatform>(foundGoalPlatforms[i]);
            if (gp)
            {
                goalActors.Add(gp);
                offsets.Add(FVector::ZeroVector);
            }
        }
    }
    GoalManager->ResetGoals(goalActors, offsets);

    // 4) Each object => goal index
    int32 numGoals = GoalColors.Num();
    for (int32 i = 0; i < NumObjects; i++)
    {
        ObjectGoalIndices[i] = (numGoals > 0) ? (i % numGoals) : -1;
    }
}

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
        FVector(0.f,+offset, 0.f),
        FVector(0.f,-offset, 0.f),
        FVector(-offset,0.f,0.f),
        FVector(+offset,0.f,0.f)
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
            gp->InitializeGoalPlatform(
                FVector::ZeroVector,
                s,
                GoalColors[i],
                Platform
            );
        }
    }
}

void UStateManager::UpdateGridObjectFlags()
{
    // references guaranteed
    float halfX = PlatformWorldSize.X * 0.5f;
    float halfY = PlatformWorldSize.Y * 0.5f;
    float minZLocal = PlatformCenter.Z + MinZ;
    float maxZLocal = PlatformCenter.Z + MaxZ;

    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        FVector wPos = Obj->GetObjectLocation();

        if (!bHasReached[i])
        {
            int32 gIdx = ObjectGoalIndices[i];
            if (gIdx >= 0)
            {
                bool bInRadius = GoalManager->IsInRadiusOf(gIdx, wPos, GoalRadius);
                if (bInRadius)
                {
                    if (bRespawnGridObjectOnGoalReached)
                    {
                        bHasReached[i] = true;
                        bShouldCollect[i] = true;
                    }
                    else
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
                        ObjectMgr->DisableGridObject(i);
                    }
                }
            }
        }

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
            else if (zPos<minZLocal || zPos>maxZLocal)
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
                    ObjectMgr->DisableGridObject(i);
                }
                else
                {
                    bHasActive[i] = false;
                    bShouldResp[i] = true;
                    ObjectMgr->DisableGridObject(i);
                }
            }
        }
    }
}

void UStateManager::UpdateObjectStats(float DeltaTime)
{
    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
        {
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

        PrevVel[i] = CurrVel[i];
        PrevAcc[i] = CurrAcc[i];
        PrevDist[i] = CurrDist[i];
        PrevPos[i] = CurrPos[i];

        CurrVel[i] = locVel;
        CurrAcc[i] = (DeltaTime > SMALL_NUMBER) ? (locVel - PrevVel[i]) / DeltaTime
            : FVector::ZeroVector;
        CurrPos[i] = locPos;
        CurrDist[i] = FVector::Dist(locPos, locGoal);

        if (bShouldResp[i])
        {
            RespawnTimer[i] += DeltaTime;
        }
    }
}

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
            FVector spawnLoc = GenerateRandomGridLocation();
            ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(ObjectScale), ObjectMass);

            AGridObject* newObj = ObjectMgr->GetGridObject(i);
            if (newObj)
            {
                int32 gIdx = ObjectGoalIndices[i];
                int32 cIdx = (gIdx >= 0 && gIdx < nObjColors) ? gIdx : 0;
                newObj->SetGridObjectColor(GridObjectColors[cIdx]);
            }

            bHasActive[i] = true;
            bHasReached[i] = false;
            bFallenOff[i] = false;
            bShouldResp[i] = false;
            RespawnTimer[i] = 0.f;
        }
    }
}

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

void UStateManager::BuildCentralState()
{
    UWorld* w = Grid->GetWorld();
    if (!w)
    {
        UE_LOG(LogTemp, Warning, TEXT("BuildCentralState => no valid UWorld."));
        return;
    }

    float half = (GridSize > 0) ? PlatformWorldSize.X * 0.5f : 0.f;
    float pZ = Platform->GetActorLocation().Z;

    FMatrix2D HeightTmp(GridSize, GridSize, 0.f);
    for (int32 row = 0; row < GridSize; row++)
    {
        for (int32 col = 0; col < GridSize; col++)
        {
            float lx = (col + 0.5f) * CellSize - half;
            float ly = (row + 0.5f) * CellSize - half;
            FVector startPos(lx, ly, pZ + MaxZ);
            FVector endPos(lx, ly, pZ - MaxZ);

            FVector wStart = Grid->GetActorTransform().TransformPosition(startPos);
            FVector wEnd = Grid->GetActorTransform().TransformPosition(endPos);

            FHitResult hit;
            FCollisionQueryParams tParams(FName(TEXT("TopDownRay")), true);
            bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility, tParams);

            float finalZ = 0.f;
            if (bHit)
            {
                FVector localHit = Grid->GetActorTransform().InverseTransformPosition(hit.ImpactPoint);
                finalZ = localHit.Z;
            }
            float cVal = FMath::Clamp(finalZ, MinZ, MaxZ);
            float norm = (cVal - MinZ) / (MaxZ - MinZ);
            float mapped = (norm * 2.f) - 1.f;
            HeightTmp[row][col] = mapped;
        }
    }

    PreviousHeight = CurrentHeight;
    CurrentHeight = HeightTmp;

    if (OverheadCaptureActor)
    {
        OverheadCaptureActor->GetCaptureComponent2D()->CaptureScene();
    }
    Step++;
}

TArray<float> UStateManager::GetCentralState()
{
    UWorld* w = (Grid) ? Grid->GetWorld() : nullptr;
    float dt = (w) ? w->GetDeltaSeconds() : 0.f;

    TArray<float> outArr;
    outArr.Append(CurrentHeight.Data);

    if (Step > 1 && dt > SMALL_NUMBER)
    {
        FMatrix2D diff = (CurrentHeight - PreviousHeight) / dt;
        outArr.Append(diff.Data);
    }
    else
    {
        outArr.Append(PreviousHeight.Data);
    }

    TArray<float> overhead = CaptureOverheadImage();
    outArr.Append(overhead);

    return outArr;
}

TArray<float> UStateManager::GetAgentState(int32 AgentIndex) const
{
    if (!WaveSim)
        return TArray<float>();

    return WaveSim->GetAgentState(AgentIndex);
}

void UStateManager::UpdateGridColumnsColors()
{
    float mn = Grid->GetMinHeight();
    float mx = Grid->GetMaxHeight();

    bool bColorByGoal = bUseRandomGoals;
    int32 numGoals = GoalColors.Num();

    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    {
        float h = Grid->GetColumnHeight(c);
        float ratio = FMath::GetMappedRangeValueClamped(FVector2D(mn, mx), FVector2D(0.f, 1.f), h);

        FLinearColor finalCol = FLinearColor::LerpUsingHSV(FLinearColor::Black, FLinearColor::White, ratio);

        if (bColorByGoal)
        {
            FVector colCenter = Grid->GetColumnWorldLocation(c);
            for (int32 g = 0; g < numGoals; g++)
            {
                if (GoalManager->IsInRadiusOf(g, colCenter, ColumnRadius))
                {
                    finalCol = GoalColors[g];
                    break;
                }
            }
        }
        Grid->SetColumnColor(c, finalCol);
    }
}

// --------------------------------------------------
//  Accessors
// --------------------------------------------------
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
    return ObjectGoalIndices.IsValidIndex(ObjIndex) ? ObjectGoalIndices[ObjIndex] : -1;
}

FVector UStateManager::GetCurrentVelocity(int32 ObjIndex) const
{
    return CurrVel.IsValidIndex(ObjIndex) ? CurrVel[ObjIndex] : FVector::ZeroVector;
}
FVector UStateManager::GetPreviousVelocity(int32 ObjIndex) const
{
    return PrevVel.IsValidIndex(ObjIndex) ? PrevVel[ObjIndex] : FVector::ZeroVector;
}

float UStateManager::GetCurrentDistance(int32 ObjIndex) const
{
    return CurrDist.IsValidIndex(ObjIndex) ? CurrDist[ObjIndex] : -1.f;
}
float UStateManager::GetPreviousDistance(int32 ObjIndex) const
{
    return PrevDist.IsValidIndex(ObjIndex) ? PrevDist[ObjIndex] : -1.f;
}

// -------------- Helpers --------------
FVector UStateManager::GetColumnTopWorldLocation(int32 GridX, int32 GridY) const
{
    checkf(Grid, TEXT("GetColumnTopWorldLocation => Grid is null!"));

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
    return PlatformCenter + FVector(0.f, 0.f, 100.f);
}

FVector UStateManager::GenerateRandomGridLocation() const
{
    checkf(Grid, TEXT("GenerateRandomGridLocation => Grid is null!"));

    static const int32 MaxSpawnAttempts = 3;
    bool bAnyActive = bHasActive.Contains(true);

    for (int32 attempt = 0; attempt < MaxSpawnAttempts; attempt++)
    {
        if (!bAnyActive)
        {
            int32 xStart = FMath::Clamp(MarginCells, 0, GridSize - 1);
            int32 xEnd = FMath::Clamp(GridSize - MarginCells, 0, GridSize - 1);
            if (xStart > xEnd)
            {
                return PlatformCenter + FVector(0, 0, 100.f);
            }
            int32 chosenX = FMath::RandRange(xStart, xEnd);
            int32 chosenY = FMath::RandRange(xStart, xEnd);
            return GetColumnTopWorldLocation(chosenX, chosenY);
        }
        else
        {
            // occupancy NxN => 0=free
            FMatrix2D occ(GridSize, GridSize, 0.f);

            for (int32 iObj = 0; iObj < bHasActive.Num(); iObj++)
            {
                if (!bHasActive[iObj]) continue;
                AGridObject* Obj = ObjectMgr->GetGridObject(iObj);

                float sphRadius = Obj->MeshComponent->Bounds.SphereRadius
                    * Obj->GetActorScale3D().GetMax();
                FVector oLoc = Obj->GetObjectLocation();

                float half = PlatformWorldSize.X * 0.5f;
                float gx0 = PlatformCenter.X - half;
                float gy0 = PlatformCenter.Y - half;

                FVector minPt = oLoc - FVector(sphRadius, sphRadius, 0.f);
                FVector maxPt = oLoc + FVector(sphRadius, sphRadius, 0.f);

                int32 minX = FMath::FloorToInt((minPt.X - gx0) / CellSize);
                int32 maxX = FMath::FloorToInt((maxPt.X - gx0) / CellSize);
                int32 minY = FMath::FloorToInt((minPt.Y - gy0) / CellSize);
                int32 maxY = FMath::FloorToInt((maxPt.Y - gy0) / CellSize);

                minX = FMath::Clamp(minX, 0, GridSize - 1);
                maxX = FMath::Clamp(maxX, 0, GridSize - 1);
                minY = FMath::Clamp(minY, 0, GridSize - 1);
                maxY = FMath::Clamp(maxY, 0, GridSize - 1);

                for (int gx = minX; gx <= maxX; gx++)
                {
                    for (int gy = minY; gy <= maxY; gy++)
                    {
                        float cellWX = gx0 + (gx + 0.5f) * CellSize;
                        float cellWY = gy0 + (gy + 0.5f) * CellSize;
                        float d2d = FVector::Dist2D(oLoc, FVector(cellWX, cellWY, oLoc.Z));
                        if (d2d <= sphRadius)
                        {
                            occ[gx][gy] = 1.f;
                        }
                    }
                }
            }

            TArray<int32> freeCells;
            freeCells.Reserve(GridSize * GridSize);
            int32 xs = FMath::Clamp(MarginCells, 0, GridSize - 1);
            int32 xe = FMath::Clamp(GridSize - MarginCells, 0, GridSize - 1);
            if (xs > xe)
            {
                UE_LOG(LogTemp, Warning, TEXT("Margin too large => fallback center."));
                return PlatformCenter + FVector(0, 0, 100.f);
            }

            for (int gx = xs; gx <= xe; gx++)
            {
                for (int gy = xs; gy <= xe; gy++)
                {
                    if (occ[gx][gy] < 0.5f)
                    {
                        freeCells.Add(gx * GridSize + gy);
                    }
                }
            }

            if (freeCells.Num() == 0)
            {
                continue;
            }
            int32 chosen = freeCells[FMath::RandRange(0, freeCells.Num() - 1)];
            int32 cX = chosen / GridSize;
            int32 cY = chosen % GridSize;
            return GetColumnTopWorldLocation(cX, cY);
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("No free cell => fallback center."));
    return PlatformCenter + FVector(0, 0, 100.f);
}

void UStateManager::SetupOverheadCamera()
{
    checkf(Platform, TEXT("SetupOverheadCamera => Platform is null!"));

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
        UE_LOG(LogTemp, Error, TEXT("Failed spawn overhead camera."));
        return;
    }

    OverheadRenderTarget = NewObject<UTextureRenderTarget2D>();
    OverheadRenderTarget->RenderTargetFormat = RTF_RGBA8;   // or RTF_RGBA16f
    OverheadRenderTarget->InitAutoFormat(OverheadCamResX, OverheadCamResY);

    USceneCaptureComponent2D* comp = OverheadCaptureActor->GetCaptureComponent2D();
    comp->ProjectionType = ECameraProjectionMode::Perspective;
    comp->FOVAngle = OverheadCamFOV;
    comp->TextureTarget = OverheadRenderTarget;
    comp->CaptureSource = ESceneCaptureSource::SCS_BaseColor;
    comp->bCaptureEveryFrame = false;
    comp->bCaptureOnMovement = false;
}

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