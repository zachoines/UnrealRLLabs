#include "TerraShift/StateManager.h"
#include "TerraShift/GridObject.h"
#include "TerraShift/GoalPlatform.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Kismet/GameplayStatics.h"

/** Helper static for potential config error logging */
static float SM_GetOrErrorNumber(UEnvironmentConfig* Cfg, const FString& Path)
{
    if (!Cfg || !Cfg->HasPath(*Path))
    {
        UE_LOG(LogTemp, Error, TEXT("StateManager config missing path: %s"), *Path);
        return 0.f;
    }
    return Cfg->Get(*Path)->AsNumber();
}
static int32 SM_GetOrErrorInt(UEnvironmentConfig* Cfg, const FString& Path)
{
    if (!Cfg || !Cfg->HasPath(*Path))
    {
        UE_LOG(LogTemp, Error, TEXT("StateManager config missing path: %s"), *Path);
        return 0;
    }
    return Cfg->Get(*Path)->AsInt();
}

void UStateManager::InitializeFromConfig(UEnvironmentConfig* SMConfig)
{
    if (!SMConfig || !SMConfig->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::InitializeFromConfig => null or invalid config!"));
        return;
    }
    LoadConfig(SMConfig);
}

void UStateManager::LoadConfig(UEnvironmentConfig* Config)
{
    MaxGridObjects = Config->GetOrDefaultInt(TEXT("max_grid_objects"), 50);
    GoalThreshold = Config->GetOrDefaultNumber(TEXT("GoalThreshold"), 1.75f);
    MarginXY = Config->GetOrDefaultNumber(TEXT("MarginXY"), 1.5f);
    MinZ = Config->GetOrDefaultNumber(TEXT("MinZ"), 0.f);
    MaxZ = Config->GetOrDefaultNumber(TEXT("MaxZ"), 1000.f);
    SpawnCollisionRadius = Config->GetOrDefaultNumber(TEXT("SpawnCollisionRadius"), 100.f);
    MarginCells = Config->GetOrDefaultInt(TEXT("MarginCells"), 4);
    BoundingSphereScale = Config->GetOrDefaultNumber(TEXT("BoundingSphereScale"), 1.5f);
    ObjectScale = Config->GetOrDefaultNumber(TEXT("ObjectScale"), 0.1f);
    ObjectMass = Config->GetOrDefaultNumber(TEXT("ObjectMass"), 0.1f);
    MaxColumnHeight = Config->GetOrDefaultNumber(TEXT("max_column_height"), 4.f);
    bUseRaycastForHeight = Config->GetOrDefaultBool(TEXT("bUseRaycastForHeight"), false);
    BaseRespawnDelay = Config->GetOrDefaultNumber(TEXT("base_respawn_delay"), 0.f);
}

void UStateManager::SetReferences(
    AMainPlatform* InPlatform,
    AGridObjectManager* InObjMgr,
    AGrid* InGrid,
    const TArray<AGoalPlatform*>& InGoalPlatforms,
    UMultiAgentGaussianWaveHeightMap* InWaveSim
)
{
    Platform = InPlatform;
    ObjectMgr = InObjMgr;
    Grid = InGrid;
    GoalPlatforms = InGoalPlatforms;
    WaveSim = InWaveSim;

    if (!Platform || !ObjectMgr || !Grid || !WaveSim)
    {
        UE_LOG(LogTemp, Error, TEXT("SetReferences => some references are null!"));
        return;
    }

    // infer grid size from # columns
    GridSize = (Grid->GetTotalColumns() > 0)
        ? FMath::FloorToInt(FMath::Sqrt((float)Grid->GetTotalColumns()))
        : 50;

    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
        * 2.f
        * Platform->GetActorScale3D();

    PlatformCenter = Platform->GetActorLocation();
    CellSize = (float)PlatformWorldSize.X / (float)GridSize;
}

void UStateManager::Reset(int32 NumObjects)
{
    if (!Platform || !ObjectMgr || !Grid || !WaveSim)
    {
        UE_LOG(LogTemp, Error, TEXT("UStateManager::Reset => references missing!"));
        return;
    }

    // allocate arrays
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

        if (GoalPlatforms.Num() > 0)
            ObjectGoalIndices[i] = FMath::RandRange(0, GoalPlatforms.Num() - 1);
        else
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
        RespawnDelays[i] = BaseRespawnDelay;
    }

    // NxN channels => init to 0
    ChannelHeight = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelVelX = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelVelY = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelVelZ = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelAccX = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelAccY = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelAccZ = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelDirX = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelDirY = FMatrix2D(GridSize, GridSize, 0.f);
    ChannelDirZ = FMatrix2D(GridSize, GridSize, 0.f);
}

int32 UStateManager::GetMaxGridObjects() const
{
    return MaxGridObjects;
}

void UStateManager::UpdateGridObjectFlags()
{
    if (!Platform || !ObjectMgr || !Grid)
    {
        UE_LOG(LogTemp, Error, TEXT("UpdateGridObjectFlags => references missing!"));
        return;
    }

    float halfX = PlatformWorldSize.X * 0.5f;
    float halfY = PlatformWorldSize.Y * 0.5f;

    float minZLocal = PlatformCenter.Z + MinZ;
    float maxZLocal = PlatformCenter.Z + MaxZ;

    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj)
            continue;

        FVector wPos = Obj->GetObjectLocation();
        FVector Ext = Obj->MeshComponent->Bounds.BoxExtent;

        bool ShouldRespawnNow = false;

        // (1) check if object reached its goal
        if (!bHasReached[i])
        {
            int32 gIdx = ObjectGoalIndices[i];
            if (GoalPlatforms.IsValidIndex(gIdx))
            {
                AGoalPlatform* gp = GoalPlatforms[gIdx];
                FVector gpPos = gp->GetActorLocation();
                float dist = FVector::Dist(wPos, gpPos);
                // if within threshold
                if (dist <= (Ext.GetAbsMax() * GoalThreshold))
                {
                    bHasActive[i] = false;
                    bHasReached[i] = true;
                    bShouldCollect[i] = true;
                    ShouldRespawnNow = false;
                    ObjectMgr->DisableGridObject(i);
                }
            }
        }

        // (2) check bounding
        if (!bFallenOff[i] && !bHasReached[i])
        {
            float dx = FMath::Abs(wPos.X - PlatformCenter.X);
            float dy = FMath::Abs(wPos.Y - PlatformCenter.Y);
            float zPos = wPos.Z;

            bool bOutOfBounds = false;
            if (dx > (halfX + MarginXY) || dy > (halfY + MarginXY))
            {
                bOutOfBounds = true;
            }
            else if (zPos < minZLocal || zPos > maxZLocal)
            {
                bOutOfBounds = true;
            }

            if (bOutOfBounds)
            {
                bHasActive[i] = false;
                bFallenOff[i] = true;
                bShouldCollect[i] = true;
                ShouldRespawnNow = false;
                ObjectMgr->DisableGridObject(i);
            }
        }

        // If we flagged "ShouldRespawnNow", we set the array
        if (ShouldRespawnNow)
        {
            bShouldResp[i] = true;
            ObjectMgr->DisableGridObject(i);
        }
    }
}

void UStateManager::UpdateObjectStats(float DeltaTime)
{
    if (!Platform || !ObjectMgr)
    {
        UE_LOG(LogTemp, Error, TEXT("UpdateObjectStats => references missing!"));
        return;
    }

    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
        {
            // keep them zeroed-out
            PrevVel[i] = FVector::ZeroVector;
            CurrVel[i] = FVector::ZeroVector;
            PrevAcc[i] = FVector::ZeroVector;
            CurrAcc[i] = FVector::ZeroVector;
            PrevDist[i] = -1.f;
            CurrDist[i] = -1.f;
            PrevPos[i] = FVector::ZeroVector;
            CurrPos[i] = FVector::ZeroVector;

            // increment timer if we are waiting to respawn
            if (bShouldResp[i])
            {
                RespawnTimer[i] += DeltaTime;
            }
            continue;
        }

        // get this object
        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj)
            continue;

        FVector wVel = Obj->MeshComponent->GetPhysicsLinearVelocity();
        FVector wPos = Obj->GetObjectLocation();

        int32 gIdx = ObjectGoalIndices[i];
        FVector wGoal = (GoalPlatforms.IsValidIndex(gIdx)
            ? GoalPlatforms[gIdx]->GetActorLocation()
            : FVector::ZeroVector);

        // localize
        FVector locVel = Platform->GetActorTransform().InverseTransformVector(wVel);
        FVector locPos = Platform->GetActorTransform().InverseTransformPosition(wPos);
        FVector locGoal = Platform->GetActorTransform().InverseTransformPosition(wGoal);

        // shift old => prev
        PrevVel[i] = CurrVel[i];
        PrevAcc[i] = CurrAcc[i];
        PrevDist[i] = CurrDist[i];
        PrevPos[i] = CurrPos[i];

        // set curr
        CurrVel[i] = locVel;
        CurrAcc[i] = (DeltaTime > 1e-6f)
            ? (locVel - PrevVel[i]) / DeltaTime
            : FVector::ZeroVector;

        CurrPos[i] = locPos;
        CurrDist[i] = FVector::Dist(locPos, locGoal);

        // increment respawn timer if flagged
        if (bShouldResp[i])
        {
            RespawnTimer[i] += DeltaTime;
        }
    }
}

void UStateManager::RespawnGridObjects()
{
    if (!ObjectMgr)
    {
        UE_LOG(LogTemp, Error, TEXT("RespawnGridObjects => no object manager!"));
        return;
    }

    for (int32 i = 0; i < bShouldResp.Num(); i++)
    {
        if (bShouldResp[i] && RespawnTimer[i] >= RespawnDelays[i])
        {
            // Attempt to find a random location
            FVector spawnLoc = GenerateRandomGridLocation();
            if (GoalPlatforms.Num() > 0)
            {
                ObjectGoalIndices[i] = FMath::RandRange(0, GoalPlatforms.Num() - 1);
            }

            ObjectMgr->SpawnGridObjectAtIndex(i, spawnLoc, FVector(ObjectScale), ObjectMass);

            // color to match assigned goal
            if (GoalPlatforms.IsValidIndex(ObjectGoalIndices[i]))
            {
                AGridObject* newObj = ObjectMgr->GetGridObject(i);
                if (newObj)
                {
                    FLinearColor col = GoalPlatforms[ObjectGoalIndices[i]]->GetGoalColor();
                    newObj->SetGridObjectColor(col);
                }
            }

            // reset flags
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
    // if any object is active or about to respawn => not done
    if (bHasActive.Contains(true)) return false;
    if (bShouldResp.Contains(true)) return false;
    return true;
}

void UStateManager::BuildCentralState()
{
    if (!WaveSim || !Grid)
    {
        UE_LOG(LogTemp, Error, TEXT("BuildCentralState => missing wave or grid!"));
        return;
    }
    const FMatrix2D& wave = WaveSim->GetHeightMap();
    if (wave.GetNumRows() != GridSize || wave.GetNumColumns() != GridSize)
    {
        UE_LOG(LogTemp, Warning, TEXT("Wave size mismatch => cannot build central."));
        return;
    }
    // base height = wave
    ChannelHeight = wave;

    // zero velocity/acc/dir
    ChannelVelX.Init(0.f);
    ChannelVelY.Init(0.f);
    ChannelVelZ.Init(0.f);

    ChannelAccX.Init(0.f);
    ChannelAccY.Init(0.f);
    ChannelAccZ.Init(0.f);

    ChannelDirX.Init(0.f);
    ChannelDirY.Init(0.f);
    ChannelDirZ.Init(0.f);

    // For each active object => overlay velocity,acc,dir, possibly raise height
    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj)
            continue;

        FVector wPos = Obj->GetObjectLocation();
        FVector wVel = Obj->MeshComponent->GetPhysicsLinearVelocity();
        float rad = Obj->MeshComponent->Bounds.SphereRadius * ObjectScale;

        // local to grid
        FVector localPos = Grid->GetActorTransform().InverseTransformPosition(wPos);
        FVector localVel = Grid->GetActorTransform().InverseTransformVector(wVel);
        FVector localAcc = CurrAcc.IsValidIndex(i) ? CurrAcc[i] : FVector::ZeroVector;

        int32 gIdx = ObjectGoalIndices[i];
        FVector wGoal = (GoalPlatforms.IsValidIndex(gIdx)
            ? GoalPlatforms[gIdx]->GetActorLocation()
            : FVector::ZeroVector);
        FVector localGoal = Grid->GetActorTransform().InverseTransformPosition(wGoal);
        FVector dir = (localGoal - localPos).GetSafeNormal();

        // bounding box in grid coords
        int32 minX = FMath::FloorToInt((localPos.X - rad) / CellSize);
        int32 maxX = FMath::FloorToInt((localPos.X + rad) / CellSize);
        int32 minY = FMath::FloorToInt((localPos.Y - rad) / CellSize);
        int32 maxY = FMath::FloorToInt((localPos.Y + rad) / CellSize);

        minX = FMath::Clamp(minX, 0, GridSize - 1);
        maxX = FMath::Clamp(maxX, 0, GridSize - 1);
        minY = FMath::Clamp(minY, 0, GridSize - 1);
        maxY = FMath::Clamp(maxY, 0, GridSize - 1);

        for (int32 rx = minX; rx <= maxX; rx++)
        {
            for (int32 ry = minY; ry <= maxY; ry++)
            {
                float ccx = (rx + 0.5f) * CellSize;
                float ccy = (ry + 0.5f) * CellSize;
                float dx = ccx - localPos.X;
                float dy = ccy - localPos.Y;
                float dist2 = (dx * dx + dy * dy);
                if (dist2 > rad * rad)
                    continue;

                // overlay velocity,acc,dir
                ChannelVelX[rx][ry] = localVel.X;
                ChannelVelY[rx][ry] = localVel.Y;
                ChannelVelZ[rx][ry] = localVel.Z;

                ChannelAccX[rx][ry] = localAcc.X;
                ChannelAccY[rx][ry] = localAcc.Y;
                ChannelAccZ[rx][ry] = localAcc.Z;

                ChannelDirX[rx][ry] = dir.X;
                ChannelDirY[rx][ry] = dir.Y;
                ChannelDirZ[rx][ry] = dir.Z;

                float oldH = ChannelHeight[rx][ry];
                float newH = oldH;

                // if bUseRaycastForHeight => line trace from above
                if (bUseRaycastForHeight)
                {
                    float cellWX = Grid->GetActorLocation().X + (ccx * Grid->GetActorScale3D().X);
                    float cellWY = Grid->GetActorLocation().Y + (ccy * Grid->GetActorScale3D().Y);
                    float cellWZ = Grid->GetActorLocation().Z + (MaxColumnHeight * 2.f);

                    FVector cellWorld(cellWX, cellWY, cellWZ);
                    newH = RaycastColumnTopWorld(cellWorld, oldH);
                }
                else
                {
                    // approximate top => localPos.Z + sqrt(r^2 - dist2)
                    float sphereTopZ = localPos.Z + FMath::Sqrt(FMath::Max(0.f, rad * rad - dist2));
                    newH = FMath::Max(oldH, sphereTopZ);
                }

                ChannelHeight[rx][ry] = newH;
            }
        }
    }
}

float UStateManager::RaycastColumnTopWorld(const FVector& CellWorldCenter, float waveVal) const
{
    if (!Grid)
        return waveVal;

    UWorld* w = Grid->GetWorld();
    if (!w)
        return waveVal;

    FVector start = CellWorldCenter;
    FVector end = start - FVector(0, 0, MaxColumnHeight * 5.f);

    FHitResult hit;
    FCollisionQueryParams qParams(FName(TEXT("HeightRay")), true);

    bool bHit = w->LineTraceSingleByChannel(hit, start, end, ECC_Visibility, qParams);
    if (!bHit)
    {
        return waveVal; // no collision => fallback
    }
    // localize
    FVector localHit = Grid->GetActorTransform().InverseTransformPosition(hit.Location);
    return FMath::Max(waveVal, localHit.Z);
}

TArray<float> UStateManager::GetCentralState() const
{
    TArray<float> outArr;
    outArr.Reserve(GridSize * GridSize * 10);

    for (int32 r = 0; r < GridSize; r++)
    {
        for (int32 c = 0; c < GridSize; c++)
        {
            outArr.Add(ChannelHeight[r][c]);
            outArr.Add(ChannelVelX[r][c]);
            outArr.Add(ChannelVelY[r][c]);
            outArr.Add(ChannelVelZ[r][c]);
            outArr.Add(ChannelAccX[r][c]);
            outArr.Add(ChannelAccY[r][c]);
            outArr.Add(ChannelAccZ[r][c]);
            outArr.Add(ChannelDirX[r][c]);
            outArr.Add(ChannelDirY[r][c]);
            outArr.Add(ChannelDirZ[r][c]);
        }
    }
    return outArr;
}

TArray<float> UStateManager::GetAgentState(int32 AgentIndex) const
{
    // simply forward to WaveSim->GetAgentState if valid
    if (!WaveSim)
        return TArray<float>();

    return WaveSim->GetAgentState(AgentIndex);
}

// ----------- Accessors --------------

bool UStateManager::GetHasActive(int32 i) const
{
    return bHasActive.IsValidIndex(i) ? bHasActive[i] : false;
}
void UStateManager::SetHasActive(int32 i, bool bVal)
{
    if (bHasActive.IsValidIndex(i))
        bHasActive[i] = bVal;
}

bool UStateManager::GetHasReachedGoal(int32 i) const
{
    return bHasReached.IsValidIndex(i) ? bHasReached[i] : false;
}
void UStateManager::SetHasReachedGoal(int32 i, bool bVal)
{
    if (bHasReached.IsValidIndex(i))
        bHasReached[i] = bVal;
}

bool UStateManager::GetHasFallenOff(int32 i) const
{
    return bFallenOff.IsValidIndex(i) ? bFallenOff[i] : false;
}
void UStateManager::SetHasFallenOff(int32 i, bool bVal)
{
    if (bFallenOff.IsValidIndex(i))
        bFallenOff[i] = bVal;
}

bool UStateManager::GetShouldCollectReward(int32 i) const
{
    return bShouldCollect.IsValidIndex(i) ? bShouldCollect[i] : false;
}
void UStateManager::SetShouldCollectReward(int32 i, bool bVal)
{
    if (bShouldCollect.IsValidIndex(i))
        bShouldCollect[i] = bVal;
}

bool UStateManager::GetShouldRespawn(int32 i) const
{
    return bShouldResp.IsValidIndex(i) ? bShouldResp[i] : false;
}
void UStateManager::SetShouldRespawn(int32 i, bool bVal)
{
    if (bShouldResp.IsValidIndex(i))
        bShouldResp[i] = bVal;
}

int32 UStateManager::GetGoalIndex(int32 i) const
{
    return ObjectGoalIndices.IsValidIndex(i) ? ObjectGoalIndices[i] : -1;
}
void UStateManager::SetGoalIndex(int32 i, int32 Goal)
{
    if (ObjectGoalIndices.IsValidIndex(i))
        ObjectGoalIndices[i] = Goal;
}

FVector UStateManager::GetCurrentVelocity(int32 i) const
{
    return CurrVel.IsValidIndex(i) ? CurrVel[i] : FVector::ZeroVector;
}
void UStateManager::SetCurrentVelocity(int32 i, FVector val)
{
    if (CurrVel.IsValidIndex(i))
        CurrVel[i] = val;
}

FVector UStateManager::GetPreviousVelocity(int32 i) const
{
    return PrevVel.IsValidIndex(i) ? PrevVel[i] : FVector::ZeroVector;
}
void UStateManager::SetPreviousVelocity(int32 i, FVector val)
{
    if (PrevVel.IsValidIndex(i))
        PrevVel[i] = val;
}

FVector UStateManager::GetCurrentAcceleration(int32 i) const
{
    return CurrAcc.IsValidIndex(i) ? CurrAcc[i] : FVector::ZeroVector;
}
void UStateManager::SetCurrentAcceleration(int32 i, FVector val)
{
    if (CurrAcc.IsValidIndex(i))
        CurrAcc[i] = val;
}

FVector UStateManager::GetPreviousAcceleration(int32 i) const
{
    return PrevAcc.IsValidIndex(i) ? PrevAcc[i] : FVector::ZeroVector;
}
void UStateManager::SetPreviousAcceleration(int32 i, FVector val)
{
    if (PrevAcc.IsValidIndex(i))
        PrevAcc[i] = val;
}

float UStateManager::GetCurrentDistance(int32 i) const
{
    return CurrDist.IsValidIndex(i) ? CurrDist[i] : -1.f;
}
void UStateManager::SetCurrentDistance(int32 i, float val)
{
    if (CurrDist.IsValidIndex(i))
        CurrDist[i] = val;
}

float UStateManager::GetPreviousDistance(int32 i) const
{
    return PrevDist.IsValidIndex(i) ? PrevDist[i] : -1.f;
}
void UStateManager::SetPreviousDistance(int32 i, float val)
{
    if (PrevDist.IsValidIndex(i))
        PrevDist[i] = val;
}

FVector UStateManager::GetCurrentPosition(int32 i) const
{
    return CurrPos.IsValidIndex(i) ? CurrPos[i] : FVector::ZeroVector;
}
void UStateManager::SetCurrentPosition(int32 i, FVector val)
{
    if (CurrPos.IsValidIndex(i))
        CurrPos[i] = val;
}

FVector UStateManager::GetPreviousPosition(int32 i) const
{
    return PrevPos.IsValidIndex(i) ? PrevPos[i] : FVector::ZeroVector;
}
void UStateManager::SetPreviousPosition(int32 i, FVector val)
{
    if (PrevPos.IsValidIndex(i))
        PrevPos[i] = val;
}

// --------------------------------------------------
//   Collision-based random spawn
// --------------------------------------------------
FVector UStateManager::GenerateRandomGridLocation() const
{
    if (!Grid)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateRandomGridLocation => no Grid => fallback center."));
        return PlatformCenter + FVector(0, 0, 100.f);
    }

    // We do multiple attempts with the collision-dist matrix approach
    static const int32 MaxSpawnAttempts = 5;

    for (int32 attempt = 0; attempt < MaxSpawnAttempts; attempt++)
    {
        // build distance matrix
        FMatrix2D distMat = ComputeCollisionDistanceMatrix();

        TArray<int32> freeCells;
        freeCells.Reserve(GridSize * GridSize);

        // skip margin cells
        int32 xStart = FMath::Clamp(MarginCells, 0, GridSize - 1);
        int32 xEnd = FMath::Clamp(GridSize - MarginCells, 0, GridSize - 1);
        int32 yStart = xStart;
        int32 yEnd = xEnd;

        for (int32 X = xStart; X <= xEnd; X++)
        {
            for (int32 Y = yStart; Y <= yEnd; Y++)
            {
                float dd = distMat[X][Y];
                // must have enough clearance
                if (dd > SpawnCollisionRadius)
                {
                    freeCells.Add(X * GridSize + Y);
                }
            }
        }

        if (freeCells.Num() == 0)
        {
            // No free cell found => continue to next attempt
            continue;
        }
        // pick random cell
        int32 chosen = freeCells[FMath::RandRange(0, freeCells.Num() - 1)];
        int32 cX = chosen / GridSize;
        int32 cY = chosen % GridSize;

        float cWX = (cX + 0.5f) * CellSize + (PlatformCenter.X - 0.5f * PlatformWorldSize.X);
        float cWY = (cY + 0.5f) * CellSize + (PlatformCenter.Y - 0.5f * PlatformWorldSize.Y);
        float spawnZ = Grid->GetActorLocation().Z + (CellSize * 2) * MaxColumnHeight;

        return FVector(cWX, cWY, spawnZ);
    }

    // if we fail all attempts => fallback
    UE_LOG(LogTemp, Warning, TEXT("No free cell found after multiple attempts => fallback center."));
    return PlatformCenter + FVector(0, 0, 100.f);
}

// we compute the distance to the nearest active object for each cell
FMatrix2D UStateManager::ComputeCollisionDistanceMatrix() const
{
    FMatrix2D distMat(GridSize, GridSize, 1e6f);
    if (!ObjectMgr)
        return distMat;

    // for each active => fill
    for (int32 i = 0; i < bHasActive.Num(); i++)
    {
        if (!bHasActive[i])
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj)
            continue;

        FVector wPos = Obj->GetObjectLocation();
        // incorporate object scale, bounding
        FVector objScale = Obj->GetActorScale3D();
        float sphereRadius = Obj->MeshComponent->Bounds.SphereRadius
            * BoundingSphereScale
            * objScale.GetAbsMax();

        float half = PlatformWorldSize.X * 0.5f;
        float gridOx = PlatformCenter.X - half;
        float gridOy = PlatformCenter.Y - half;

        FVector minC = wPos - FVector(sphereRadius, sphereRadius, 0.f);
        FVector maxC = wPos + FVector(sphereRadius, sphereRadius, 0.f);

        int32 minX = FMath::FloorToInt((minC.X - gridOx) / CellSize);
        int32 maxX = FMath::FloorToInt((maxC.X - gridOx) / CellSize);
        int32 minY = FMath::FloorToInt((minC.Y - gridOy) / CellSize);
        int32 maxY = FMath::FloorToInt((maxC.Y - gridOy) / CellSize);

        minX = FMath::Clamp(minX, 0, GridSize - 1);
        maxX = FMath::Clamp(maxX, 0, GridSize - 1);
        minY = FMath::Clamp(minY, 0, GridSize - 1);
        maxY = FMath::Clamp(maxY, 0, GridSize - 1);

        // fill matrix
        for (int32 xx = minX; xx <= maxX; xx++)
        {
            for (int32 yy = minY; yy <= maxY; yy++)
            {
                float cX = gridOx + (xx + 0.5f) * CellSize;
                float cY = gridOy + (yy + 0.5f) * CellSize;
                float dd = FVector::Dist2D(wPos, FVector(cX, cY, wPos.Z));
                if (dd < distMat[xx][yy])
                {
                    distMat[xx][yy] = dd;
                }
            }
        }
    }
    return distMat;
}