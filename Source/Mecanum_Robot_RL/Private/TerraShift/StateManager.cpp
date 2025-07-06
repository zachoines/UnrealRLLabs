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
        RespawnDelays[i] = 0.0f;
        if (BaseRespawnDelay > 0.0f)
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

            float halfZ = col->ColumnMesh->Bounds.BoxExtent.Z;
            float objWorldRadius = ObjectUnscaledSize * ObjectScale;
            FVector offset(0.f, 0.f, halfZ + objWorldRadius);

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
        if (ObjectSlotStates[i] != EObjectSlotState::Active)
            continue;

        AGridObject* Obj = ObjectMgr->GetGridObject(i);
        if (!Obj) continue;

        FVector wPos = Obj->GetObjectLocation();

        // (1) Check if object reached its goal
        int32 gIdx = ObjectGoalIndices[i];
        if (gIdx >= 0)
        {
            bool bInRadius = GoalManager->IsInRadiusOf(gIdx, wPos, GoalCollectRadius);
            if (bInRadius)
            {
                ObjectSlotStates[i] = EObjectSlotState::GoalReached;
                bShouldCollect[i] = true;
                bShouldResp[i] = bRespawnOnGoal; // Use the new toggle

                if (!bShouldResp[i])
                {
                    ObjectMgr->DisableGridObject(i);
                    OccupancyGrid->RemoveObject(i, FName("GridObjects"));
                }
                continue;
            }
        }

        // (2) OOB Check
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

// ------------------------------------------
//   UpdateObjectStats
// ------------------------------------------
void UStateManager::UpdateObjectStats(float DeltaTime)
{
    for (int32 i = 0; i < MaxGridObjects; i++)
    {
        if (GetObjectSlotState(i) != EObjectSlotState::Active)
        {
            // Zero out stats for inactive/terminal objects
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
            bShouldCollect[i] = false;
            RespawnTimer[i] = 0.f;

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
        if (GetObjectSlotState(i) == EObjectSlotState::Active || GetObjectSlotState(i) == EObjectSlotState::Empty)
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
    if (bIncludeHeightMapInState) {
        FTransform GridTransform = Grid->GetActorTransform();
        float visMinZ = -MaxColumnHeight;
        float visMaxZ = MaxColumnHeight;
        float traceDistUp = FMath::Abs(MaxZ) + 100.0f;
        float traceDistDown = FMath::Abs(visMinZ) + 100.0f;

        for (int32 r_state = 0; r_state < CurrentStateMapH; ++r_state) {
            for (int32 c_state = 0; c_state < CurrentStateMapW; ++c_state) {
                float norm_x = (CurrentStateMapW > 1) ? static_cast<float>(c_state) / (CurrentStateMapW - 1) : 0.5f;
                float norm_y = (CurrentStateMapH > 1) ? static_cast<float>(r_state) / (CurrentStateMapH - 1) : 0.5f;
                float lx = (norm_x - 0.5f) * PlatformWorldSize.X;
                float ly = (norm_y - 0.5f) * PlatformWorldSize.Y;

                FVector wStart = GridTransform.TransformPosition(FVector(lx, ly, traceDistUp));
                FVector wEnd = GridTransform.TransformPosition(FVector(lx, ly, -traceDistDown));

                FHitResult hit;
                bool bHit = w->LineTraceSingleByChannel(hit, wStart, wEnd, ECC_Visibility);

                float finalLocalZ = bHit ? GridTransform.InverseTransformPosition(hit.ImpactPoint).Z : 0.f;
                float clampedVisZ = FMath::Clamp(finalLocalZ, visMinZ, visMaxZ);
                float norm_height = (visMaxZ > visMinZ) ? (clampedVisZ - visMinZ) / (visMaxZ - visMinZ) : 0.f;
                HeightTmp[r_state][c_state] = (norm_height * 2.f) - 1.f;
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
            if (GetObjectSlotState(i) == EObjectSlotState::Active)
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
    // ... (ShowFlags settings remain the same) ...
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