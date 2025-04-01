#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    // Our wave simulator (Gaussian wave).
    WaveSimulator = CreateDefaultSubobject<UMultiAgentGaussianWaveHeightMap>(TEXT("WaveSimulator"));

    // Our new State Manager for grid objects & central state.
    StateManager = CreateDefaultSubobject<UStateManager>(TEXT("StateManager"));

    CurrentStep = 0;
    Initialized = false;

    // By default, we can guess 1 wave agent; actual is set in ResetEnv.
    CurrentAgents = 1;

    // We'll set this once we parse "max_grid_objects" from the StateManager config:
    CurrentGridObjects = 1;

    GoalColors = {
        FLinearColor::Red,
        FLinearColor::Green,
        FLinearColor::Blue,
        FLinearColor::Yellow
    };
}

ATerraShiftEnvironment::~ATerraShiftEnvironment()
{
}

void ATerraShiftEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (Initialized)
    {
        // UpdateActiveColumns();
        UpdateColumnColors();
    }
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* Params)
{
    check(Params);
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(Params);
    check(TerraShiftParams);
    check(TerraShiftParams->EnvConfig);

    CurrentStep = 0;
    CurrentAgents = 1;  // wave-sim default
    CurrentGridObjects = 1; // state manager default (will set properly after config)

    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath);

    // read config
    UEnvironmentConfig* EnvConfig = TerraShiftParams->EnvConfig;

    auto GetOrErrorNumber = [&](const FString& Path, float defaultVal)->float
        {
            if (!EnvConfig->HasPath(*Path))
            {
                UE_LOG(LogTemp, Error, TEXT("TerraShift config missing path: %s"), *Path);
                return defaultVal;
            }
            return EnvConfig->Get(*Path)->AsNumber();
        };
    auto GetOrErrorInt = [&](const FString& Path, int32 defaultVal)->int32
        {
            if (!EnvConfig->HasPath(*Path))
            {
                UE_LOG(LogTemp, Error, TEXT("TerraShift config missing path: %s"), *Path);
                return defaultVal;
            }
            return EnvConfig->Get(*Path)->AsInt();
        };

    // environment/params
    PlatformSize = GetOrErrorNumber(TEXT("environment/params/PlatformSize"), 1.f);
    MaxColumnHeight = GetOrErrorNumber(TEXT("environment/params/MaxColumnHeight"), 4.f);

    // object size
    if (EnvConfig->HasPath(TEXT("environment/params/ObjectSize")))
    {
        TArray<float> arr = EnvConfig->Get(TEXT("environment/params/ObjectSize"))->AsArrayOfNumbers();
        if (arr.Num() == 3)
        {
            ObjectSize = FVector(arr[0], arr[1], arr[2]);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("ObjectSize array must have 3 floats => fallback (0.1,0.1,0.1)"));
            ObjectSize = FVector(0.1f);
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("No path environment/params/ObjectSize => fallback (0.1,0.1,0.1)"));
        ObjectSize = FVector(0.1f);
    }

    ObjectMass = GetOrErrorNumber(TEXT("environment/params/ObjectMass"), 0.1f);
    GridSize = GetOrErrorInt(TEXT("environment/params/GridSize"), 50);
    MaxSteps = GetOrErrorInt(TEXT("environment/params/MaxSteps"), 512);
    NumGoals = GetOrErrorInt(TEXT("environment/params/NumGoals"), 4);
    SpawnDelay = GetOrErrorNumber(TEXT("environment/params/SpawnDelay"), 0.25f);
    MaxAgents = GetOrErrorInt(TEXT("environment/params/MaxAgents"), 5);
    GoalThreshold = GetOrErrorNumber(TEXT("environment/params/GoalThreshold"), 1.75f);

    // place environment at location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // spawn platform
    Platform = SpawnPlatform(TerraShiftParams->Location);
    Platform->SetActorScale3D(FVector(PlatformSize));

    // compute geometry
    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
        * 2.f
        * Platform->GetActorScale3D();
    PlatformCenter = Platform->GetActorLocation();
    CellSize = PlatformWorldSize.X / (float)GridSize;

    // spawn grid
    {
        FVector GridLocation = PlatformCenter + FVector(0.f, 0.f, MaxColumnHeight);
        FActorSpawnParameters sp;
        sp.Owner = this;
        Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator, sp);
        if (Grid)
        {
            Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepRelativeTransform);
            Grid->SetColumnMovementBounds(-MaxColumnHeight, MaxColumnHeight);
            Grid->InitializeGrid(GridSize, PlatformWorldSize.X, GridLocation);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to spawn Grid actor!"));
        }
    }

    // spawn object manager
    {
        GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
        if (GridObjectManager)
        {
            GridObjectManager->SetPlatformActor(Platform);
            GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + "/GridObjectManager")));
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to spawn GridObjectManager!"));
        }
    }

    // wave simulator config => environment/params/MultiAgentGaussianWaveHeightMap
    {
        UEnvironmentConfig* waveCfg = EnvConfig->Get(TEXT("environment/params/MultiAgentGaussianWaveHeightMap"));
        if (!waveCfg)
        {
            UE_LOG(LogTemp, Error, TEXT("Missing environment/params/MultiAgentGaussianWaveHeightMap in config!"));
        }
        else if (WaveSimulator)
        {
            WaveSimulator->InitializeFromConfig(waveCfg);
        }
    }

    // state manager => environment/params/StateManager
    {
        UEnvironmentConfig* smCfg = EnvConfig->Get(TEXT("environment/params/StateManager"));
        if (!smCfg)
        {
            UE_LOG(LogTemp, Error, TEXT("Missing environment/params/StateManager in config!"));
        }
        else if (StateManager)
        {
            StateManager->InitializeFromConfig(smCfg);

            // read out the max_grid_objects from the manager’s config
            // so we can store an initial guess. The actual usage is in ResetEnv
            int32 MaxGridObjs = smCfg->GetOrDefaultInt(TEXT("max_grid_objects"), 5);
            CurrentGridObjects = MaxGridObjs;  // default
        }
    }

    // finalize references
    if (StateManager)
    {
        TArray<AGoalPlatform*> tmpGoals; // updated properly in Reset
        StateManager->SetReferences(Platform, GridObjectManager, Grid, tmpGoals, WaveSimulator);
    }

    Initialized = true;
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents; // wave-sim agent count

    // (1) read the manager’s max_grid_objects or keep existing
    int32 ManagerMax = (StateManager ? StateManager->GetMaxGridObjects() : 5);
    CurrentGridObjects = ManagerMax; // or some logic if you want fewer

    // reset columns & objects
    if (Grid) Grid->ResetGrid();
    if (GridObjectManager) GridObjectManager->ResetGridObjects();

    // wave-sim reset for RL wave agents
    if (WaveSimulator) WaveSimulator->Reset(CurrentAgents);

    // clear old goals
    for (AGoalPlatform* gp : GoalPlatforms)
    {
        if (gp) gp->Destroy();
    }
    GoalPlatforms.Empty();

    // create new goals
    for (int32 i = 0; i < NumGoals; i++)
    {
        UpdateGoal(i);
    }

    // pass final references to state manager
    if (StateManager)
    {
        StateManager->SetReferences(Platform, GridObjectManager, Grid, GoalPlatforms, WaveSimulator);

        // Here is the big difference: we pass CurrentGridObjects
        StateManager->Reset(CurrentGridObjects);
    }

    return State();
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    if (!WaveSimulator)
    {
        UE_LOG(LogTemp, Warning, TEXT("Act => wave sim is null."));
        return;
    }
    // pass to wave sim (the RL wave-agents’ actions)
    WaveSimulator->Step(Action.Values, GetWorld()->GetDeltaSeconds());

    // apply final wave to columns
    const FMatrix2D& wave = WaveSimulator->GetHeightMap();
    if (Grid)
    {
        Grid->UpdateColumnHeights(wave);
    }
}

void ATerraShiftEnvironment::PostTransition()
{
    // optional
}

void ATerraShiftEnvironment::PreStep()
{
    // optional
}

void ATerraShiftEnvironment::PreTransition()
{
    if (!StateManager) return;

    StateManager->UpdateGridObjectFlags();
    StateManager->UpdateObjectStats(GetWorld()->GetDeltaSeconds());
    StateManager->RespawnGridObjects();
    StateManager->BuildCentralState();
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep++;
}

FState ATerraShiftEnvironment::State()
{
    FState st;

    // combine central state + wave-agent states
    if (StateManager)
    {
        // add central furst
        TArray<float> c = StateManager->GetCentralState();
        st.Values.Append(c);

        // then each wave agent state
        for (int32 i = 0; i < CurrentAgents; i++)
        {
            TArray<float> waveAgentArr = StateManager->GetAgentState(i);
            st.Values.Append(waveAgentArr);
        }
    }
    else if (WaveSimulator)
    {
        // fallback => wave only
        st.Values.Append(WaveSimulator->GetHeightMap().Data);
    }
    return st;
}

bool ATerraShiftEnvironment::Done()
{
    // We check if state manager says all objects are handled
    // only after at least 1 step
    if (CurrentStep > 0 && StateManager && StateManager->AllGridObjectsHandled())
    {
        return true;
    }
    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    return (CurrentStep >= MaxSteps);
}

float ATerraShiftEnvironment::Reward()
{
    float dt = GetWorld()->GetDeltaSeconds();
    if (dt < KINDA_SMALL_NUMBER) return 0.f;

    float accum = 0.f;
    if (!StateManager) return accum;

    // We must iterate over the grid objects, not the wave-sim agents!
    // So from 0..CurrentGridObjects-1
    for (int ObjIndex = 0; ObjIndex < CurrentGridObjects; ObjIndex++)
    {
        bool bActive = StateManager->GetHasActive(ObjIndex);
        bool bReached = StateManager->GetHasReachedGoal(ObjIndex);
        bool bFallenOff = StateManager->GetHasFallenOff(ObjIndex);
        bool bCollect = StateManager->GetShouldCollectReward(ObjIndex);

        if (bCollect && bFallenOff)
        {
            accum += FALL_OFF_PENALTY;
            StateManager->SetShouldCollectReward(ObjIndex, false);
            continue;
        }
        else if (bCollect && bReached)
        {
            accum += REACH_GOAL_REWARD;
            StateManager->SetShouldCollectReward(ObjIndex, false);
            continue;
        }

        if (!bActive)
        {
            accum += STEP_PENALTY;
            continue;
        }

        // distance improvements, etc.
        FVector vel = StateManager->GetCurrentVelocity(ObjIndex);
        FVector pvel = StateManager->GetPreviousVelocity(ObjIndex);
        float dist = StateManager->GetCurrentDistance(ObjIndex);
        float pdist = StateManager->GetPreviousDistance(ObjIndex);

        float sub = 0.f;

        if (bUseXYDistanceImprovement && pdist > 0.f && dist > 0.f)
        {
            float delta = (pdist - dist) / PlatformWorldSize.X;
            float clampDelta = FMath::Clamp(delta, DistImprove_Min, DistImprove_Max);
            sub += DistImprove_Scale * clampDelta;
        }
        if (bUseZAccelerationPenalty && !pvel.IsNearlyZero())
        {
            FVector aTot = (vel - pvel) / dt;
            float posZ = (aTot.Z > 0.f) ? aTot.Z : 0.f;
            float cz = ThresholdAndClamp(posZ, ZAccel_Min, ZAccel_Max);
            sub -= (ZAccel_Scale * cz);
        }

        accum += sub;
    }

    return accum * dt;
}

void ATerraShiftEnvironment::UpdateActiveColumns()
{
    if (!GridObjectManager || !Grid) return;

    TSet<int32> newActive = GridObjectManager->GetActiveColumnsInProximity(
        GridSize,
        Grid->GetColumnCenters(),
        PlatformCenter,
        PlatformWorldSize.X,
        CellSize
    );
    TSet<int32> en = newActive.Difference(ActiveColumns);
    TSet<int32> dis = ActiveColumns.Difference(newActive);

    if (en.Num() > 0 || dis.Num() > 0)
    {
        TArray<int32> idxs;
        TArray<bool>   vals;
        for (int32 c : en) { idxs.Add(c); vals.Add(true); }
        for (int32 c : dis) { idxs.Add(c); vals.Add(false); }
        Grid->TogglePhysicsForColumns(idxs, vals);
    }
    ActiveColumns = newActive;
}

void ATerraShiftEnvironment::UpdateColumnColors()
{
    if (!Grid) return;

    // Color each column by height
    float mn = Grid->GetMinHeight();
    float mx = Grid->GetMaxHeight();
    for (int32 c = 0; c < Grid->GetTotalColumns(); c++)
    {
        float h = Grid->GetColumnHeight(c);
        float ratio = FMath::GetMappedRangeValueClamped(FVector2D(mn, mx), FVector2D(0.f, 1.f), h);
        FLinearColor col = FLinearColor::LerpUsingHSV(FLinearColor::Black, FLinearColor::White, ratio);
        Grid->SetColumnColor(c, col);
    }
}

float ATerraShiftEnvironment::ThresholdAndClamp(float value, float minVal, float maxVal)
{
    if (FMath::Abs(value) < minVal) return 0.f;
    return FMath::Clamp(value, -maxVal, maxVal);
}

AMainPlatform* ATerraShiftEnvironment::SpawnPlatform(FVector Location)
{
    UWorld* w = GetWorld();
    if (!w)
    {
        UE_LOG(LogTemp, Error, TEXT("SpawnPlatform => no world."));
        return nullptr;
    }
    UStaticMesh* plane = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
    if (!plane)
    {
        UE_LOG(LogTemp, Error, TEXT("No plane mesh found => can't spawn platform."));
        return nullptr;
    }

    FActorSpawnParameters sp;
    AMainPlatform* p = w->SpawnActor<AMainPlatform>(
        AMainPlatform::StaticClass(),
        Location,
        FRotator::ZeroRotator,
        sp
    );
    if (!p)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn main platform."));
        return nullptr;
    }
    UMaterial* mat = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Platform_Material.Platform_Material'"));
    p->InitializePlatform(plane, mat);
    p->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);
    return p;
}

FVector ATerraShiftEnvironment::CalculateGoalPlatformLocation(int32 EdgeIndex)
{
    // place them around edges
    float offset = (PlatformWorldSize.X * 0.5f) + (PlatformWorldSize.X * ObjectSize.X * 0.5f);
    switch (EdgeIndex)
    {
    case 0: return FVector(0, offset, 0);   // top
    case 1: return FVector(0, -offset, 0);   // bottom
    case 2: return FVector(-offset, 0, 0);   // left
    case 3: return FVector(offset, 0, 0);    // right
    default: return FVector::ZeroVector;
    }
}

void ATerraShiftEnvironment::UpdateGoal(int32 GoalIndex)
{
    FVector scale = ObjectSize;
    FVector loc = CalculateGoalPlatformLocation(GoalIndex);
    FLinearColor col = GoalColors[GoalIndex % GoalColors.Num()];

    FActorSpawnParameters sp;
    sp.Owner = this;
    AGoalPlatform* gp = GetWorld()->SpawnActor<AGoalPlatform>(
        AGoalPlatform::StaticClass(),
        Platform->GetActorLocation(), // spawn near platform
        FRotator::ZeroRotator,
        sp
    );
    if (gp)
    {
        gp->InitializeGoalPlatform(loc, scale, col, Platform);
        GoalPlatforms.Add(gp);
    }
}
