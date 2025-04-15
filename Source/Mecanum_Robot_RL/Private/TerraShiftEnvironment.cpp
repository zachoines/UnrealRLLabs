#include "TerraShiftEnvironment.h"
#include "TerraShift/GoalManager.h"
#include "Kismet/GameplayStatics.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    WaveSimulator = CreateDefaultSubobject<UMultiAgentGaussianWaveHeightMap>(TEXT("WaveSimulator"));
    StateManager = CreateDefaultSubobject<UStateManager>(TEXT("StateManager"));

    CurrentStep = 0;
    Initialized = false;
    CurrentAgents = 1;
    CurrentGridObjects = 1;

    // default references
    GoalManager = nullptr;

    // Default reward toggles
    bUseAlignedDistanceShaping = false;
    bUseVelAlignment = false;
    bUseXYDistanceImprovement = false;
    bUseZAccelerationPenalty = false;

    // Default reward scales
    VelAlign_Scale = 0.1f;
    VelAlign_Min = -100.f;
    VelAlign_Max = 100.f;

    DistImprove_Scale = 10.f;
    DistImprove_Min = -1.f;
    DistImprove_Max = 1.f;

    ZAccel_Scale = 0.0001f;
    ZAccel_Min = 0.1f;
    ZAccel_Max = 2000.f;

    REACH_GOAL_REWARD = 1.f;
    FALL_OFF_PENALTY = -1.f;
    STEP_PENALTY = -0.0001f;
}

ATerraShiftEnvironment::~ATerraShiftEnvironment()
{
}

void ATerraShiftEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* Params)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(Params);
    check(TerraShiftParams && TerraShiftParams->EnvConfig);

    CurrentStep = 0;
    CurrentAgents = 1;
    CurrentGridObjects = 1;
    Initialized = false;

    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath);

    UEnvironmentConfig* EnvConfig = TerraShiftParams->EnvConfig;

    // read TerraShift top-level config
    auto GetOrDefaultNumber = [&](const FString& Path, float DefVal) -> float
        {
            return EnvConfig->HasPath(*Path)
                ? EnvConfig->Get(*Path)->AsNumber()
                : DefVal;
        };
    auto GetOrDefaultInt = [&](const FString& Path, int DefVal) -> int
        {
            return EnvConfig->HasPath(*Path)
                ? EnvConfig->Get(*Path)->AsInt()
                : DefVal;
        };
    auto GetOrDefaultBool = [&](const FString& Path, bool DefVal) -> bool
        {
            return EnvConfig->HasPath(*Path)
                ? EnvConfig->Get(*Path)->AsBool()
                : DefVal;
        };

    // environment/params
    PlatformSize = GetOrDefaultNumber(TEXT("environment/params/PlatformSize"), 1.f);
    MaxColumnHeight = GetOrDefaultNumber(TEXT("environment/params/MaxColumnHeight"), 4.f);
    MaxSteps = GetOrDefaultInt(TEXT("environment/params/MaxSteps"), 512);
    MaxAgents = GetOrDefaultInt(TEXT("environment/params/MaxAgents"), 5);

    // object size
    if (EnvConfig->HasPath(TEXT("environment/params/ObjectSize")))
    {
        TArray<float> arr = EnvConfig->Get(TEXT("environment/params/ObjectSize"))->AsArrayOfNumbers();
        if (arr.Num() == 3) ObjectSize = FVector(arr[0], arr[1], arr[2]);
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("ObjectSize must have 3 floats => fallback (0.1)"));
            ObjectSize = FVector(0.1f);
        }
    }
    else
    {
        ObjectSize = FVector(0.1f);
    }

    ObjectMass = GetOrDefaultNumber(TEXT("environment/params/ObjectMass"), 0.1f);
    GridSize = GetOrDefaultInt(TEXT("environment/params/GridSize"), 50);

    // Now read environment/params/TerraShiftEnvironment sub-block for reward toggles
    UEnvironmentConfig* envCfg2 = EnvConfig->Get(TEXT("environment/params/TerraShiftEnvironment"));
    if (envCfg2)
    {
        // toggles
        bUseVelAlignment = envCfg2->GetOrDefaultBool(TEXT("bUseVelAlignment"), false);
        bUseXYDistanceImprovement = envCfg2->GetOrDefaultBool(TEXT("bUseXYDistanceImprovement"), true);
        bUseZAccelerationPenalty = envCfg2->GetOrDefaultBool(TEXT("bUseZAccelerationPenalty"), false);
        bUseAlignedDistanceShaping = envCfg2->GetOrDefaultBool(TEXT("bUseAlignedDistanceShaping"), false);

        // scales
        VelAlign_Scale = envCfg2->GetOrDefaultNumber(TEXT("VelAlign_Scale"), 0.1f);
        VelAlign_Min = envCfg2->GetOrDefaultNumber(TEXT("VelAlign_Min"), -100.f);
        VelAlign_Max = envCfg2->GetOrDefaultNumber(TEXT("VelAlign_Max"), 100.f);

        DistImprove_Scale = envCfg2->GetOrDefaultNumber(TEXT("DistImprove_Scale"), 10.f);
        DistImprove_Min = envCfg2->GetOrDefaultNumber(TEXT("DistImprove_Min"), -1.f);
        DistImprove_Max = envCfg2->GetOrDefaultNumber(TEXT("DistImprove_Max"), 1.f);

        ZAccel_Scale = envCfg2->GetOrDefaultNumber(TEXT("ZAccel_Scale"), 0.0001f);
        ZAccel_Min = envCfg2->GetOrDefaultNumber(TEXT("ZAccel_Min"), 0.1f);
        ZAccel_Max = envCfg2->GetOrDefaultNumber(TEXT("ZAccel_Max"), 2000.f);

        REACH_GOAL_REWARD = envCfg2->GetOrDefaultNumber(TEXT("REACH_GOAL_REWARD"), 1.f);
        FALL_OFF_PENALTY = envCfg2->GetOrDefaultNumber(TEXT("FALL_OFF_PENALTY"), -1.f);
        STEP_PENALTY = envCfg2->GetOrDefaultNumber(TEXT("STEP_PENALTY"), -0.0001f);
    }

    // place environment at location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // spawn platform
    Platform = SpawnPlatform(TerraShiftParams->Location);
    Platform->SetActorScale3D(FVector(PlatformSize));

    // compute geometry
    if (Platform && Platform->PlatformMeshComponent)
    {
        PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
            * 2.f
            * Platform->GetActorScale3D();
        PlatformCenter = Platform->GetActorLocation();
        CellSize = (GridSize > 0) ? PlatformWorldSize.X / (float)GridSize : 1.f;
    }

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
            // Grid->SetFolderPath(FName(*(EnvironmentFolderPath + "/Grid")));
        }
    }

    // spawn object manager
    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager)
    {
        GridObjectManager->SetPlatformActor(Platform);
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + "/GridObjectManager")));
    }

    // wave simulator config
    {
        UEnvironmentConfig* waveCfg = EnvConfig->Get(TEXT("environment/params/MultiAgentGaussianWaveHeightMap"));
        if (waveCfg && WaveSimulator)
        {
            WaveSimulator->InitializeFromConfig(waveCfg);
        }
    }

    // state manager
    {
        UEnvironmentConfig* smCfg = EnvConfig->Get(TEXT("environment/params/StateManager"));
        if (smCfg && StateManager)
        {
            StateManager->LoadConfig(smCfg);
            int32 MaxGridObjs = smCfg->GetOrDefaultInt(TEXT("max_grid_objects"), 8);
            CurrentGridObjects = MaxGridObjs;
        }
    }

    // spawn + init goal manager
    {
        UEnvironmentConfig* gmCfg = EnvConfig->Get(TEXT("environment/params/GoalManager"));
        GoalManager = GetWorld()->SpawnActor<AGoalManager>();
        if (GoalManager && gmCfg)
        {
            GoalManager->InitializeFromConfig(gmCfg);
        }
    }

    // finalize references => pass GoalManager
    if (StateManager)
    {
        StateManager->SetReferences(Platform, GridObjectManager, Grid, WaveSimulator, GoalManager);
    }

    Initialized = true;
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents;
    CurrentGridObjects = StateManager->GetMaxGridObjects();

    // Let the state manager do the sub resets:
    if (StateManager)
    {
        StateManager->Reset(CurrentGridObjects, CurrentAgents);
    }

    return State();
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    if (WaveSimulator)
    {
        WaveSimulator->Step(Action.Values, GetWorld()->GetDeltaSeconds());
        // apply final wave to columns
        const FMatrix2D& wave = WaveSimulator->GetHeightMap();
        if (Grid)
        {
            Grid->UpdateColumnHeights(wave);
        }
    }
}

void ATerraShiftEnvironment::PostTransition()
{
    // optional hook
}

void ATerraShiftEnvironment::PreStep()
{
    // optional hook
}

void ATerraShiftEnvironment::PreTransition()
{
    if (!StateManager) return;

    // Let the manager handle objects
    StateManager->UpdateGridObjectFlags();
    StateManager->UpdateObjectStats(GetWorld()->GetDeltaSeconds());
    StateManager->RespawnGridObjects();
    StateManager->UpdateGridColumnsColors();
    StateManager->BuildCentralState();
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep++;
}

FState ATerraShiftEnvironment::State()
{
    FState st;
    // central state
    TArray<float> c = StateManager->GetCentralState();
    st.Values.Append(c);

    // wave agent states
    for (int32 i = 0; i < CurrentAgents; i++)
    {
        TArray<float> waveState = StateManager->GetAgentState(i);
        st.Values.Append(waveState);
    }
    return st;
}

bool ATerraShiftEnvironment::Done()
{
    // done if StateManager says all handled, after at least 1 step
    if (CurrentStep > 0 && StateManager->AllGridObjectsHandled())
    {
        return true;
    }
    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    // truncated if max steps reached
    return (CurrentStep >= MaxSteps);
}

float ATerraShiftEnvironment::Reward()
{
    float dt = GetWorld()->GetDeltaSeconds();
    if (dt < KINDA_SMALL_NUMBER) return 0.f;

    float accum = 0.f;

    for (int ObjIndex = 0; ObjIndex < CurrentGridObjects; ObjIndex++)
    {
        bool bActive = StateManager->GetHasActive(ObjIndex);
        bool bReached = StateManager->GetHasReachedGoal(ObjIndex);
        bool bFallen = StateManager->GetHasFallenOff(ObjIndex);
        bool bCollect = StateManager->GetShouldCollectReward(ObjIndex);

        // 1) if we are set to collect => handle "fallen" or "reached"
        if (bCollect)
        {
            StateManager->SetShouldCollectReward(ObjIndex, false);

            if (bFallen)
            {
                accum += FALL_OFF_PENALTY;
                continue;
            }
            if (bReached)
            {
                accum += REACH_GOAL_REWARD;
                continue;
            }
        }

        // 2) step penalty (only if object is active)
        if (bActive)
        {
            accum += STEP_PENALTY;
        }

        // 3) Additional sub‐rewards or penalties
        float sub = 0.f;

        // 3A) XY distance improvement
        if (bUseXYDistanceImprovement)
        {
            float pdist = StateManager->GetPreviousDistance(ObjIndex);
            float dist = StateManager->GetCurrentDistance(ObjIndex);
            if (pdist > 0.f && dist > 0.f)
            {
                float delta = (pdist - dist) / PlatformWorldSize.X;
                float clampDelta = FMath::Clamp(delta, DistImprove_Min, DistImprove_Max);
                sub += DistImprove_Scale * clampDelta;
            }
        }

        // 3B) Z acceleration penalty
        if (bUseZAccelerationPenalty)
        {
            FVector pvel = StateManager->GetPreviousVelocity(ObjIndex);
            if (!pvel.IsNearlyZero())
            {
                FVector vel = StateManager->GetCurrentVelocity(ObjIndex);
                FVector aTot = (vel - pvel) / dt;
                float posZ = (aTot.Z > 0.f) ? aTot.Z : 0.f;
                float cz = ThresholdAndClamp(posZ, ZAccel_Min, ZAccel_Max);
                sub -= (ZAccel_Scale * cz);
            }
        }

        // 3C) Velocity alignment to the Goal
        if (bUseVelAlignment && bActive)
        {
            int32 gIndex = StateManager->GetGoalIndex(ObjIndex);
            if (gIndex >= 0)
            {
                // 1) get positions in local coords
                FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(
                    GoalManager->GetGoalLocation(gIndex));
                FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);
                FVector velLocal = StateManager->GetCurrentVelocity(ObjIndex);

                // 2) if velocity is not zero
                if (!velLocal.IsNearlyZero())
                {
                    // direction from object to goal
                    FVector toGoal = goalPosLocal - objPosLocal;
                    float distToGoal = toGoal.Size();

                    if (distToGoal > KINDA_SMALL_NUMBER)
                    {
                        // normalized direction to goal
                        FVector dirToGoal = toGoal / distToGoal;

                        // normalized velocity
                        float speed = velLocal.Size();
                        FVector velNorm = velLocal / speed;

                        // dot in [-1..1]; rawAlign * speed => “speed towards (or away) from goal”
                        float dot = FVector::DotProduct(velNorm, dirToGoal); // -1..+1
                        //float speedTowardsGoal = dot * speed;                // negative if going away

                        //// clamp to [VelAlign_Min..VelAlign_Max]
                        //float stgClamped = FMath::Clamp(speedTowardsGoal,
                        //    VelAlign_Min,
                        //    VelAlign_Max);

                        // multiply by your scale factor
                        float alignReward = VelAlign_Scale * dot;
                        sub += alignReward;
                    }
                }
            }
        }

        // 3D) Aligned Distance Shaping Reward
        if (bUseAlignedDistanceShaping && bActive)
        {
            // We combine raw velocity alignment (dot product) with distance improvement,
            int32 gIndex = StateManager->GetGoalIndex(ObjIndex);
            if (gIndex >= 0)
            {

                FVector rawGoalPos = GoalManager->GetGoalLocation(gIndex);
                FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(rawGoalPos);

                // Get the object position (already stored in StateManager in local coordinates)
                FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);

                // Distance improvement: previous distance minus current distance, normalized by platform size so we get range => [-1, 1]
                float pdist = StateManager->GetPreviousDistance(ObjIndex);
                float currDist = StateManager->GetCurrentDistance(ObjIndex);
                float delta = (pdist - currDist) / PlatformWorldSize.X;

                // Calculate the velocity alignment: normalized_direction @ normalized_velocity => [-1, 1]
                FVector velLocal = StateManager->GetCurrentVelocity(ObjIndex);
                if (!velLocal.IsNearlyZero())
                {
                    FVector velNorm = velLocal / velLocal.Size();
                    FVector dirToGoal = goalPosLocal - objPosLocal;
                    float goalDist = dirToGoal.Size();

                    if (goalDist > KINDA_SMALL_NUMBER)
                    {
                        dirToGoal /= goalDist;

                        float dot = FVector::DotProduct(velNorm, dirToGoal);

                        /*
                        Here we want to use the raw dot (which will be negative if movement is away) and multiply it by the improvement signal.
                        However, to ensure that positive improvement in distance yields reward and negative improvement yields a penalty, we use a piecewise approach:
                        */

                        float alignedReward = 0.f;
                        // Object is moving closer to its goal
                        if (delta >= 0.f)
                        {
                            // Only consider positive velocity alignment
                            float positiveAlign = FMath::Max(dot, 0.f); // zero out any negative alignment
                            alignedReward = positiveAlign * delta * DistImprove_Scale;
                        }
                        // Object is moving away
                        /*else
                        {
                            alignedReward = FMath::Abs(delta);
                        }*/
                        sub += alignedReward;
                    }
                }
            }
        }

        accum += sub;
    }

    return accum / static_cast<float>(CurrentGridObjects);
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
        UE_LOG(LogTemp, Error, TEXT("No plane mesh => can't spawn platform."));
        return nullptr;
    }

    FActorSpawnParameters sp;
    AMainPlatform* p = w->SpawnActor<AMainPlatform>(
        AMainPlatform::StaticClass(),
        Location, FRotator::ZeroRotator,
        sp
    );
    if (!p)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn main platform."));
        return nullptr;
    }
    UMaterial* mat = LoadObject<UMaterial>(
        nullptr,
        TEXT("Material'/Game/Material/Platform_Material.Platform_Material'")
    );
    p->InitializePlatform(plane, mat);
    p->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);
    return p;
}
