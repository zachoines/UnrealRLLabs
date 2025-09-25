// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.
// This version incorporates the "Fixed-Slot Reward Structure" and preserves existing dense shaping logic.

#include "TerraShiftEnvironment.h"
#include "TerraShift/GoalManager.h"
#include "Kismet/GameplayStatics.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Grid.h"
#include "TerraShift/GridObjectManager.h"

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
    GoalManager = nullptr;
    Platform = nullptr;
    Grid = nullptr;
    PlatformSize = 1.0f;
    MaxColumnHeight = 4.0f;
    MaxSteps = 512;
    MaxAgents = 5;
    ObjectSize = FVector(0.1f);
    ObjectMass = 0.1f;
    GridSize = 50;

    // Default Reward Toggles & Scales
    bUsePotentialShaping = false;
    PotentialShaping_Scale = 1.0f;
    PotentialShaping_Gamma = 0.99f;
    bUseVelAlignment = false;
    bUseXYDistanceImprovement = false;
    bUseZAccelerationPenalty = false;
    VelAlign_Scale = 0.1f;
    VelAlign_Min = -100.f;
    VelAlign_Max = 100.f;
    DistImprove_Scale = 10.f;
    DistImprove_Min = -1.f;
    DistImprove_Max = 1.f;
    ZAccel_Scale = 0.0001f;
    ZAccel_Min = 0.1f;
    ZAccel_Max = 2000.f;

    PlatformWorldSize = FVector::ZeroVector;
    PlatformCenter = FVector::ZeroVector;
    CellSize = 1.0f;
}

ATerraShiftEnvironment::~ATerraShiftEnvironment() {}

void ATerraShiftEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* Params)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(Params);
    check(TerraShiftParams && TerraShiftParams->EnvConfig);

    CurrentStep = 0;
    Initialized = false;
    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath);

    UEnvironmentConfig* EnvConfig = TerraShiftParams->EnvConfig;

    PlatformSize = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/PlatformSize"), 1.f);
    MaxColumnHeight = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/MaxColumnHeight"), 4.f);
    MaxSteps = EnvConfig->GetOrDefaultInt(TEXT("environment/params/MaxSteps"), 512);
    MaxAgents = EnvConfig->GetOrDefaultInt(TEXT("environment/params/MaxAgents"), 5);

    if (EnvConfig->HasPath(TEXT("environment/params/ObjectSize")))
    {
        TArray<float> arr = EnvConfig->Get(TEXT("environment/params/ObjectSize"))->AsArrayOfNumbers();
        ObjectSize = (arr.Num() == 3) ? FVector(arr[0], arr[1], arr[2]) : FVector(0.1f);
    }
    else
    {
        ObjectSize = FVector(0.1f);
    }
    ObjectMass = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/ObjectMass"), 0.1f);
    GridSize = EnvConfig->GetOrDefaultInt(TEXT("environment/params/GridSize"), 50);

    UEnvironmentConfig* envSpecificCfg = EnvConfig->Get(TEXT("environment/params/TerraShiftEnvironment"));
    if (envSpecificCfg)
    {
        bUsePotentialShaping = envSpecificCfg->GetOrDefaultBool(TEXT("bUsePotentialShaping"), false);
        PotentialShaping_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("PotentialShaping_Scale"), 1.0f);
        PotentialShaping_Gamma = envSpecificCfg->GetOrDefaultNumber(TEXT("PotentialShaping_Gamma"), 0.99f);
        bUseVelAlignment = envSpecificCfg->GetOrDefaultBool(TEXT("bUseVelAlignment"), false);
        bUseXYDistanceImprovement = envSpecificCfg->GetOrDefaultBool(TEXT("bUseXYDistanceImprovement"), false);
        bUseZAccelerationPenalty = envSpecificCfg->GetOrDefaultBool(TEXT("bUseZAccelerationPenalty"), false);
        VelAlign_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Scale"), 0.1f);
        VelAlign_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Min"), -100.f);
        VelAlign_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Max"), 100.f);
        DistImprove_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Scale"), 10.f);
        DistImprove_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Min"), -1.f);
        DistImprove_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Max"), 1.f);
        ZAccel_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Scale"), 0.0001f);
        ZAccel_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Min"), 0.1f);
        ZAccel_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Max"), 2000.f);

        EventReward_GoalReached = envSpecificCfg->GetOrDefaultNumber(TEXT("EventReward_GoalReached"), 10.0f);
        EventReward_OutOfBounds = envSpecificCfg->GetOrDefaultNumber(TEXT("EventReward_OutOfBounds"), -10.0f);
        TimeStepPenalty = envSpecificCfg->GetOrDefaultNumber(TEXT("TimeStepPenalty"), -0.001f);

        bUseDistanceBasedReward = envSpecificCfg->GetOrDefaultBool(TEXT("bUseDistanceBasedReward"), false);
        bDisableEventRewards = envSpecificCfg->GetOrDefaultBool(TEXT("bDisableEventRewards"), false);
    }

    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);
    Platform = SpawnPlatform(TerraShiftParams->Location);
    if (Platform)
    {
        Platform->SetActorScale3D(FVector(PlatformSize));
        if (Platform->PlatformMeshComponent && Platform->PlatformMeshComponent->GetStaticMesh())
        {
            PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent * 2.f * Platform->GetActorScale3D();
            PlatformCenter = Platform->GetActorLocation();
            CellSize = (GridSize > 0) ? PlatformWorldSize.X / static_cast<float>(GridSize) : 1.f;
        }
    }

    FVector GridLocation = PlatformCenter + FVector(0.f, 0.f, MaxColumnHeight);
    Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator);
    if (Grid)
    {
        Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepWorldTransform);
        Grid->SetColumnMovementBounds(-MaxColumnHeight, MaxColumnHeight);
        Grid->InitializeGrid(GridSize, PlatformWorldSize.X, GridLocation);
    }

    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager)
    {
        GridObjectManager->SetPlatformActor(Platform);
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + TEXT("/") + GridObjectManager->GetName())));
    }

    UEnvironmentConfig* waveCfg = EnvConfig->Get(TEXT("environment/params/MultiAgentGaussianWaveHeightMap"));
    if (waveCfg && WaveSimulator) WaveSimulator->InitializeFromConfig(waveCfg);

    UEnvironmentConfig* smCfg = EnvConfig->Get(TEXT("environment/params/StateManager"));
    if (smCfg && StateManager)
    {
        StateManager->LoadConfig(smCfg);
        CurrentGridObjects = StateManager->GetMaxGridObjects();
    }

    if (smCfg)
    {
        bTerminateOnAllGoalsReached = smCfg->GetOrDefaultBool(TEXT("bTerminateOnAllGoalsReached"), false);
        bTerminateOnMaxSteps = smCfg->GetOrDefaultBool(TEXT("bTerminateOnMaxSteps"), true);
    }

    UEnvironmentConfig* gmCfg = EnvConfig->Get(TEXT("environment/params/GoalManager"));
    GoalManager = GetWorld()->SpawnActor<AGoalManager>();
    if (GoalManager && gmCfg)
    {
        GoalManager->InitializeFromConfig(gmCfg);
        GoalManager->AttachToActor(Platform, FAttachmentTransformRules::KeepWorldTransform);
        GoalManager->SetFolderPath(FName(*(EnvironmentFolderPath + TEXT("/") + GoalManager->GetName())));
    }

    if (StateManager) StateManager->SetReferences(Platform, GridObjectManager, Grid, WaveSimulator, GoalManager);

    Initialized = true;
    UE_LOG(LogTemp, Log, TEXT("TerraShiftEnvironment Initialized Successfully."));
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents;
    CurrentGridObjects = StateManager ? StateManager->GetMaxGridObjects() : 1;
    PreviousPotential.Init(0.0f, CurrentGridObjects);

    if (StateManager)
    {
        StateManager->Reset(CurrentGridObjects, CurrentAgents);
        for (int32 i = 0; i < CurrentGridObjects; ++i)
        {
            PreviousPotential[i] = CalculatePotential(i);
        }
    }

    return State();
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    if (!Initialized || !WaveSimulator || !Grid) return;
    WaveSimulator->Step(Action.Values, 0.1);
    const FMatrix2D& wave = WaveSimulator->GetHeightMap();
    Grid->UpdateColumnHeights(wave);
}

void ATerraShiftEnvironment::PreTransition()
{
    if (!Initialized || !StateManager) return;
    StateManager->UpdateGridObjectFlags();
    StateManager->UpdateObjectStats(GetWorld()->GetDeltaSeconds());
    StateManager->RespawnGridObjects();
    // Optional optimization: toggle column collision based on proximity to grid objects
    StateManager->UpdateColumnCollisionBasedOnOccupancy();
    StateManager->UpdateGridColumnsColors();
    StateManager->BuildCentralState();
}

void ATerraShiftEnvironment::PostStep()
{
    if (!Initialized) return;
    CurrentStep++;
}

FState ATerraShiftEnvironment::State()
{
    FState CurrentState;
    if (!Initialized || !StateManager) return CurrentState;

    TArray<float> CentralStateData = StateManager->GetCentralState();
    CurrentState.Values.Append(CentralStateData);

    for (int32 i = 0; i < CurrentAgents; i++)
    {
        TArray<float> AgentStateData = StateManager->GetAgentState(i);
        CurrentState.Values.Append(AgentStateData);
    }
    return CurrentState;
}

bool ATerraShiftEnvironment::Done()
{
    if (!Initialized || !StateManager) return true;

    if (bTerminateOnAllGoalsReached && StateManager->AllGridObjectsHandled())
    {
        return true;
    }

    if (bTerminateOnMaxSteps && CurrentStep >= MaxSteps)
    {
        return true;
    }

    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (!Initialized && !!bTerminateOnMaxSteps) return true;
    if (bTerminateOnMaxSteps) return false;
    bool bTruncated = (CurrentStep >= MaxSteps);
    return bTruncated;
}

float ATerraShiftEnvironment::Reward()
{
    // --- Initial Safety Checks ---
    if (!Initialized || !StateManager)
    {
        return 0.f;
    }

    float DeltaTime = GetWorld()->GetDeltaSeconds();
    if (DeltaTime < KINDA_SMALL_NUMBER)
    {
        return 0.f;
    }

    // --- Reward Calculation ---

    float activeObjectCount = 0;
    float totalAlignmentReward = 0;

    // Start with the global, constant penalty applied at each time step.
    float AccumulatedReward = TimeStepPenalty;

    // Iterate through each grid object to calculate its contribution to the reward.
    for (int32 ObjIndex = 0; ObjIndex < CurrentGridObjects; ++ObjIndex)
    {
        // Check for one-time events (Goal Reached / Out of Bounds)
        if (StateManager->GetShouldCollectReward(ObjIndex) && !bDisableEventRewards)
        {
            if (StateManager->GetHasReachedGoal(ObjIndex))
            {
                AccumulatedReward += EventReward_GoalReached;
            }
            else if (StateManager->GetHasFallenOff(ObjIndex))
            {
                AccumulatedReward += EventReward_OutOfBounds;
            }
            // Reset the flag to ensure this event reward is only given once.
            StateManager->SetShouldCollectReward(ObjIndex, false);
        }

        // --- Dense Shaping Rewards (only for active objects) ---
        EObjectSlotState SlotState = StateManager->GetObjectSlotState(ObjIndex);
        if (SlotState == EObjectSlotState::Active)
        {
            float ShapingSubReward = 0.f;
            activeObjectCount += 1.0;

            // 1. Potential-based shaping to encourage progress towards the goal.
            if (bUsePotentialShaping)
            {
                float currentPotential = CalculatePotential(ObjIndex);
                if (PreviousPotential.IsValidIndex(ObjIndex))
                {
                    ShapingSubReward += PotentialShaping_Scale * (PotentialShaping_Gamma * currentPotential - PreviousPotential[ObjIndex]);
                    PreviousPotential[ObjIndex] = currentPotential;
                }
            }

            // 2. Reward for decreasing the XY-distance to the goal.
            if (bUseXYDistanceImprovement)
            {
                float previousDistance = StateManager->GetPreviousDistance(ObjIndex);
                float currentDistanceValue = StateManager->GetCurrentDistance(ObjIndex);
                if (previousDistance > 0.f && currentDistanceValue > 0.f)
                {
                    float deltaDistance = (previousDistance - currentDistanceValue) / PlatformWorldSize.X;
                    ShapingSubReward += DistImprove_Scale * ThresholdAndClamp(deltaDistance, DistImprove_Min, DistImprove_Max);
                }
            }

            // 3. Penalty for excessive upward Z-axis acceleration.
            if (bUseZAccelerationPenalty)
            {
                FVector previousVelocity = StateManager->GetPreviousVelocity(ObjIndex);
                if (!previousVelocity.IsNearlyZero())
                {
                    FVector currentVelocity = StateManager->GetCurrentVelocity(ObjIndex);
                    FVector acceleration = (currentVelocity - previousVelocity) / DeltaTime;
                    float upwardZAcceleration = (acceleration.Z > 0.f) ? acceleration.Z : 0.f;
                    ShapingSubReward -= ZAccel_Scale * ThresholdAndClamp(upwardZAcceleration, ZAccel_Min, ZAccel_Max);
                }
            }

            // 4. Reward for aligning the object's velocity vector towards its goal.
            if (bUseVelAlignment)
            {
                int32 goalIndex = StateManager->GetGoalIndex(ObjIndex);
                if (goalIndex >= 0 && GoalManager)
                {
                    FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(GoalManager->GetGoalLocation(goalIndex));
                    FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);
                    FVector velLocal = StateManager->GetCurrentVelocity(ObjIndex);
                    if (!velLocal.IsNearlyZero())
                    {
                        FVector dirToObjectToGoal = (goalPosLocal - objPosLocal).GetSafeNormal();
                        FVector velLocalNormalized = velLocal.GetSafeNormal();
                        float dotProduct = FVector::DotProduct(velLocalNormalized, dirToObjectToGoal);
                        float alignReward = dotProduct * velLocal.Size();

                        // float boundedAlignReward = FMath::Tanh(alignReward);
                        // ThresholdAndClamp(alignReward, VelAlign_Min, VelAlign_Max)
                        ShapingSubReward += VelAlign_Scale * alignReward;
                    }
                }
            }

            //float speed = StateManager->GetCurrentVelocity(ObjIndex).Size();
            //if (speed < 10.0f)  // Below minimum acceptable speed
            //{
            //    ShapingSubReward -= 0.01f;  // Constant drain for stationary objects
            //}

            AccumulatedReward += ShapingSubReward;
        }
        else // Reset potential for non-active objects
        {
            if (bUsePotentialShaping && PreviousPotential.IsValidIndex(ObjIndex))
            {
                PreviousPotential[ObjIndex] = 0.f;
            }
        }

        // --- Distance-based Reward (Play Mode) - only for active objects ---
        if (bUseDistanceBasedReward)
        {
            if (SlotState == EObjectSlotState::Active || SlotState == EObjectSlotState::GoalReached)
            {
                // Calculate 1 - distance_i for the ith gridobject
                int32 goalIndex = StateManager->GetGoalIndex(ObjIndex);
                if (goalIndex >= 0 && GoalManager)
                {
                    FVector goalPosWorld = GoalManager->GetGoalLocation(goalIndex);
                    FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);
                    FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(goalPosWorld);
                    
                    float distance = FVector::Dist(objPosLocal, goalPosLocal);
                    float normalizedDistance = distance / PlatformWorldSize.X;  // Normalize by platform size
                    float distanceReward = 1.0f - FMath::Clamp(normalizedDistance, 0.0f, 1.0f);
                    AccumulatedReward += distanceReward;
                }
            }
        }
    }

    return AccumulatedReward / CurrentGridObjects;
}

float ATerraShiftEnvironment::ThresholdAndClamp(float value, float minThreshold, float maxClamp)
{
    if (FMath::Abs(value) < minThreshold) return 0.f;
    return FMath::Clamp(value, -maxClamp, maxClamp);
}

AMainPlatform* ATerraShiftEnvironment::SpawnPlatform(FVector Location)
{
    UWorld* World = GetWorld();
    if (!World) return nullptr;

    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
    AMainPlatform* SpawnedPlatform = World->SpawnActor<AMainPlatform>(AMainPlatform::StaticClass(), Location, FRotator::ZeroRotator);

    if (SpawnedPlatform)
    {
        SpawnedPlatform->InitializePlatform(PlaneMesh, nullptr); // Material can be set in BP or elsewhere
        SpawnedPlatform->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);
    }
    return SpawnedPlatform;
}

float ATerraShiftEnvironment::CalculatePotential(int32 ObjIndex) const
{
    if (!StateManager || !(PlatformWorldSize.X > KINDA_SMALL_NUMBER)) return 0.0f;
    if (StateManager->GetObjectSlotState(ObjIndex) != EObjectSlotState::Active) return 0.0f;

    float currentDistance = StateManager->GetCurrentDistance(ObjIndex);
    if (currentDistance < 0.f) return 0.0f;

    // Potential is inversely proportional to distance (more reward for being closer)
    return -currentDistance / PlatformWorldSize.X;
}
