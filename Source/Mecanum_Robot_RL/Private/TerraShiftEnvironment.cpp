// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShiftEnvironment.h"
#include "TerraShift/GoalManager.h"
#include "Kismet/GameplayStatics.h"
#include "Components/StaticMeshComponent.h" // Added for GetStaticMesh()
#include "Engine/StaticMesh.h" // Added for GetBounds()
#include "Materials/Material.h" // Added for UMaterial*
#include "EnvironmentConfig.h" // Added for UEnvironmentConfig
#include "TerraShift/Grid.h" // Added for AGrid
#include "TerraShift/GridObjectManager.h" // Added for AGridObjectManager

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    WaveSimulator = CreateDefaultSubobject<UMultiAgentGaussianWaveHeightMap>(TEXT("WaveSimulator"));
    StateManager = CreateDefaultSubobject<UStateManager>(TEXT("StateManager"));

    // --- Default State ---
    CurrentStep = 0;
    Initialized = false;
    CurrentAgents = 1;
    CurrentGridObjects = 1;

    // --- Default References ---
    GoalManager = nullptr;
    Platform = nullptr;
    Grid = nullptr;

    // --- Default Configuration Values ---
    PlatformSize = 1.0f;
    MaxColumnHeight = 4.0f;
    MaxSteps = 512;
    MaxAgents = 5;
    ObjectSize = FVector(0.1f);
    ObjectMass = 0.1f;
    GridSize = 50;

    // --- Default Reward Toggles & Scales ---
    bUsePotentialShaping = false; // Default off
    PotentialShaping_Scale = 1.0f;
    // Assuming user updated C++ to use PotentialShaping_Gamma from env config
    PotentialShaping_Gamma = 0.99f; // Default gamma for potential shaping

    bUseAlignedDistanceShaping = false;
    bUseVelAlignment = false;
    bUseXYDistanceImprovement = false; // Changed default to false based on original code
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

    REACH_GOAL_REWARD = 1.f;
    FALL_OFF_PENALTY = -1.f;
    STEP_PENALTY = -0.0001f;

    PlatformWorldSize = FVector::ZeroVector;
    PlatformCenter = FVector::ZeroVector;
    CellSize = 1.0f;
}

ATerraShiftEnvironment::~ATerraShiftEnvironment()
{
    // Destructor logic if needed
}

void ATerraShiftEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    // Tick logic if needed
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* Params)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(Params);
    check(TerraShiftParams && TerraShiftParams->EnvConfig); // Ensure params and config are valid

    // Reset internal state
    CurrentStep = 0;
    CurrentAgents = 1; // Will be updated in ResetEnv based on actual request
    CurrentGridObjects = 1; // Will be updated based on StateManager config
    Initialized = false;

    // Setup folder paths for potential logging/debugging (Matches original)
    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath); // Sets folder for this environment actor

    UEnvironmentConfig* EnvConfig = TerraShiftParams->EnvConfig;

    // --- Read Global Environment Parameters ---
    PlatformSize = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/PlatformSize"), 1.f);
    MaxColumnHeight = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/MaxColumnHeight"), 4.f);
    MaxSteps = EnvConfig->GetOrDefaultInt(TEXT("environment/params/MaxSteps"), 512);
    MaxAgents = EnvConfig->GetOrDefaultInt(TEXT("environment/params/MaxAgents"), 5); // Max possible agents

    // Read ObjectSize vector (Matches original)
    if (EnvConfig->HasPath(TEXT("environment/params/ObjectSize")))
    {
        TArray<float> arr = EnvConfig->Get(TEXT("environment/params/ObjectSize"))->AsArrayOfNumbers();
        if (arr.Num() == 3)
        {
            ObjectSize = FVector(arr[0], arr[1], arr[2]);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("ObjectSize in config must have 3 floats. Using default (0.1, 0.1, 0.1)."));
            ObjectSize = FVector(0.1f);
        }
    }
    else
    {
        ObjectSize = FVector(0.1f); // Default if not specified
    }

    ObjectMass = EnvConfig->GetOrDefaultNumber(TEXT("environment/params/ObjectMass"), 0.1f);
    GridSize = EnvConfig->GetOrDefaultInt(TEXT("environment/params/GridSize"), 50);

    // --- Read TerraShift Specific Reward Parameters ---
    UEnvironmentConfig* envSpecificCfg = EnvConfig->Get(TEXT("environment/params/TerraShiftEnvironment"));
    if (envSpecificCfg)
    {
        // Potential Shaping Params (NEW)
        bUsePotentialShaping = envSpecificCfg->GetOrDefaultBool(TEXT("bUsePotentialShaping"), false);
        PotentialShaping_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("PotentialShaping_Scale"), 1.0f);
        // Read the gamma specific to potential shaping if defined, else use default
        PotentialShaping_Gamma = envSpecificCfg->GetOrDefaultNumber(TEXT("PotentialShaping_Gamma"), 0.99f);

        // Original Reward Toggles & Scales (Logic matches original, reads from config)
        bUseVelAlignment = envSpecificCfg->GetOrDefaultBool(TEXT("bUseVelAlignment"), false);
        // Note: Original code had default true here, but config loading overrides it.
        bUseXYDistanceImprovement = envSpecificCfg->GetOrDefaultBool(TEXT("bUseXYDistanceImprovement"), false);
        bUseZAccelerationPenalty = envSpecificCfg->GetOrDefaultBool(TEXT("bUseZAccelerationPenalty"), false);
        bUseAlignedDistanceShaping = envSpecificCfg->GetOrDefaultBool(TEXT("bUseAlignedDistanceShaping"), false);

        VelAlign_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Scale"), 0.1f);
        VelAlign_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Min"), -100.f);
        VelAlign_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("VelAlign_Max"), 100.f);

        DistImprove_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Scale"), 10.f);
        DistImprove_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Min"), -1.f);
        DistImprove_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("DistImprove_Max"), 1.f);

        ZAccel_Scale = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Scale"), 0.0001f);
        ZAccel_Min = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Min"), 0.1f);
        ZAccel_Max = envSpecificCfg->GetOrDefaultNumber(TEXT("ZAccel_Max"), 2000.f);

        REACH_GOAL_REWARD = envSpecificCfg->GetOrDefaultNumber(TEXT("REACH_GOAL_REWARD"), 1.f);
        FALL_OFF_PENALTY = envSpecificCfg->GetOrDefaultNumber(TEXT("FALL_OFF_PENALTY"), -1.f);
        STEP_PENALTY = envSpecificCfg->GetOrDefaultNumber(TEXT("STEP_PENALTY"), -0.0001f);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("TerraShiftEnvironment specific reward config block not found. Using defaults."));
    }

    // --- Setup Environment Actors (Matches original logic) ---

    // Place environment root at the requested location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // Spawn and scale the main platform
    Platform = SpawnPlatform(TerraShiftParams->Location);
    if (Platform)
    {
        Platform->SetActorScale3D(FVector(PlatformSize));

        // Compute platform geometry after spawning and scaling
        if (Platform->PlatformMeshComponent && Platform->PlatformMeshComponent->GetStaticMesh())
        {
            PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
                * 2.f // BoxExtent is half-size
                * Platform->GetActorScale3D(); // Apply scaling
            PlatformCenter = Platform->GetActorLocation();
            CellSize = (GridSize > 0) ? PlatformWorldSize.X / static_cast<float>(GridSize) : 1.f; // Avoid division by zero
            UE_LOG(LogTemp, Log, TEXT("Platform World Size: %s, Cell Size: %f"), *PlatformWorldSize.ToString(), CellSize);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Platform spawned but mesh component or static mesh is invalid. Cannot calculate geometry."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Fatal, TEXT("Failed to spawn Platform Actor. Environment cannot initialize."));
        return; // Stop initialization if platform fails
    }

    // Spawn the grid actor
    {
        FVector GridLocation = PlatformCenter + FVector(0.f, 0.f, MaxColumnHeight); // Position grid above platform center
        FActorSpawnParameters SpawnParams;
        SpawnParams.Owner = this;
        Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator, SpawnParams);
        if (Grid)
        {
            Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepWorldTransform); // Attach to platform
            Grid->SetColumnMovementBounds(-MaxColumnHeight, MaxColumnHeight);
            Grid->InitializeGrid(GridSize, PlatformWorldSize.X, GridLocation);
            // Grid->SetFolderPath(FName(*(EnvironmentFolderPath + "/Grid"))); // Remains commented out as in original
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to spawn Grid Actor."));
        }
    }

    // Spawn the grid object manager
    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager)
    {
        GridObjectManager->SetPlatformActor(Platform); // Provide reference to platform
        // Set folder path for the manager actor itself (Matches original)
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + TEXT("/") + GridObjectManager->GetName())));
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn GridObjectManager Actor."));
    }

    // --- Configure Components (Matches original logic) ---

    // Configure Wave Simulator from its config block
    {
        UEnvironmentConfig* waveCfg = EnvConfig->Get(TEXT("environment/params/MultiAgentGaussianWaveHeightMap"));
        if (waveCfg && WaveSimulator)
        {
            WaveSimulator->InitializeFromConfig(waveCfg);
        }
        else if (!WaveSimulator)
        {
            UE_LOG(LogTemp, Error, TEXT("WaveSimulator component is null."));
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("MultiAgentGaussianWaveHeightMap config block not found. WaveSimulator might not be configured correctly."));
        }
    }

    // Configure State Manager from its config block
    {
        UEnvironmentConfig* smCfg = EnvConfig->Get(TEXT("environment/params/StateManager"));
        if (smCfg && StateManager)
        {
            StateManager->LoadConfig(smCfg);
            // Update CurrentGridObjects based on actual config value (Matches original)
            CurrentGridObjects = StateManager->GetMaxGridObjects();
        }
        else if (!StateManager)
        {
            UE_LOG(LogTemp, Error, TEXT("StateManager component is null."));
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("StateManager config block not found. StateManager might not be configured correctly."));
        }
    }

    // Spawn and Initialize Goal Manager
    {
        UEnvironmentConfig* gmCfg = EnvConfig->Get(TEXT("environment/params/GoalManager"));
        GoalManager = GetWorld()->SpawnActor<AGoalManager>();
        if (GoalManager && gmCfg)
        {
            GoalManager->InitializeFromConfig(gmCfg);
            GoalManager->AttachToActor(Platform, FAttachmentTransformRules::KeepWorldTransform); // Attach for relative positioning if needed
        }
        else if (!GoalManager)
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to spawn GoalManager Actor."));
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("GoalManager config block not found. GoalManager might not be configured correctly."));
        }
    }

    // --- Finalize Initialization ---

    // Pass necessary references to the State Manager (Matches original)
    if (StateManager)
    {
        StateManager->SetReferences(Platform, GridObjectManager, Grid, WaveSimulator, GoalManager);
    }

    Initialized = true; // Mark environment as initialized
    UE_LOG(LogTemp, Log, TEXT("TerraShiftEnvironment Initialized Successfully."));
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents; // Set the number of active agents for this episode

    // Ensure CurrentGridObjects reflects the configuration (Matches original logic)
    if (StateManager)
    {
        CurrentGridObjects = StateManager->GetMaxGridObjects();
    }
    else
    {
        // Should not happen if InitEnv succeeded, but handle defensively
        CurrentGridObjects = 1;
        UE_LOG(LogTemp, Error, TEXT("StateManager is null during ResetEnv."));
    }

    // Initialize PreviousPotential array for potential shaping (NEW)
    PreviousPotential.Init(0.0f, CurrentGridObjects);

    // Reset the state manager (handles object positions, goals, agent states, etc.)
    if (StateManager)
    {
        StateManager->Reset(CurrentGridObjects, CurrentAgents); // Original call

        // Calculate initial potential AFTER state manager has reset object positions (NEW)
        for (int32 i = 0; i < CurrentGridObjects; ++i)
        {
            PreviousPotential[i] = CalculatePotential(i);
        }
    }

    UE_LOG(LogTemp, Verbose, TEXT("Environment Reset: Agents=%d, Objects=%d"), CurrentAgents, CurrentGridObjects);

    // Return the initial state after reset
    return State(); // Original call
}

// Act, PostTransition, PreStep, PreTransition, PostStep, State, Done, Trunc match original functionality

void ATerraShiftEnvironment::Act(FAction Action)
{
    if (!Initialized) return; // Don't act if not initialized

    // Pass actions to the wave simulator to generate the height map
    if (WaveSimulator)
    {
        WaveSimulator->Step(Action.Values, GetWorld()->GetDeltaSeconds());

        // Apply the resulting wave height map to the grid columns
        const FMatrix2D& wave = WaveSimulator->GetHeightMap();
        if (Grid)
        {
            Grid->UpdateColumnHeights(wave);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Grid is null during Act. Cannot apply wave heights."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("WaveSimulator is null during Act. Cannot process actions."));
    }
}


void ATerraShiftEnvironment::PreTransition()
{
    if (!Initialized || !StateManager) return; // Added Initialized check for safety

    // Update object states (flags, stats) before calculating reward/next state
    StateManager->UpdateGridObjectFlags();
    StateManager->UpdateObjectStats(GetWorld()->GetDeltaSeconds()); // Pass delta time for velocity/accel calculations

    // Handle object respawning if necessary
    StateManager->RespawnGridObjects();

    // Update grid visuals (e.g., column colors)
    StateManager->UpdateGridColumnsColors();

    // Build the central state representation (e.g., height maps, object info)
    StateManager->BuildCentralState();
}

void ATerraShiftEnvironment::PostStep()
{
    if (!Initialized) return;
    CurrentStep++; // Increment step counter
}

FState ATerraShiftEnvironment::State()
{
    FState CurrentState;
    if (!Initialized || !StateManager)
    {
        UE_LOG(LogTemp, Error, TEXT("Attempting to get State from uninitialized or invalid environment."));
        return CurrentState; // Return empty state
    }

    // Append the central state representation
    TArray<float> CentralStateData = StateManager->GetCentralState();
    CurrentState.Values.Append(CentralStateData);

    // Append individual agent states
    for (int32 i = 0; i < CurrentAgents; i++)
    {
        TArray<float> AgentStateData = StateManager->GetAgentState(i);
        CurrentState.Values.Append(AgentStateData);
    }
    return CurrentState;
}

bool ATerraShiftEnvironment::Done()
{
    if (!Initialized || !StateManager) return true; // Treat uninitialized as done to prevent errors

    // Episode is done if the StateManager determines all objects are handled (reached goal or fallen off)
    // Only check after at least one step to allow initial state setup
    if (CurrentStep > 0 && StateManager->AllGridObjectsHandled())
    {
        UE_LOG(LogTemp, Verbose, TEXT("Episode Done: All objects handled at step %d."), CurrentStep);
        return true;
    }
    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (!Initialized) return true; // Treat uninitialized as truncated

    // Episode is truncated if the maximum step count is reached
    bool bTruncated = (CurrentStep >= MaxSteps);
    if (bTruncated)
    {
        UE_LOG(LogTemp, Verbose, TEXT("Episode Truncated: Max steps (%d) reached."), MaxSteps);
    }
    return bTruncated;
}


float ATerraShiftEnvironment::Reward()
{
    if (!Initialized || !StateManager) return 0.f;

    float DeltaTime = GetWorld()->GetDeltaSeconds();
    // If DeltaTime is too small, physics might not have updated properly, skip reward calculation
    if (DeltaTime < KINDA_SMALL_NUMBER) return 0.f;

    float AccumulatedReward = 0.f;

    for (int ObjIndex = 0; ObjIndex < CurrentGridObjects; ObjIndex++)
    {
        // Get current status flags for the object from the StateManager
        bool bIsActive = StateManager->GetHasActive(ObjIndex);
        bool bHasReachedGoal = StateManager->GetHasReachedGoal(ObjIndex);
        bool bHasFallenOff = StateManager->GetHasFallenOff(ObjIndex);
        bool bShouldCollectTerminalReward = StateManager->GetShouldCollectReward(ObjIndex); // Flag indicating terminal state reached *this step*

        float currentPotential = 0.0f;
        // Calculate potential based on the CURRENT state (after the last action) only if needed (NEW)
        if (bUsePotentialShaping && bIsActive) // Calculate potential only for active objects if shaping is enabled
        {
            currentPotential = CalculatePotential(ObjIndex);
        }

        // --- 1. Handle Terminal Rewards (Matches original logic, uses configured values) ---
        if (bShouldCollectTerminalReward)
        {
            StateManager->SetShouldCollectReward(ObjIndex, false); // Reset the flag

            if (bHasFallenOff)
            {
                AccumulatedReward += FALL_OFF_PENALTY;
                UE_LOG(LogTemp, Verbose, TEXT("Object %d: Fell off. Reward: %f"), ObjIndex, FALL_OFF_PENALTY);
                if (bUsePotentialShaping) PreviousPotential[ObjIndex] = 0.0f; // Reset potential for this object for next episode (NEW)
                continue; // No further rewards for this object this step
            }
            if (bHasReachedGoal)
            {
                AccumulatedReward += REACH_GOAL_REWARD;
                UE_LOG(LogTemp, Verbose, TEXT("Object %d: Reached goal. Reward: %f"), ObjIndex, REACH_GOAL_REWARD);
                if (bUsePotentialShaping) PreviousPotential[ObjIndex] = 0.0f; // Reset potential for this object for next episode (NEW)
                continue; // No further rewards for this object this step
            }
        }

        // If object is not active (and wasn't terminated this step), skip remaining rewards
        if (!bIsActive)
        {
            if (bUsePotentialShaping) PreviousPotential[ObjIndex] = 0.0f; // Ensure potential is reset if inactive mid-episode somehow (NEW)
            continue;
        }


        // --- 2. Step Penalty (Matches original logic, uses configured value) ---
        AccumulatedReward += STEP_PENALTY;

        // --- 3. Potential-Based Shaping Reward (NEW) ---
        if (bUsePotentialShaping)
        {
            float prevPotential = PreviousPotential[ObjIndex];
            // Shaping Reward = Scale * (gamma * Phi(s_t+1) - Phi(s_t))
            // Using PotentialShaping_Gamma read from config
            float shapingReward = PotentialShaping_Scale * (PotentialShaping_Gamma * currentPotential - prevPotential);
            AccumulatedReward += shapingReward;

            // IMPORTANT: Update PreviousPotential *after* using it, ready for the next step
            PreviousPotential[ObjIndex] = currentPotential;
        }


        // --- 4. Additional Dense Shaping Rewards (Matches original logic, uses configured toggles/scales) ---
        float ShapingSubReward = 0.f; // Renamed from 'sub' in original for clarity

        // 4A) XY Distance Improvement Reward
        if (bUseXYDistanceImprovement)
        {
            float previousDistance = StateManager->GetPreviousDistance(ObjIndex);
            float currentDistance = StateManager->GetCurrentDistance(ObjIndex);
            if (previousDistance > 0.f && currentDistance > 0.f)
            {
                float deltaDistance = (previousDistance - currentDistance) / PlatformWorldSize.X;
                float clampedDelta = FMath::Clamp(deltaDistance, DistImprove_Min, DistImprove_Max);
                ShapingSubReward += DistImprove_Scale * clampedDelta;
            }
        }

        // 4B) Z Acceleration Penalty
        if (bUseZAccelerationPenalty)
        {
            FVector previousVelocity = StateManager->GetPreviousVelocity(ObjIndex);
            if (!previousVelocity.IsNearlyZero())
            {
                FVector currentVelocity = StateManager->GetCurrentVelocity(ObjIndex);
                FVector acceleration = (currentVelocity - previousVelocity) / DeltaTime;
                float upwardZAcceleration = (acceleration.Z > 0.f) ? acceleration.Z : 0.f;
                float clampedZAccel = ThresholdAndClamp(upwardZAcceleration, ZAccel_Min, ZAccel_Max);
                ShapingSubReward -= (ZAccel_Scale * clampedZAccel);
            }
        }

        // 4C) Velocity Alignment Reward
        if (bUseVelAlignment) // Removed redundant bActive check from original as we check !bIsActive earlier
        {
            int32 goalIndex = StateManager->GetGoalIndex(ObjIndex);
            if (goalIndex >= 0)
            {
                FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(GoalManager->GetGoalLocation(goalIndex));
                FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);
                FVector velLocal = StateManager->GetCurrentVelocity(ObjIndex);

                if (!velLocal.IsNearlyZero())
                {
                    FVector dirToObjectToGoal = goalPosLocal - objPosLocal;
                    float distanceToGoal = dirToObjectToGoal.Size();

                    if (distanceToGoal > KINDA_SMALL_NUMBER)
                    {
                        dirToObjectToGoal.Normalize();
                        velLocal.Normalize();
                        float dotProduct = FVector::DotProduct(velLocal, dirToObjectToGoal);
                        float alignReward = VelAlign_Scale * dotProduct * velLocal.Size();
                        ShapingSubReward += alignReward;
                    }
                }
            }
        }

        // 4D) Aligned Distance Shaping Reward
        if (bUseAlignedDistanceShaping) // Removed redundant bActive check
        {
            int32 goalIndex = StateManager->GetGoalIndex(ObjIndex);
            if (goalIndex >= 0)
            {
                FVector goalPosLocal = Platform->GetActorTransform().InverseTransformPosition(GoalManager->GetGoalLocation(goalIndex));
                FVector objPosLocal = StateManager->GetCurrentPosition(ObjIndex);
                FVector velLocal = StateManager->GetCurrentVelocity(ObjIndex);

                float previousDistance = StateManager->GetPreviousDistance(ObjIndex);
                float currentDistance = StateManager->GetCurrentDistance(ObjIndex);
                float deltaDistance = 0.0f;
                if (previousDistance > 0.f && currentDistance > 0.f)
                {
                    deltaDistance = (previousDistance - currentDistance) / PlatformWorldSize.X; // Normalized delta
                }

                if (!velLocal.IsNearlyZero() && deltaDistance > 0.f)
                {
                    FVector dirToObjectToGoal = goalPosLocal - objPosLocal;
                    float distanceToGoal = dirToObjectToGoal.Size();

                    if (distanceToGoal > KINDA_SMALL_NUMBER)
                    {
                        dirToObjectToGoal.Normalize();
                        velLocal.Normalize();
                        float dotProduct = FVector::DotProduct(velLocal, dirToObjectToGoal);
                        float positiveAlignment = FMath::Max(dotProduct, 0.f);

                        float alignedReward = positiveAlignment * deltaDistance * DistImprove_Scale;
                        ShapingSubReward += alignedReward;
                    }
                }
            }
        }

        AccumulatedReward += ShapingSubReward; // Add all shaping sub-rewards for this object

    } // End of object loop

    return AccumulatedReward;
}


/**
 * Helper function to clamp a value if its absolute value exceeds a minimum threshold.
 */
float ATerraShiftEnvironment::ThresholdAndClamp(float value, float minThreshold, float maxClamp)
{
    if (FMath::Abs(value) < minThreshold) return 0.f;
    return FMath::Clamp(value, -maxClamp, maxClamp); // Clamp both positive and negative
}

/**
 * Spawns the main platform actor.
 */
AMainPlatform* ATerraShiftEnvironment::SpawnPlatform(FVector Location)
{
    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("SpawnPlatform failed: GetWorld() returned null."));
        return nullptr;
    }

    // Load the plane mesh and platform material (Consider moving paths to config or making properties)
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
    UMaterial* PlatformMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Platform_Material.Platform_Material'"));

    if (!PlaneMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("SpawnPlatform failed: Could not load /Engine/BasicShapes/Plane.Plane mesh."));
        return nullptr;
    }
    if (!PlatformMaterial)
    {
        UE_LOG(LogTemp, Warning, TEXT("SpawnPlatform: Could not load Platform_Material. Using default material."));
        // Platform will use default material
    }

    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    AMainPlatform* SpawnedPlatform = World->SpawnActor<AMainPlatform>(AMainPlatform::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);

    if (!SpawnedPlatform)
    {
        UE_LOG(LogTemp, Error, TEXT("SpawnPlatform failed: SpawnActor<AMainPlatform> returned null."));
        return nullptr;
    }

    SpawnedPlatform->InitializePlatform(PlaneMesh, PlatformMaterial);
    // *** THE FIX: Reverted to KeepRelativeTransform ***
    SpawnedPlatform->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);
    return SpawnedPlatform;
}


/**
 * Calculates the potential function Phi(s) for a given object's state. (NEW)
 * Uses negative normalized distance to the goal as the potential.
 */
float ATerraShiftEnvironment::CalculatePotential(int32 ObjIndex) const
{
    // Ensure StateManager is valid and the object index is within bounds
    if (!StateManager || !PreviousPotential.IsValidIndex(ObjIndex) || !(PlatformWorldSize.X > 0)) // Added check for positive PlatformWorldSize.X
    {
        return 0.0f; // Return zero potential if state is invalid or platform size is zero
    }

    // Get current status - don't calculate potential for objects that are already terminal
    bool bIsActive = StateManager->GetHasActive(ObjIndex);
    if (!bIsActive)
    {
        return 0.0f; // Potential is 0 for inactive/terminal states
    }


    // Get the current distance to the goal for the object
    float currentDistance = StateManager->GetCurrentDistance(ObjIndex);

    // Handle invalid distance (e.g., before first calculation or if goal is invalid)
    if (currentDistance < 0.f)
    {
        return 0.0f;
    }

    // Potential is negative distance, normalized by platform width.
    // Lower distance (better state) -> higher potential (less negative).
    float potential = -currentDistance / PlatformWorldSize.X;

    return potential;
}
