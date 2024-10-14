// TerraShiftEnvironment.cpp

#include "TerraShiftEnvironment.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Materials/Material.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/StaticMesh.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));

    WaveSimulator = nullptr;
}

ATerraShiftEnvironment::~ATerraShiftEnvironment()
{
    if (WaveSimulator)
    {
        delete WaveSimulator;
        WaveSimulator = nullptr;
    }
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    MaxAgents = TerraShiftParams->MaxAgents;
    CurrentAgents = BaseParams->NumAgents;
    CurrentStep = 0;

    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    Platform = SpawnPlatform(
        TerraShiftParams->Location,
        FVector(TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize, 1.0f) // Z scale set to 1.0f
    );

    // Initialize columns
    Columns.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);
    GridCenterPoints.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);

    FVector PlatformScale = Platform->GetActorScale3D();
    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * PlatformScale;
    FVector PlatformCenter = Platform->GetActorLocation();

    // Calculate cell dimensions
    float CellWidth = PlatformWorldSize.X / static_cast<float>(TerraShiftParams->GridSize);
    float CellLength = PlatformWorldSize.Y / static_cast<float>(TerraShiftParams->GridSize);

    // Load column mesh to get its original dimensions
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    FVector ColumnMeshExtent = ColumnMeshAsset->GetBounds().BoxExtent * 2.0f;
    float OriginalColumnHeight = ColumnMeshExtent.Z;

    // Calculate initial and maximum column heights in world units
    float MinColumnHeight = CellWidth * TerraShiftParams->ColumnHeight;
    float MaxColumnHeight = MinColumnHeight * TerraShiftParams->MaxColumnHeight;

    // Calculate starting and maximum scale factors
    float StartingScaleZ = MinColumnHeight / OriginalColumnHeight;
    float MaximumScaleFactor = TerraShiftParams->MaxColumnHeight;

    // Initialize columns
    for (int i = 0; i < TerraShiftParams->GridSize; ++i)
    {
        for (int j = 0; j < TerraShiftParams->GridSize; ++j)
        {
            AColumn* Column = GetWorld()->SpawnActor<AColumn>(AColumn::StaticClass());
            if (Column)
            {
                // Position base on top of platform
                FVector GridCenter = PlatformCenter + FVector(
                    (i - TerraShiftParams->GridSize / 2.0f + 0.5f) * CellWidth,
                    (j - TerraShiftParams->GridSize / 2.0f + 0.5f) * CellLength,
                    PlatformCenter.Z + (OriginalColumnHeight * StartingScaleZ) / 2.0f
                );

                FVector ColumnScale = FVector(
                    CellWidth / ColumnMeshExtent.X,
                    CellLength / ColumnMeshExtent.Y,
                    StartingScaleZ
                );

                Column->InitColumn(ColumnScale, GridCenter, MaximumScaleFactor);

                // Attach the column to the platform
                Column->AttachToActor(Platform, FAttachmentTransformRules::KeepWorldTransform);

                Columns[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = Column;
                GridCenterPoints[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = GridCenter;
            }
        }
    }

    // Update EnvInfo
    const int NumAgentWaveParameters = static_cast<int>(EAgentParameterIndex::Count);
    const int NumAgentStateParameters = NumAgentWaveParameters + 3 /*GridObject position*/ + 2 /*Goal position*/;
    EnvInfo.SingleAgentObsSize = NumAgentStateParameters;
    EnvInfo.StateSize = MaxAgents * EnvInfo.SingleAgentObsSize;

    // Set up continuous action space
    const int NumAgentActions = 7; // VelocityX, VelocityY, Amplitude, WaveOrientation, Wavenumber, Phase, Sigma
    TArray<FContinuousActionSpec> ContinuousActions;
    for (int i = 0; i < NumAgentActions; ++i)
    {
        FContinuousActionSpec ActionSpec;
        ActionSpec.Low = -1.0f;
        ActionSpec.High = 1.0f;
        ContinuousActions.Add(ActionSpec);
    }

    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init(ContinuousActions, {});
    }

    // Initialize MorletWavelets2D
    WaveSimulator = new MorletWavelets2D(TerraShiftParams->GridSize, TerraShiftParams->GridSize, 1.0f);

    // Initialize AgentParametersArray
    AgentParametersArray.SetNum(MaxAgents);

    // Initialize agents with starting parameters
    for (int i = 0; i < MaxAgents; ++i)
    {
        AgentParameters AgentParam;
        AgentParam.Position = FVector2f(FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)),
            FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)));
        AgentParam.Velocity = FVector2f(0.0f, 0.0f);
        AgentParam.Amplitude = FMath::FRandRange(TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        AgentParam.WaveOrientation = FMath::FRandRange(TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        AgentParam.Wavenumber = FMath::FRandRange(TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        AgentParam.Phase = FMath::FRandRange(TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        AgentParam.Sigma = FMath::FRandRange(TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);
        AgentParam.Time = 0.0f;
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();

        AgentParametersArray[i] = AgentParam;
    }

    // Load materials and meshes
    UMaterial* DefaultObjectMaterial = LoadObject<UMaterial>(this, TEXT("Material'/Game/Material/Manip_Object_Material.Manip_Object_Material'"));
    UStaticMesh* DefaultObjectMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));

    for (int i = 0; i < MaxAgents; ++i)
    {
        AGridObject* GridObject = GetWorld()->SpawnActor<AGridObject>(AGridObject::StaticClass());
        GridObject->InitializeGridObject(TerraShiftParams->ObjectSize, DefaultObjectMesh, DefaultObjectMaterial);
        Objects.Add(GridObject);

        // Initially hide all grid objects; only activate when required
        GridObject->SetGridObjectActive(false);
    }

    LastColumnIndexArray.SetNum(MaxAgents);
    AgentGoalIndices.SetNum(MaxAgents);

    SetActiveGridObjects(CurrentAgents);

    WaveSimulator->Initialize();
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents;

    // Reset goals and columns
    GoalPositionArray.Empty();
    GoalPositionArray.SetNum(TerraShiftParams->NumGoals);
    TArray<FIntPoint> GoalIndices2D;
    GoalIndices2D.SetNum(TerraShiftParams->NumGoals);

    for (int i = 0; i < TerraShiftParams->NumGoals; ++i)
    {
        int Side;
        FIntPoint GoalIndex2D;
        do
        {
            Side = FMath::RandRange(0, 3);
            switch (Side)
            {
            case 0: // Top
                GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), 0);
                break;
            case 1: // Bottom
                GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), TerraShiftParams->GridSize - 1);
                break;
            case 2: // Left
                GoalIndex2D = FIntPoint(0, FMath::RandRange(0, TerraShiftParams->GridSize - 1));
                break;
            case 3: // Right
                GoalIndex2D = FIntPoint(TerraShiftParams->GridSize - 1, FMath::RandRange(0, TerraShiftParams->GridSize - 1));
                break;
            }
        } while (GoalIndices2D.Contains(GoalIndex2D));

        GoalIndices2D[i] = GoalIndex2D;
        GoalPositionArray[i] = GridCenterPoints[Get1DIndexFromPoint(GoalIndex2D, TerraShiftParams->GridSize)];
    }

    for (int i = 0; i < Columns.Num(); ++i)
    {
        Columns[i]->ResetColumn();
    }

    for (int i = 0; i < TerraShiftParams->NumGoals; ++i)
    {
        FLinearColor GoalColor = GoalColors[i % GoalColors.Num()];
        Columns[Get1DIndexFromPoint(GoalIndices2D[i], TerraShiftParams->GridSize)]->SetColumnColor(GoalColor);
    }

    SetActiveGridObjects(CurrentAgents);

    WaveSimulator->Reset();

    AgentParametersArray.SetNum(MaxAgents);

    for (int i = 0; i < MaxAgents; ++i)
    {
        AgentParameters AgentParam;
        AgentParam.Position = FVector2f(FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)),
            FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)));
        AgentParam.Velocity = FVector2f(0.0f, 0.0f);
        AgentParam.Amplitude = FMath::FRandRange(TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        AgentParam.WaveOrientation = FMath::FRandRange(TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        AgentParam.Wavenumber = FMath::FRandRange(TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        AgentParam.Phase = FMath::FRandRange(TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        AgentParam.Sigma = FMath::FRandRange(TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);
        AgentParam.Time = 0.0f;
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();

        AgentParametersArray[i] = AgentParam;
    }

    WaveSimulator->Initialize();

    return State();
}

void ATerraShiftEnvironment::SetActiveGridObjects(int NumAgents)
{
    // Use GetComponentBounds to get platform bounds considering scale
    FVector PlatformOrigin, PlatformExtent;
    Platform->GetStaticMeshComponent()->GetLocalBounds(PlatformOrigin, PlatformExtent);
    PlatformExtent *= Platform->GetActorScale3D(); // Scale it to match platform's world size

    FVector PlatformCenter = Platform->GetActorLocation();
    for (int i = 0; i < MaxAgents; ++i)
    {
        Objects[i]->SetGridObjectActive(false);
        if (i < NumAgents)
        {
            // Get the extent of the object, scaled by TerraShiftParams->ObjectSize
            FVector GridObjectExtent = Objects[i]->GetObjectExtent() * TerraShiftParams->ObjectSize;

            // Compute a random location above and within the platform bounds
            FVector RandomLocation = PlatformCenter + FVector(
                FMath::RandRange(-PlatformExtent.X + GridObjectExtent.X, PlatformExtent.X - GridObjectExtent.X),
                FMath::RandRange(-PlatformExtent.Y + GridObjectExtent.Y, PlatformExtent.Y - GridObjectExtent.Y),
                PlatformCenter.Z + PlatformExtent.Z + GridObjectExtent.Z + (PlatformExtent.Z * 0.1f) // Ensure it's above the platform
            );

            SetSpawnGridObject(
                i,
                static_cast<float>(i) * TerraShiftParams->SpawnDelay,
                RandomLocation
            );
        }
    }
}

void ATerraShiftEnvironment::SetSpawnGridObject(int AgentIndex, float Delay, FVector Location)
{
    FTimerHandle TimerHandle;
    GetWorldTimerManager().SetTimer(
        TimerHandle,
        [this, AgentIndex, Location]() {
            Objects[AgentIndex]->SetActorLocationAndActivate(Location);
        },
        Delay,
        false
    );
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location, FVector Size)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
        if (PlaneMesh)
        {
            FActorSpawnParameters SpawnParams;
            AStaticMeshActor* NewPlatform = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (NewPlatform)
            {
                NewPlatform->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
                NewPlatform->GetStaticMeshComponent()->SetWorldScale3D(Size);
                NewPlatform->SetMobility(EComponentMobility::Static);
                NewPlatform->GetStaticMeshComponent()->SetSimulatePhysics(false);
                // NewPlatform->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                // NewPlatform->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
            }
            return NewPlatform;
        }
    }
    return nullptr;
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex)
{
    TArray<float> State;

    const AgentParameters& Agent = AgentParametersArray[AgentIndex];

    float AgentPosX_Grid = Agent.Position.X;
    float AgentPosY_Grid = Agent.Position.Y;

    FVector AgentWorldPosition = GridPositionToWorldPosition(FVector2D(AgentPosX_Grid, AgentPosY_Grid));
    FVector AgentRelativePosition = Platform->GetActorTransform().InverseTransformPosition(AgentWorldPosition);

    FVector ObjectWorldPosition = Objects[AgentIndex]->GetActorLocation();
    FVector ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);

    int AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalWorldPosition = GoalPositionArray[AgentGoalIndex];
    FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(GoalWorldPosition);

    State.Add(AgentRelativePosition.X);
    State.Add(AgentRelativePosition.Y);
    State.Add(Agent.Velocity.X);
    State.Add(Agent.Velocity.Y);
    State.Add(Agent.Amplitude);
    State.Add(Agent.WaveOrientation);
    State.Add(Agent.Wavenumber);
    State.Add(Agent.Frequency);
    State.Add(Agent.Phase);
    State.Add(Agent.Sigma);
    State.Add(Agent.Time);
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);
    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);

    return State;
}

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const
{
    return Point.X * GridSize + Point.Y;
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep += 1;
}

FState ATerraShiftEnvironment::State()
{
    FState CurrentState;
    for (int i = 0; i < CurrentAgents; ++i)
    {
        CurrentState.Values.Append(AgentGetState(i));
    }
    return CurrentState;
}

bool ATerraShiftEnvironment::Done()
{
    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (CurrentStep > TerraShiftParams->MaxSteps)
    {
        CurrentStep = 0;
        return true;
    }

    return false;
}

float ATerraShiftEnvironment::Reward()
{
    return 0.0f;
}

void ATerraShiftEnvironment::PostTransition()
{
}

float ATerraShiftEnvironment::Map(float x, float in_min, float in_max, float out_min, float out_max) const
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    const int NumAgentActions = EnvInfo.ActionSpace->ContinuousActions.Num();
    if (Action.Values.Num() != CurrentAgents * NumAgentActions)
    {
        UE_LOG(LogTemp, Error, TEXT("Action array size mismatch. Expected %d, got %d"), CurrentAgents * NumAgentActions, Action.Values.Num());
        return;
    }

    float DeltaTime = GetWorld()->GetDeltaSeconds();

    for (int i = 0; i < CurrentAgents; ++i)
    {
        int ActionIndex = i * NumAgentActions;
        float VelocityXAction = Action.Values[ActionIndex];
        float VelocityYAction = Action.Values[ActionIndex + 1];
        float AmplitudeAction = Action.Values[ActionIndex + 2];
        float WaveOrientationAction = Action.Values[ActionIndex + 3];
        float WavenumberAction = Action.Values[ActionIndex + 4];
        float PhaseAction = Action.Values[ActionIndex + 5];
        float SigmaAction = Action.Values[ActionIndex + 6];

        // Map actions from [-1, 1] to the expected parameter ranges
        float VelocityX = Map(VelocityXAction, -1.0f, 1.0f, TerraShiftParams->VelocityRange.X, TerraShiftParams->VelocityRange.Y);
        float VelocityY = Map(VelocityYAction, -1.0f, 1.0f, TerraShiftParams->VelocityRange.X, TerraShiftParams->VelocityRange.Y);
        float Amplitude = Map(AmplitudeAction, -1.0f, 1.0f, TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        float WaveOrientation = Map(WaveOrientationAction, -1.0f, 1.0f, TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        float Wavenumber = Map(WavenumberAction, -1.0f, 1.0f, TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        float Phase = Map(PhaseAction, -1.0f, 1.0f, TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        float Sigma = Map(SigmaAction, -1.0f, 1.0f, TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);

        AgentParameters& AgentParam = AgentParametersArray[i];

        AgentParam.Velocity = FVector2f(VelocityX, VelocityY);
        AgentParam.Amplitude = Amplitude;
        AgentParam.WaveOrientation = WaveOrientation;
        AgentParam.Wavenumber = Wavenumber;
        AgentParam.Phase = Phase;
        AgentParam.Sigma = Sigma;
        AgentParam.Position += AgentParam.Velocity * DeltaTime;

        // Keep agents within grid boundaries
        float GridSize = static_cast<float>(TerraShiftParams->GridSize);
        AgentParam.Position.X = FMath::Fmod(AgentParam.Position.X + GridSize, GridSize);
        AgentParam.Position.Y = FMath::Fmod(AgentParam.Position.Y + GridSize, GridSize);

        AgentParam.Time += DeltaTime;

        // Calculate frequency based on wavenumber and phase velocity
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();
    }

    WaveSimulator->Update(AgentParametersArray);

    const Matrix2D& HeightMap = WaveSimulator->GetHeights();

    // Update column heights based on the height map
    int GridSize = TerraShiftParams->GridSize;
    for (int i = 0; i < GridSize; ++i)
    {
        for (int j = 0; j < GridSize; ++j)
        {
            int Index = Get1DIndexFromPoint(FIntPoint(i, j), GridSize);
            float HeightValue = HeightMap[i][j];

            // Normalize HeightValue to 0 to 1 (since columns cannot go below the platform)
            float NormalizedHeight = FMath::Clamp((HeightValue + TerraShiftParams->MaxColumnHeight) / (2.0f * TerraShiftParams->MaxColumnHeight), 0.0f, 1.0f);

            Columns[Index]->SetColumnHeight(NormalizedHeight);
        }
    }
}

bool ATerraShiftEnvironment::ObjectOffPlatform(int AgentIndex) const
{
    if (Objects[AgentIndex]->IsHidden())
    {
        return false;
    }

    FVector ObjectPosition = Objects[AgentIndex]->GetActorLocation();
    FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();
    FVector PlatformCenter = Platform->GetActorLocation();

    // Check if the object is within the bounds of the platform
    if (ObjectPosition.X < PlatformCenter.X - PlatformExtent.X ||
        ObjectPosition.X > PlatformCenter.X + PlatformExtent.X ||
        ObjectPosition.Y < PlatformCenter.Y - PlatformExtent.Y ||
        ObjectPosition.Y > PlatformCenter.Y + PlatformExtent.Y)
    {
        return true;
    }

    return false;
}

FVector ATerraShiftEnvironment::GridPositionToWorldPosition(FVector2D GridPosition)
{
    FVector PlatformCenter = Platform->GetActorLocation();
    FVector PlatformScale = Platform->GetActorScale3D();
    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * PlatformScale;

    float PosX = PlatformCenter.X + (GridPosition.X - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.X / TerraShiftParams->GridSize);
    float PosY = PlatformCenter.Y + (GridPosition.Y - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.Y / TerraShiftParams->GridSize);
    float PosZ = PlatformCenter.Z; // Assuming Z is at the platform's height

    return FVector(PosX, PosY, PosZ);
}
