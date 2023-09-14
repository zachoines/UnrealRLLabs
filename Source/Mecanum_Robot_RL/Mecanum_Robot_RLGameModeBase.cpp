#include "Mecanum_Robot_RLGameModeBase.h"

AMecanum_Robot_RLGameModeBase::AMecanum_Robot_RLGameModeBase()
{
    // Enable ticking
    PrimaryActorTick.bCanEverTick = true;
}

TArray<FVector> AMecanum_Robot_RLGameModeBase::CreateGridLocations(int32 NumEnvironments, FVector Offset)
{
    TArray<FVector> Locations;

    // Initial estimate based on cube root
    int32 initialEstimate = FMath::RoundToInt(FMath::Pow(NumEnvironments, 1.0f / 3.0f));

    int32 xCount = initialEstimate;
    int32 yCount = initialEstimate;
    int32 zCount = initialEstimate;

    // Adjust counts to get as close as possible to the desired number of environments
    while (xCount * yCount * zCount < NumEnvironments)
    {
        if (xCount <= yCount && xCount <= zCount)
            xCount++;
        else if (yCount <= xCount && yCount <= zCount)
            yCount++;
        else
            zCount++;
    }

    // Calculate the starting point to ensure the environments are centered around the world's origin
    FVector StartLocation = FVector(-Offset.X * (xCount - 1) * 0.5f, -Offset.Y * (yCount - 1) * 0.5f, -Offset.Z * (zCount - 1) * 0.5f);

    int32 envsCreated = 0;
    for (int32 i = 0; i < xCount && envsCreated < NumEnvironments; i++)
    {
        for (int32 j = 0; j < yCount && envsCreated < NumEnvironments; j++)
        {
            for (int32 k = 0; k < zCount && envsCreated < NumEnvironments; k++)
            {
                FVector Location = StartLocation + FVector(i * Offset.X, j * Offset.Y, k * Offset.Z);
                Locations.Add(Location);
                envsCreated++;
            }
        }
    }

    return Locations;
}

void AMecanum_Robot_RLGameModeBase::BeginPlay()
{
    // UE_LOG(LOG_MECANUM_ROBOT_RL, Warning, TEXT("Running the Torch Agent Test"));
    // FTorchPlugin& torchPlugin = FModuleManager::Get().LoadModuleChecked<FTorchPlugin>("TorchPlugin");
    // torchPlugin.Init();
    // torchPlugin.RunAgentTest();

    if (!AgentComm)
    {
        AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
        FSharedMemoryAgentCommunicatorConfig Config;
        Config.NumEnvironments = 5;
        Config.NumActions = 2;
        Config.StateSize = 4;
        Config.TrainingBatchSize = 10;

        AgentComm->Init(Config);
    }

    UE_LOG(LOG_MECANUM_ROBOT_RL, Warning, TEXT("Running microenvironments"));
    

    FVector GroundPlaneSize = FVector::One() * 5.0;
    FVector ControlledCubeSize = FVector::One() * 0.25;
    FVector Offset(800.0f, 800.0f, 100.0f);

    // Spawn the ARLRunner
    Runner = GetWorld()->SpawnActor<ARLRunner>(ARLRunner::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);

    // Create an array of initialization parameters for the environments
    TArray<FVector> Locations = CreateGridLocations(8, Offset);

    for (int32 i = 0; i < Locations.Num(); i++)
    {
        FCubeEnvironmentInitParams* CubeParams = new FCubeEnvironmentInitParams();
        CubeParams->GroundPlaneSize = GroundPlaneSize;
        CubeParams->ControlledCubeSize = ControlledCubeSize;
        CubeParams->Location = Locations[i];

        InitParamsArray.Add(StaticCast<FBaseInitParams*>(CubeParams));
    }

    Runner->InitRunner(ACubeEnvironment::StaticClass(), InitParamsArray, 1000, 256);
}

void AMecanum_Robot_RLGameModeBase::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Dummy state data for testing.
    TArray<FState> TestStates;

    for (int32 i = 0; i < 5; ++i) // Let's assume 5 environments.
    {
        FState SingleState;
        for (int32 j = 0; j < 4; ++j) // Assuming state size is 4.
        {
            SingleState.Values.Add(FMath::RandRange(0.f, 1.f)); // Random values between 0 and 1 for demonstration.
        }
        TestStates.Add(SingleState);
    }

    // Get actions for the dummy states.
    TArray<FAction> Actions = AgentComm->GetActions(TestStates);

    // For now, just print the first action of the first environment to see if it works.
    if (Actions.Num() > 0 && Actions[0].Values.Num() > 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("First action of the first environment: %f"), Actions[0].Values[0]);
    }
}

void AMecanum_Robot_RLGameModeBase::InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage)
{
    Super::InitGame(MapName, Options, ErrorMessage);
    UE_LOG(LOG_MECANUM_ROBOT_RL, Log, TEXT("RL session has started: % %"), *MapName, *Options);
}

void AMecanum_Robot_RLGameModeBase::BeginDestroy()
{
    Super::BeginDestroy();

    // Delete all the FCubeEnvironmentInitParams instances
    for (FBaseInitParams* Params : InitParamsArray)
    {
        delete Params;
    }
}

