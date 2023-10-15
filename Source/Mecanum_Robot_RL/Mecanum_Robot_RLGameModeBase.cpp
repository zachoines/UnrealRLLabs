#include "Mecanum_Robot_RLGameModeBase.h"

AMecanum_Robot_RLGameModeBase::AMecanum_Robot_RLGameModeBase()
{
    // Enable ticking
    PrimaryActorTick.bCanEverTick = false;
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
    
    int BufferSize = 16;
    int BatchSize = 16;
    int NumEnvironments = 4096 * 2;
    int StateSize = 6;
    int NumActions = 2;
    FVector GroundPlaneSize = FVector::One() * 5.0;
    FVector ControlledCubeSize = FVector::One() * 0.25;
    FVector Offset(800.0f, 800.0f, 100.0f);

    // Spawn the ARLRunner
    Runner = GetWorld()->SpawnActor<ARLRunner>(ARLRunner::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);

    // Create an array of initialization parameters for the environments
    TArray<FVector> Locations = CreateGridLocations(NumEnvironments, Offset);

    for (int32 i = 0; i < Locations.Num(); i++)
    {
        FCubeEnvironmentInitParams* CubeParams = new FCubeEnvironmentInitParams();
        CubeParams->GroundPlaneSize = GroundPlaneSize;
        CubeParams->ControlledCubeSize = ControlledCubeSize;
        CubeParams->Location = Locations[i];

        InitParamsArray.Add(StaticCast<FBaseInitParams*>(CubeParams));
    }

    Runner->InitRunner(
        ACubeEnvironment::StaticClass(), 
        InitParamsArray, 
        BufferSize, 
        BatchSize,
        Locations.Num(),
        StateSize,
        NumActions
    );
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

