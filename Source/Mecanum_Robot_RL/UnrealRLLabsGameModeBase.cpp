#include "UnrealRLLabsGameModeBase.h"
#include "EnvironmentConfig.h"
#include "Misc/Paths.h"

// Constructor
AUnrealRLLabsGameModeBase::AUnrealRLLabsGameModeBase()
{
    // Disable ticking by default
    PrimaryActorTick.bCanEverTick = false;
}

// Helper to create environment spawn locations in a grid arrangement
TArray<FVector> AUnrealRLLabsGameModeBase::CreateGridLocations(int32 NumEnvironments, FVector Offset)
{
    TArray<FVector> Locations;

    // Initial estimate based on cube root
    int32 initialEstimate = FMath::RoundToInt(FMath::Pow(NumEnvironments, 1.0f / 3.0f));

    int32 xCount = initialEstimate;
    int32 yCount = initialEstimate;
    int32 zCount = initialEstimate;

    // Grow xCount/yCount/zCount until we can fit all environments
    while (xCount * yCount * zCount < NumEnvironments)
    {
        if (xCount <= yCount && xCount <= zCount)
            xCount++;
        else if (yCount <= xCount && yCount <= zCount)
            yCount++;
        else
            zCount++;
    }

    // Center them around the origin
    FVector StartLocation = FVector(
        -Offset.X * (xCount - 1) * 0.5f,
        -Offset.Y * (yCount - 1) * 0.5f,
        -Offset.Z * (zCount - 1) * 0.5f
    );

    // Populate
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

void AUnrealRLLabsGameModeBase::BeginPlay()
{
    Super::BeginPlay();

    // 1) Load config from "Configs/TerraShift.json"
    EnvConfig = NewObject<UEnvironmentConfig>(this, UEnvironmentConfig::StaticClass());
    if (!EnvConfig || !EnvConfig->LoadFromFile(FPaths::ProjectContentDir() + TEXT("Python/Configs/TerraShift.json")))
    {
        UE_LOG(LogTemp, Error, TEXT("Could not load TerraShift.json config!"));
        return;
    }

    // 2) Retrieve the number of environments from the config
    //    e.g. "train/num_environments" => int
    int32 NumEnvironments = EnvConfig
        ->Get(TEXT("train/num_environments"))
        ->AsInt();

    // 3) Spawn the RLRunner actor
    Runner = GetWorld()->SpawnActor<ARLRunner>(ARLRunner::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
    if (!Runner)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn RLRunner!"));
        return;
    }

    // 4) Generate environment spawn locations
    TArray<FVector> Locations = CreateGridLocations(NumEnvironments, FVector(300.f, 300.f, 300.f));
    if (Locations.Num() != NumEnvironments)
    {
        UE_LOG(LogTemp, Error, TEXT("Could not spawn environments - mismatch in location count."));
    }

    // 5) Build our environment init-params array
    for (int32 i = 0; i < NumEnvironments; i++)
    {
        // For TerraShift environment, we might have a derived struct: FTerraShiftEnvironmentInitParams
        FTerraShiftEnvironmentInitParams* TerraParams = new FTerraShiftEnvironmentInitParams();
        TerraParams->Location = Locations[i];
        TerraParams->EnvConfig = EnvConfig;

        // Add to our array
        InitParamsArray.Add(StaticCast<FBaseInitParams*>(TerraParams));
    }

    // 6) Initialize runner with the environment type + init params + config
    Runner->InitRunner(
        ATerraShiftEnvironment::StaticClass(),
        InitParamsArray,
        EnvConfig
    );
}

void AUnrealRLLabsGameModeBase::BeginDestroy()
{
    Super::BeginDestroy();

    // Clean up any allocated FBaseInitParams
    for (FBaseInitParams* Params : InitParamsArray)
    {
        delete Params;
    }
    InitParamsArray.Empty();
}

void AUnrealRLLabsGameModeBase::InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage)
{
    Super::InitGame(MapName, Options, ErrorMessage);
    UE_LOG(LogTemp, Log, TEXT("RL session has started: %s %s"), *MapName, *Options);
}