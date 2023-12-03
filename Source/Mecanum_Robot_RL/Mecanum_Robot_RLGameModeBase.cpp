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
    bool loaded = ReadJsonConfig(
        FPaths::ProjectContentDir() + TEXT("EnvConfigs/MultiAgentCube.json"),
        TrainParams
    );

    if (loaded) {
        Runner = GetWorld()->SpawnActor<ARLRunner>(ARLRunner::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
        TArray<FVector> Locations = CreateGridLocations(TrainParams.NumEnvironments, FVector(800.0f, 800.0f, 100.0f));
    
        if (TrainParams.NumEnvironments != Locations.Num()) {
            // TODO:: Throw error
        }

        for (int32 i = 0; i < TrainParams.NumEnvironments; i++)
        {
            FMultiAgentCubeEnvironmentInitParams* MultiCubeParams = new FMultiAgentCubeEnvironmentInitParams();
            MultiCubeParams->Location = Locations[i];
            InitParamsArray.Add(StaticCast<FBaseInitParams*>(MultiCubeParams));
        }

        Runner->InitRunner(
            AMultiAgentCubeEnvironment::StaticClass(),
            InitParamsArray,
            TrainParams
        );
    }
    else {
        // TODO:: Throw error
    }
}

bool AMecanum_Robot_RLGameModeBase::ReadJsonConfig(const FString& FilePath, FTrainParams& OutTrainParams)
{
    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *FilePath))
    {
       return false;
    }

    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        return false;
    }

    // Parse TrainParams
    TSharedPtr<FJsonObject> TrainParamsJson = JsonObject->GetObjectField(TEXT("TrainInfo"));
    OutTrainParams.BufferSize = TrainParamsJson->GetIntegerField(TEXT("BufferSize"));
    OutTrainParams.BatchSize = TrainParamsJson->GetIntegerField(TEXT("BatchSize"));
    OutTrainParams.NumEnvironments = TrainParamsJson->GetIntegerField(TEXT("NumEnvironments"));
    OutTrainParams.MaxAgents = TrainParamsJson->GetIntegerField(TEXT("MaxAgents"));

    return true;
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

void AMecanum_Robot_RLGameModeBase::InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage)
{
    Super::InitGame(MapName, Options, ErrorMessage);
    UE_LOG(LOG_MECANUM_ROBOT_RL, Log, TEXT("RL session has started: % %"), *MapName, *Options);
}
