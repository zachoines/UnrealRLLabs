#pragma once

#include "CoreMinimal.h"
#include "ActionSpace.h"
#include "RLTypes.generated.h"

USTRUCT(BlueprintType)
struct UNREALRLLABS_API FTrainParams
{
    GENERATED_USTRUCT_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int BufferSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int BatchSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int NumEnvironments;

    // For multi-agent environments
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int MaxAgents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int MinAgents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int AgentsResetFrequency;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int ActionRepeat;
};

USTRUCT(BlueprintType)
struct UNREALRLLABS_API FEnvInfo
{
    GENERATED_USTRUCT_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    UActionSpace* ActionSpace;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    int StateSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    bool IsMultiAgent;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    int MaxAgents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    int SingleAgentObsSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Info")
    int EnvID;
};

USTRUCT(BlueprintType)
struct UNREALRLLABS_API FAction
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Action")
    TArray<float> Values;
};

USTRUCT(BlueprintType)
struct UNREALRLLABS_API FState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "State")
    TArray<float> Values;
};

USTRUCT(BlueprintType)
struct UNREALRLLABS_API FBaseInitParams
{
    GENERATED_USTRUCT_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector Location; // spawn location 

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int NumAgents;

    /*UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MaxAgents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    int MinAgents;*/

};




