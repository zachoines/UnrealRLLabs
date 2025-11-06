#pragma once

#include "CoreMinimal.h"
#include "ActionSpace.h"
#include "EnvironmentConfig.h"
#include "RLTypes.generated.h"

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
    FVector Location;

    UPROPERTY()
    UEnvironmentConfig* EnvConfig = nullptr;
};