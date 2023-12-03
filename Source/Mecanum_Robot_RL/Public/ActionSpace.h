#pragma once

#include "CoreMinimal.h"
#include "ActionSpace.generated.h"

USTRUCT(BlueprintType)
struct FContinuousActionSpec
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Continuous Action")
    float Low;

    UPROPERTY(BlueprintReadWrite, Category = "Continuous Action")
    float High;
};

USTRUCT(BlueprintType)
struct FDiscreteActionSpec
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Discrete Action")
    int32 NumChoices;
};

UCLASS()
class MECANUM_ROBOT_RL_API UActionSpace : public UObject
{
    GENERATED_BODY()

public:
    UActionSpace();

    UPROPERTY(BlueprintReadWrite, Category = "Continuous Action")
    TArray<FContinuousActionSpec> ContinuousActions;

    UPROPERTY(BlueprintReadWrite, Category = "Discrete Action")
    TArray<FDiscreteActionSpec> DiscreteActions;

    // Initialize the action space with arrays of continuous and discrete actions
    void Init(
        const TArray<FContinuousActionSpec>& InContinuousActions,
        const TArray<FDiscreteActionSpec>& InDiscreteActions
    );

    int TotalActions();
};
